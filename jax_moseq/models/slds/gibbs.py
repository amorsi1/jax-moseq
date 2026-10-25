import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from jax_moseq.utils import mixed_map, mixed_map_jit, apply_affine
from jax_moseq.models import arhmm
from jax_moseq.utils.kalman import (
    kalman_sample,
    ar_to_lds_dynamics,
    ar_to_lds_emissions,
)


na = jnp.newaxis


def _map_sessions(fun, *args, mixed_map_iters, mixed_map_gpus, **shared):
    """Apply ``fun(*session_args, **shared)`` to every session (leading axis
    of ``args``) with `mixed_map`, so peak memory follows ``mixed_map_iters``."""
    return mixed_map(
        partial(fun, **shared), iters=mixed_map_iters, gpus=mixed_map_gpus
    )(*args)


def _sample_x_one_session(
    seed,
    mask,
    z,
    *session_data,
    to_vanilla,
    Cd,
    sigmasq,
    A_,
    b_,
    Q_,
    n_lags,
    latent_dim,
    jitter,
    parallel,
):
    """Kalman-sample the latent trajectory of one session."""
    y, s, Cd, sigmasq = to_vanilla(Cd, sigmasq, *session_data)

    # TODO Parameterize these distributional hyperparameter
    m0 = jnp.zeros(latent_dim * n_lags)
    S0 = 10 * jnp.eye(latent_dim * n_lags)  # TODO: hard coded constant 10
    masked_dynamics_noise = 10
    masked_obs_noise = 10

    # Reformat n_lags'th-order AR dynamics in R^D to 1st-order dynamics in R^{DL}.
    # The conditioned initial state (m0_, S0_) is deliberately unused: the
    # sampler gets the plain m0/S0 (as in all previous versions).
    C_, d_, R_, y_, _, _ = ar_to_lds_emissions(Cd, sigmasq * s, y, m0, S0, n_lags)

    # If masked, hold the last state, i.e. set dynamics for "unlagged" states to
    # identity matrix and all other state dynamics to 0
    ar_dim = latent_dim * n_lags
    eye_zero_order = jnp.zeros((ar_dim, ar_dim))
    eye_zero_order = eye_zero_order.at[-latent_dim:, -latent_dim:].set(
        jnp.eye(latent_dim)
    )
    masked_dynamics_params = {
        "weights": eye_zero_order,
        "bias": jnp.zeros(ar_dim),
        "cov": jnp.eye(ar_dim) * masked_dynamics_noise,
    }
    masked_obs_noise_diag = jnp.ones(y.shape[-1]) * masked_obs_noise

    # Shapes of time-varying parameters going into the Kalman sampler are
    #   y_:     (n_timesteps-n_lags+1, obs_dim), corresponding to timesteps  [nlags-1, T)
    #   mask_:  (n_timesteps-n_lags+1,)
    #   z:      (n_timesteps-n_lags,), corresponding to timesteps [n_lags, T)
    #   R_:     (n_timesteps-n_lags+1, obs_dim)
    x = kalman_sample(
        seed,
        y_,
        mask[n_lags - 1 :],
        z,
        m0,
        S0,
        A_,
        b_,
        Q_,
        C_,
        d_,
        R_,
        masked_dynamics_params,
        masked_obs_noise_diag,
        jitter=jitter,
        parallel=parallel,
    )

    # Reformat sampled trajectory back into L'th order AR dynamics in R^D
    if n_lags > 1:
        x = jnp.concatenate(
            [x[0, : (n_lags - 1) * latent_dim].reshape(n_lags - 1, latent_dim),
             x[:, -latent_dim:]],
            axis=0,
        )
    return x


def _resample_x(
    seed,
    mask,
    z,
    Ab,
    Q,
    Cd,
    sigmasq,
    session_data,
    to_vanilla,
    jitter,
    parallel_message_passing,
    mixed_map_iters,
    mixed_map_gpus,
):
    """Shared by the SLDS and keypoint SLDS. ``session_data`` are arrays with a
    leading session axis, and ``to_vanilla(Cd, sigmasq, *session_slices)``
    returns ``(y, s, Cd, sigmasq)`` for one session. It runs inside the
    per-session map so no full-N copy of the observations is made."""
    latent_dim = Ab.shape[1]
    n_lags = Ab.shape[2] // latent_dim
    A_, b_, Q_ = ar_to_lds_dynamics(Ab, Q)  # shared across sessions
    return _map_sessions(
        _sample_x_one_session,
        jr.split(seed, mask.shape[0]),
        mask,
        z,
        *session_data,
        mixed_map_iters=mixed_map_iters,
        mixed_map_gpus=mixed_map_gpus,
        to_vanilla=to_vanilla,
        Cd=Cd,
        sigmasq=sigmasq,
        A_=A_,
        b_=b_,
        Q_=Q_,
        n_lags=n_lags,
        latent_dim=latent_dim,
        jitter=jitter,
        parallel=parallel_message_passing,
    )


@mixed_map_jit(static_argnames=("parallel_message_passing",))
def resample_continuous_stateseqs(
    seed,
    y,
    mask,
    z,
    s,
    Ab,
    Q,
    Cd,
    sigmasq,
    jitter=1e-3,
    parallel_message_passing=True,
    mixed_map_iters=None,
    mixed_map_gpus=None,
    **kwargs
):
    """Resample the latent trajectories `x`.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    y : jax.Array of shape (n_recordings, n_timesteps, obs_dim)
        Observations.
    mask : jax.Array of shape (n_recordings, n_timesteps)
        Binary indicator, 1=valid frames, 0=invalid frames.
    z : jax.Array of shape (n_recordings, n_timesteps-n_lags)
        Discrete state sequences, taking integer values between [0, n_states),
        for timesteps [n_lags, n_timesteps),
    s : jax.Array of shape (n_recordings, n_timesteps, obs_dim)
        Observation noise scales.
    Ab : jax.Array of shape (n_states, latent_dim, ar_dim + 1)
        Autoregressive dynamics and bias, where `ar_dim = latent_dim * n_lags`
    Q : jax.Array of shape (n_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    Cd : jax.Array of shape (obs_dim, latent_dim + 1)
        Affine transform from `latent_dim` to `state_dim`
    sigmasq : jax.Array of shape (obs_dim,)
        Unscaled noise.
    jitter : float, default=1e-3
        Amount to boost the diagonal of the covariance matrix
        during backward-sampling of the continuous states.
    parallel_message_passing : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    x : jax.Array of shape (n_recordings, n_timesteps, latent_dim)
        Posterior sample of latent trajectories.
    """
    return _resample_x(
        seed,
        mask,
        z,
        Ab,
        Q,
        Cd,
        sigmasq,
        (y, s),
        lambda Cd, sigmasq, y, s: (y, s, Cd, sigmasq),
        jitter,
        parallel_message_passing,
        mixed_map_iters,
        mixed_map_gpus,
    )


@mixed_map_jit(static_argnames=("parallel",))
def batched_kalman_sample(
    seeds,
    y_,
    mask_,
    z,
    m0,
    S0,
    A_,
    b_,
    Q_,
    C_,
    d_,
    R_,
    masked_dynamics_params,
    masked_obs_noise_diag,
    jitter,
    parallel,
    mixed_map_iters=None,
    mixed_map_gpus=None,
):
    """Wrapper that applied `mixed_map` to `kalman_sample` for batched inputs."""
    in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
    x = mixed_map(
        partial(kalman_sample, jitter=jitter, parallel=parallel),
        in_axes,
        iters=mixed_map_iters,
        gpus=mixed_map_gpus,
    )(
        seeds,
        y_,
        mask_,
        z,
        m0,
        S0,
        A_,
        b_,
        Q_,
        C_,
        d_,
        R_,
        masked_dynamics_params,
        masked_obs_noise_diag,
    )
    return x


def _weighted_sqerr_one_session(Y, x, mask, s, Cd):
    """Per-session sum over time of ``sqerr / s``, shape (obs_dim,)."""
    return (compute_squared_error(Y, x, Cd, mask) / s).sum(0)


@mixed_map_jit
def resample_obs_variance(
    seed,
    Y,
    mask,
    x,
    s,
    Cd,
    nu_sigma,
    sigmasq_0,
    mixed_map_iters=None,
    mixed_map_gpus=None,
    **kwargs
):
    """
    Resample the observation variance `sigmasq`.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    s : jax array of shape (N, T, obs_dim)
        Noise scales.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    nu_sigma : float
        Chi-squared degrees of freedom in sigmasq.
    sigmasq_0 : float
        Scaled inverse chi-squared scaling parameter for sigmasq.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    """
    S_y = _map_sessions(
        _weighted_sqerr_one_session,
        Y,
        x,
        mask,
        s,
        mixed_map_iters=mixed_map_iters,
        mixed_map_gpus=mixed_map_gpus,
        Cd=Cd,
    ).sum(0)
    return _resample_obs_variance_from_stats(seed, mask, S_y, nu_sigma, sigmasq_0)


@jax.jit
def resample_obs_variance_from_sqerr(
    seed, sqerr, mask, s, nu_sigma, sigmasq_0, **kwargs
):
    """
    Resample the observation variance `sigmasq` using the
    squared error between predicted and true observations.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    sqerr : jax array of shape (N, T, obs_dim)
        Squared error between predicted and true observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    s : jax array of shape (N, T, obs_dim)
        Noise scales.
    nu_sigma : float
        Chi-squared degrees of freedom in sigmasq.
    sigmasq_0 : float
        Scaled inverse chi-squared scaling parameter for sigmasq.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    """
    k = sqerr.shape[-1]
    S_y = (sqerr / s).reshape(-1, k).sum(0)  # (..., k) -> k
    return _resample_obs_variance_from_stats(seed, mask, S_y, nu_sigma, sigmasq_0)


def _resample_obs_variance_from_stats(seed, mask, S_y, nu_sigma, sigmasq_0):
    """Resample `sigmasq` given ``S_y = sum(sqerr / s)`` over all valid frames."""
    degs = nu_sigma + 3 * mask.sum()
    variance = nu_sigma * sigmasq_0 + S_y
    return _resample_spread(seed, degs, variance)


@jax.jit
def resample_scales(seed, Y, x, Cd, sigmasq, nu_s, s_0, **kwargs):
    """
    Resample the scale values `s`.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    s : jax array of shape (N, T, obs_dim)
        Noise scales.
    """
    sqerr = compute_squared_error(Y, x, Cd)
    return resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)


@jax.jit
def resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0, **kwargs):
    """
    Resample the scale values `s` using the squared
    error between predicted and true observations.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    sqerr : jax array of shape (N, T, obs_dim)
        Squared error between predicted and true observations.
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    s : jax array of shape (N, T, obs_dim)
        Per observation noise scales.
    """
    degs = nu_s + 3
    variance = sqerr / sigmasq + s_0 * nu_s
    return _resample_spread(seed, degs, variance)


@jax.jit
def _resample_spread(seed, degs, variance):
    """
    Resample the noise values from the computed
    degrees of freedom and variance.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    degs : scalar
        Chi-squared degrees of freedom.
    variance : jax array
        Variance computed from the data.

    Returns
    ------
    spread : jax array, same shape as `variance`
        Resampled noise values.
    """
    # same as sample_scaled_inv_chi2(seed, degs, variance / degs)
    return variance / jr.gamma(seed, degs / 2, shape=variance.shape) / 2


@jax.jit
def compute_squared_error(Y, x, Cd, mask=None):
    """
    Computes the squared error between model predicted
    and true observations.

    Parameters
    ----------
    Y : jax array of shape (..., obs_dim)
        Observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    mask : jax array of shape (...), optional
        Binary indicator for valid frames.

    Returns
    ------
    sqerr : jax array of shape (..., obs_dim)
        Squared error between model predicted and
        true observations.
    """
    Y_bar = apply_affine(x, Cd)
    sqerr = (Y - Y_bar) ** 2
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


def resample_model(
    data,
    seed,
    states,
    params,
    hypparams,
    ar_only=False,
    states_only=False,
    skip_noise=True,
    parallel_message_passing=False,
    **kwargs
):
    """
    Resamples the SLDS model given the hyperparameters, data,
    current states, and current parameters.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    seed : jr.PRNGKey
        JAX random seed.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    ar_only : bool, default=False
        Whether to restrict sampling to ARHMM components.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    skip_noise : bool, default=True
        Whether to exclude `sigmasq` and `s` from resampling.
    parallel_message_passing : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    model = arhmm.resample_model(data, seed, states, params, hypparams, states_only)
    if ar_only:
        return model

    seed = model["seed"]
    params = model["params"].copy()
    states = model["states"].copy()

    if not (states_only or skip_noise):
        params["sigmasq"] = resample_obs_variance(
            seed, **data, **states, **params, **hypparams["obs_hypparams"]
        )

    states["x"] = resample_continuous_stateseqs(
        seed,
        **data,
        **states,
        **params,
        parallel_message_passing=parallel_message_passing
    )

    if not skip_noise:
        states["s"] = resample_scales(
            seed, **data, **states, **params, **hypparams["obs_hypparams"]
        )

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
    }
