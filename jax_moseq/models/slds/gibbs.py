import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from functools import partial
from jax_moseq.utils import mixed_map, mixed_map_jit, apply_affine
from jax_moseq.utils.precision import full_float32_matmuls
from jax_moseq.models import arhmm
from jax_moseq.utils.kalman import (
    kalman_sample,
    ar_to_lds_dynamics,
    ar_to_lds_emissions,
)


na = jnp.newaxis


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
    return _resample_continuous_stateseqs(
        seed,
        y,
        mask,
        z,
        s,
        Ab,
        Q,
        Cd,
        sigmasq,
        jitter,
        parallel_message_passing,
    )


@mixed_map_jit(static_argnames=("parallel_message_passing",))
def _resample_continuous_stateseqs(
    seed,
    y,
    mask,
    z,
    s,
    Ab,
    Q,
    Cd,
    sigmasq,
    jitter,
    parallel_message_passing,
    mixed_map_iters=None,
    mixed_map_gpus=None,
):
    return sample_continuous_stateseqs_per_session(
        seed,
        mask,
        z,
        (y, s),
        lambda y, s: (y, s, Cd, sigmasq),
        Ab,
        Q,
        y.shape[-1],
        jitter,
        parallel_message_passing,
        mixed_map_iters,
        mixed_map_gpus,
    )


def sample_continuous_stateseqs_per_session(
    seed,
    mask,
    z,
    session_args,
    to_obs,
    Ab,
    Q,
    obs_dim,
    jitter,
    parallel,
    iters,
    gpus,
):
    """Sample `x` with every per-session step inside one `mixed_map`.

    `to_obs(*args)` maps one session's slice of the arrays in `session_args`
    to `(y, s, Cd, sigmasq)`, so that anything derived from the full-N inputs
    (e.g. the keypoint alignment) is only ever formed for the sessions in the
    current chunk. Returns `x` of shape (n_recordings, n_timesteps, latent_dim).
    """
    n_recordings, latent_dim = mask.shape[0], Ab.shape[1]
    n_lags = Ab.shape[2] // latent_dim

    # TODO Parameterize these distributional hyperparameter
    m0 = jnp.zeros(latent_dim * n_lags)
    S0 = 10 * jnp.eye(latent_dim * n_lags)  # TODO: hard coded constant 10
    masked_dynamics_noise = 10
    masked_obs_noise = 10

    # Reformat n_lags'th-order AR dynamics in R^D to 1st-order dynamics in R^{DL}
    A_, b_, Q_ = ar_to_lds_dynamics(Ab, Q)

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
    masked_obs_noise_diag = jnp.ones(obs_dim) * masked_obs_noise

    # Shapes going into the Kalman sampler, per recording:
    #   ys: (n_timesteps-n_lags+1, obs_dim), corresponding to timesteps [nlags-1, T)
    #   mask: (n_timesteps-n_lags+1,)
    #   zs: (n_timesteps-n_lags,), corresponding to timesteps [n_lags, T)
    #   Rs: (n_timesteps-n_lags+1, obs_dim)
    def sample_one(seed, mask, z, *args):
        y, s, Cd, sigmasq = to_obs(*args)
        # the (m0_, S0_) returned here are not used, as upstream
        C_, d_, R_, y_, _, _ = ar_to_lds_emissions(
            Cd, sigmasq * s, y, m0, S0, n_lags
        )
        return kalman_sample(
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

    x = mixed_map(sample_one, iters=iters, gpus=gpus)(
        jr.split(seed, n_recordings), mask, z, *session_args
    )

    # Reformat sampled trajectories back into L'th order AR dynamics in R^D
    if n_lags > 1:
        x = jnp.concatenate(
            [
                x[:, 0, : (n_lags - 1) * latent_dim].reshape(
                    -1, n_lags - 1, latent_dim
                ),
                x[:, :, -latent_dim:],
            ],
            axis=1,
        )
    return x


@partial(jax.jit, static_argnames=("parallel",))
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
):
    """Wrapper that applied `mixed_map` to `kalman_sample` for batched inputs."""
    in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
    x = mixed_map(
        partial(kalman_sample, jitter=jitter, parallel=parallel),
        in_axes,
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
    S_y = sum_weighted_sqerr(
        lambda Y, x, mask: compute_squared_error(Y, x, Cd, mask),
        (Y, x, mask),
        s,
        mixed_map_iters,
        mixed_map_gpus,
    )
    return resample_obs_variance_from_sums(seed, S_y, mask, nu_sigma, sigmasq_0)


def sum_weighted_sqerr(sqerr_fn, session_args, s, iters, gpus):
    """Sum of `sqerr / s` over sessions and time, computed per session.

    `sqerr_fn(*args)` maps one session's slice of the arrays in `session_args`
    to its squared error (T, k); `s` has shape (N, T, k). Returns shape k.
    """
    k = s.shape[-1]
    per_session = mixed_map(
        lambda s, *args: (sqerr_fn(*args) / s).reshape(-1, k).sum(0),
        iters=iters,
        gpus=gpus,
    )
    return per_session(s, *session_args).sum(0)


@jax.jit
def resample_obs_variance_from_sums(seed, S_y, mask, nu_sigma, sigmasq_0):
    degs = nu_sigma + 3 * mask.sum()
    variance = nu_sigma * sigmasq_0 + S_y
    return _resample_spread(seed, degs, variance)


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
    return resample_obs_variance_from_sums(seed, S_y, mask, nu_sigma, sigmasq_0)


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
    # same as sample_scaled_inv_chi2(seed, degs, variance / degs). TFP's gamma
    # sampler is much faster than jr.gamma on CPU; it samples in its
    # parameters' dtype, so they are given in the variance's.
    a = jnp.asarray(degs / 2, variance.dtype)
    gamma = tfd.Gamma(a, jnp.ones((), variance.dtype))
    return variance / gamma.sample(variance.shape, seed=seed) / 2


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


@full_float32_matmuls
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
