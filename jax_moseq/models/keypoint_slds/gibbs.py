import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from jax_moseq.utils import mixed_map_jit
from jax_moseq.utils.kalman import kalman_sample
from jax_moseq.utils.distributions import sample_vonmises_fisher

from jax_moseq.models import arhmm, slds
from jax_moseq.models.slds.gibbs import (
    _map_sessions,
    _resample_x,
    _resample_obs_variance_from_stats,
)
from jax_moseq.models.keypoint_slds.alignment import (
    to_vanilla_slds,
    estimate_coordinates,
    estimate_aligned,
    apply_rotation,
    vector_to_angle,
)

na = jnp.newaxis


@mixed_map_jit(static_argnames=("parallel_message_passing",))
def resample_continuous_stateseqs(
    seed,
    Y,
    mask,
    v,
    h,
    s,
    z,
    Cd,
    sigmasq,
    Ab,
    Q,
    jitter=1e-3,
    parallel_message_passing=True,
    mixed_map_iters=None,
    mixed_map_gpus=None,
    **kwargs
):
    """
    Resamples the latent trajectories ``x``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
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
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    """
    return _resample_x(
        seed,
        mask,
        z,
        Ab,
        Q,
        Cd,
        sigmasq,
        (Y, s, v, h),
        lambda Cd, sigmasq, Y, s, v, h: to_vanilla_slds(Y, v, h, s, Cd, sigmasq),
        jitter,
        parallel_message_passing,
        mixed_map_iters,
        mixed_map_gpus,
    )


def _weighted_sqerr_one_session(Y, x, v, h, mask, s, Cd):
    """Per-session sum over time of ``sqerr / s``, shape (k,)."""
    return (compute_squared_error(Y, x, v, h, Cd, mask) / s).sum(0)


@mixed_map_jit
def resample_obs_variance(
    seed,
    Y,
    mask,
    Cd,
    x,
    v,
    h,
    s,
    nu_sigma,
    sigmasq_0,
    mixed_map_iters=None,
    mixed_map_gpus=None,
    **kwargs
):
    """
    Resample the observation variance ``sigmasq``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    nu_sigma : float
        Chi-squared degrees of freedom in sigmasq.
    sigmasq_0 : float
        Scaled inverse chi-squared scaling parameter for sigmasq.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    sigmasq : jax_array of shape k
        Unscaled noise.
    """
    S_y = _map_sessions(
        _weighted_sqerr_one_session,
        Y,
        x,
        v,
        h,
        mask,
        s,
        mixed_map_iters=mixed_map_iters,
        mixed_map_gpus=mixed_map_gpus,
        Cd=Cd,
    ).sum(0)
    return _resample_obs_variance_from_stats(seed, mask, S_y, nu_sigma, sigmasq_0)


def _resample_scales_one_session(seed, Y, x, v, h, s_0, *, Cd, sigmasq, nu_s):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)


@mixed_map_jit
def resample_scales(
    seed,
    Y,
    x,
    v,
    h,
    Cd,
    sigmasq,
    nu_s,
    s_0,
    mixed_map_iters=None,
    mixed_map_gpus=None,
    **kwargs
):
    """
    Resample the scale values ``s``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    s_0 : scalar or jax array broadcastable to ``Y``
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    s : jax array of shape (N, T, k)
        Noise scales.
    """
    N = Y.shape[0]
    # Per-session seeds, so (unlike a single full-array gamma draw) the stream
    # differs from earlier versions: statistically equivalent and chunk-size
    # invariant.
    session_args = [jr.split(seed, N), Y, x, v, h]
    # ``s_0`` is mapped over sessions if it has the shape of ``s`` (N, T, k);
    # anything else (scalar, (k,), (T, k), ...) is shared by all sessions.
    shared = {}
    if jnp.ndim(s_0) == Y.ndim - 1 and jnp.shape(s_0)[0] == N:
        session_args.append(s_0)
    else:
        shared["s_0"] = s_0
    return _map_sessions(
        _resample_scales_one_session,
        *session_args,
        mixed_map_iters=mixed_map_iters,
        mixed_map_gpus=mixed_map_gpus,
        Cd=Cd,
        sigmasq=sigmasq,
        nu_s=nu_s,
        **shared,
    )


@jax.jit
def compute_squared_error(Y, x, v, h, Cd, mask=None):
    """
    Computes the squared error between model predicted
    and true observations.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    mask : jax array, optional
        Binary indicator for valid frames.

    Returns
    ------
    sqerr : jax array of shape (..., k)
        Squared error between model predicted and
        true observations.
    """
    Y_est = estimate_coordinates(x, v, h, Cd)
    sqerr = ((Y - Y_est) ** 2).sum(-1)
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


def _heading_mean_direction_one_session(Y, x, v, s, *, Cd, sigmasq):
    k = Y.shape[-2]

    Y_bar = estimate_aligned(x, Cd, k)
    Y_cent = Y - v[..., na, :]
    variance = s * sigmasq

    # [(t, k, d, na) * (t, k, na, d) / (t, k, na, na)] -> (t, d, d)
    S = (Y_bar[..., :2, na] * Y_cent[..., na, :2] / variance[..., na, na]).sum(-3)

    kappa_cos = S[..., 0, 0] + S[..., 1, 1]
    kappa_sin = S[..., 0, 1] - S[..., 1, 0]
    return jnp.stack([kappa_cos, kappa_sin], axis=-1)


@mixed_map_jit
def resample_heading(
    seed,
    Y,
    x,
    v,
    s,
    Cd,
    sigmasq,
    mixed_map_iters=None,
    mixed_map_gpus=None,
    **kwargs
):
    """
    Resample the heading angles ``h``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    s : jax array of shape (N, T, k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    h : jax array of shape (N, T)
        Heading angles.
    """
    mean_direction = _map_sessions(
        _heading_mean_direction_one_session,
        Y,
        x,
        v,
        s,
        mixed_map_iters=mixed_map_iters,
        mixed_map_gpus=mixed_map_gpus,
        Cd=Cd,
        sigmasq=sigmasq,
    )
    # A single draw over the assembled (N, T, 2) array keeps ``h`` identical
    # to the unchunked computation.
    sampled_direction = sample_vonmises_fisher(seed, mean_direction)
    h = vector_to_angle(sampled_direction)
    return h


def _resample_location_one_session(
    seed, Y, mask, x, h, s, *, Cd, sigmasq, sigmasq_loc, parallel
):
    k, d = Y.shape[-2:]

    Y_rot = apply_rotation(estimate_aligned(x, Cd, k), h)

    variance = s * sigmasq
    gammasq = 1 / (1 / variance).sum(-1, keepdims=True)

    mu = jnp.einsum("...tkd, ...tk->...td", Y - Y_rot, gammasq / variance)

    # Apply Kalman filter to get smooth headings
    # TODO Parameterize these distributional hyperparameter
    m0 = jnp.zeros(d)
    S0 = jnp.eye(d) * 1e4
    A = jnp.eye(d)[na]
    B = jnp.zeros(d)[na]
    Q = jnp.eye(d)[na] * sigmasq_loc
    C = jnp.eye(d)
    D = jnp.zeros(d)
    R = jnp.repeat(gammasq, d, axis=-1)
    zz = jnp.zeros_like(mask[1:], dtype=int)

    masked_dynamics_noise = sigmasq_loc * 10
    masked_obs_noise = sigmasq.max() * 10

    masked_dynamics_params = {
        "weights": jnp.eye(d),
        "bias": jnp.zeros(d),
        "cov": jnp.eye(d) * masked_dynamics_noise,
    }

    masked_obs_noise_diag = jnp.ones(d) * masked_obs_noise

    return kalman_sample(
        seed,
        mu,
        mask,
        zz,
        m0,
        S0,
        A,
        B,
        Q,
        C,
        D,
        R,
        masked_dynamics_params,
        masked_obs_noise_diag,
        parallel=parallel,
    )


@mixed_map_jit(static_argnames=("parallel_message_passing",))
def resample_location(
    seed,
    Y,
    mask,
    x,
    h,
    s,
    Cd,
    sigmasq,
    sigmasq_loc,
    parallel_message_passing=True,
    mixed_map_iters=None,
    mixed_map_gpus=None,
    **kwargs
):
    """
    Resample the centroid positions ``v``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    sigmasq_loc : float
        Assumed variance in centroid displacements.
    parallel_message_passing : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    v : jax array of shape (N, T, d)
        Centroid positions.
    """
    return _map_sessions(
        _resample_location_one_session,
        jr.split(seed, mask.shape[0]),
        Y,
        mask,
        x,
        h,
        s,
        mixed_map_iters=mixed_map_iters,
        mixed_map_gpus=mixed_map_gpus,
        Cd=Cd,
        sigmasq=sigmasq,
        sigmasq_loc=sigmasq_loc,
        parallel=parallel_message_passing,
    )


def resample_model(
    data,
    seed,
    states,
    params,
    hypparams,
    noise_prior,
    ar_only=False,
    states_only=False,
    resample_global_noise_scale=False,
    resample_local_noise_scale=True,
    fix_heading=False,
    verbose=False,
    jitter=1e-3,
    parallel_message_passing=False,
    **kwargs
):
    """
    Resamples the Keypoint SLDS model given the hyperparameters,
    data, noise prior, current states, and current parameters.

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
    noise_prior : scalar or jax array broadcastable to ``s``
        Prior on noise scale.
    ar_only : bool, default=False
        Whether to restrict sampling to ARHMM components.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    resample_global_noise_scale : bool, default=False
        Whether to resample the global noise scales (``sigmasq``)
    resample_local_noise_scale : bool, default=True
        Whether to resample the local noise scales (``s``)
    fix_heading : bool, default=False
        Whether to exclude ``h`` from resampling.
    jitter : float, default=1e-3
        Amount to boost the diagonal of the covariance matrix
        during backward-sampling of the continuous states.
    verbose : bool, default=False
        Whether to print progress info during resampling.
    parallel_message_passing : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    model = arhmm.resample_model(
        data, seed, states, params, hypparams, states_only, verbose=verbose
    )
    if ar_only:
        model["noise_prior"] = noise_prior
        return model

    seed = model["seed"]
    params = model["params"].copy()
    states = model["states"].copy()

    if (not states_only) and resample_global_noise_scale:
        if verbose:
            print("Resampling sigmasq (global noise scales)")
        params["sigmasq"] = resample_obs_variance(
            seed,
            **data,
            **states,
            **params,
            s_0=noise_prior,
            **hypparams["obs_hypparams"]
        )

    if verbose:
        print("Resampling x (continuous latent states)")
    states["x"] = resample_continuous_stateseqs(
        seed,
        **data,
        **states,
        **params,
        jitter=jitter,
        parallel_message_passing=parallel_message_passing
    )

    if not fix_heading:
        if verbose:
            print("Resampling h (heading)")
        states["h"] = resample_heading(seed, **data, **states, **params)

    if verbose:
        print("Resampling v (location)")
    states["v"] = resample_location(
        seed,
        **data,
        **states,
        **params,
        **hypparams["cen_hypparams"],
        parallel_message_passing=parallel_message_passing,
    )

    if resample_local_noise_scale:
        if verbose:
            print("Resampling s (local noise scales)")
        states["s"] = resample_scales(
            seed,
            **data,
            **states,
            **params,
            s_0=noise_prior,
            **hypparams["obs_hypparams"]
        )

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
        "noise_prior": noise_prior,
    }
