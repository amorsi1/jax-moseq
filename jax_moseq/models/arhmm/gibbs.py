import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model.inference import (
    hmm_posterior_mode,
    hmm_smoother,
)

from jax_moseq.utils import (
    pad_affine,
    psd_solve,
    psd_inv,
    nan_check,
    mixed_map,
    mixed_map_jit,
)

from jax_moseq.utils import convert_data_precision
from jax_moseq.utils.distributions import sample_mniw, sample_hmm_stateseq
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood,
)
from jax_moseq.utils.transitions import resample_hdp_transitions
from functools import partial

na = jnp.newaxis


def session_log_likelihoods(x: jnp.ndarray, Ab: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """AR log likelihood of one session under every state, (T - nlags, num_states).

    States are processed serially so the (num_states, ...) intermediates of
    ``ar_log_likelihood`` are never alive at once.
    """
    ll = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
    return jnp.moveaxis(ll, 0, -1)


@mixed_map_jit
def resample_discrete_stateseqs(
    seed, x, mask, Ab, Q, pi, *, mixed_map_iters, mixed_map_gpus, **kwargs
):
    """
    Resamples the discrete state sequence ``z``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    """
    nlags = get_nlags(Ab)

    def sample(seed_i, x_i, mask_i):
        lls = session_log_likelihoods(x_i, Ab, Q)
        return sample_hmm_stateseq(seed_i, pi, lls, mask_i.astype(float)[nlags:])[1]

    return mixed_map(sample, iters=mixed_map_iters, gpus=mixed_map_gpus)(
        jr.split(seed, mask.shape[0]), x, mask
    )


@mixed_map_jit
def stateseq_marginals(
    x, mask, Ab, Q, pi, *, mixed_map_iters, mixed_map_gpus, **kwargs
):
    """
    Computes the marginal probability of each discrete state at each time step.

    Parameters
    ----------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z_marginals : jax array of shape (N, T, num_states)
        Marginal probability of each discrete state at each time step.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states

    def smooth(x_i, mask_i):
        lls = session_log_likelihoods(x_i, Ab, Q) * mask_i[nlags:, na]
        return hmm_smoother(initial_distribution, pi, lls).smoothed_probs

    return mixed_map(smooth, iters=mixed_map_iters, gpus=mixed_map_gpus)(x, mask)


@mixed_map_jit
def stateseq_mode(x, mask, Ab, Q, pi, *, mixed_map_iters, mixed_map_gpus, **kwargs):
    """Most probable state sequence at each time step.

    The Viterbi counterpart of :py:func:`stateseq_marginals`: it returns the
    single most probable state sequence rather than the per-frame posterior or a
    draw from it. This is what moseq2-model's ``apply_model`` needs, since
    applying a fitted model to new recordings should be deterministic -- two
    runs over the same data must label it the same way.

    Parameters
    ----------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    z : jax array of shape (N, T - nlags)
        Most probable discrete state at each time step.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states

    def decode(x_i, mask_i):
        lls = session_log_likelihoods(x_i, Ab, Q) * mask_i[nlags:, na]
        return hmm_posterior_mode(initial_distribution, pi, lls)

    z = mixed_map(decode, iters=mixed_map_iters, gpus=mixed_map_gpus)(x, mask)
    return convert_data_precision(z)


@nan_check
@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_ar_params(
    seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs
):
    """
    Resamples the AR parameters ``Ab`` and ``Q``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    nlags : int
        Number of autoregressive lags.
    num_states : int
        Max number of HMM states.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (latent_dim, latent_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (latent_dim, ar_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (ar_dim, ar_dim)
        Matrix normal column scale parameter for Ab.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    """
    seeds = jr.split(seed, num_states)
    mask_flat = mask[..., nlags:].reshape(-1)
    z_flat = z.reshape(-1)
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
    x_out = x[..., nlags:, :].reshape(-1, x.shape[-1])

    def resample_state(args):
        # the per-state mask is built lazily: no (num_states, N * T) array
        seed_k, k = args
        mask_k = mask_flat * (z_flat == k).astype(mask_flat.dtype)
        x_in_m = x_in * mask_k[:, na]
        x_out_m = x_out * mask_k[:, na]
        stats = (x_out_m.T @ x_out, x_out_m.T @ x_in, x_in_m.T @ x_in)
        return _sample_regression_params(
            seed_k, *stats, mask_k.sum(), nu_0, S_0, M_0, K_0
        )

    return jax.lax.map(resample_state, (seeds, jnp.arange(num_states)))


@nan_check
def _sample_regression_params(
    seed, S_out_out, S_out_in, S_in_in, count, nu_0, S_0, M_0, K_0
):
    """Samples (Ab, Q) from the Matrix normal inverse-Wishart posterior given
    the masked sufficient statistics and the number of frames used."""
    K_0_inv = psd_inv(K_0)
    K_n_inv = K_0_inv + S_in_in

    K_n = psd_inv(K_n_inv)
    M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T

    S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
    return sample_mniw(seed, nu_0 + count, S_n, M_n, K_n)


def resample_model(
    data, seed, states, params, hypparams, states_only=False, verbose=False, **kwargs
):
    """
    Resamples the ARHMM model given the hyperparameters, data,
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
    states_only : bool, default=False
        Whether to restrict sampling to states.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    seed = jr.split(seed)[1]
    params = params.copy()
    states = states.copy()

    if not states_only:
        if verbose:
            print("Resampling pi (transition matrix)")
        params["betas"], params["pi"] = resample_hdp_transitions(
            seed, **data, **states, **params, **hypparams["trans_hypparams"]
        )

        if verbose:
            print("Resampling Ab,Q (AR parameters)")
        params["Ab"], params["Q"] = resample_ar_params(
            seed, **data, **states, **params, **hypparams["ar_hypparams"]
        )

    if verbose:
        print("Resampling z (discrete latent states)")
    states["z"] = resample_discrete_stateseqs(seed, **data, **states, **params)

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
    }
