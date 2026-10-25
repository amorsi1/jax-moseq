import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax_moseq.utils.autoregression import get_nlags, ar_log_likelihood
from jax_moseq.utils import mixed_map, mixed_map_jit
from jax_moseq.models.arhmm.gibbs import session_log_likelihoods
from dynamax.hidden_markov_model.inference import hmm_filter
from functools import partial

na = jnp.newaxis


def discrete_stateseq_log_prob(z, pi, **kwargs):
    """
    Calculate the log probability of a discrete state sequence
    at each timestep given a matrix of transition probabilities.

    Parameters
    ----------
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_pz : jax array of shape (..., T - 1)
        Log probability of ``z``.
    """
    return jnp.log(pi[z[..., :-1], z[..., 1:]])


def continuous_stateseq_log_prob(x, z, Ab, Q, **kwargs):
    """
    Calculate the log probability of the trajectory ``x`` at each time
    step, given switching autoregressive (AR) parameters

    Parameters
    ----------
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_px : jax array of shape (..., T - n_lags)
        Log probability of ``x``.
    """
    return ar_log_likelihood(x, (Ab[z], Q[z]))


@jax.jit
def log_joint_likelihood(x, mask, z, pi, Ab, Q, **kwargs):
    """
    Calculate the total log probability for each latent state

    Parameters
    ----------
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    mask : jax array
        Binary indicator for which data points are valid.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    ll = {}

    log_pz = discrete_stateseq_log_prob(z, pi)
    log_px = continuous_stateseq_log_prob(x, z, Ab, Q)

    nlags = get_nlags(Ab)
    ll["z"] = (log_pz * mask[..., nlags + 1 :]).sum()
    ll["x"] = (log_px * mask[..., nlags:]).sum()
    return ll


def model_likelihood(data, states, params, hypparams=None, **kwargs):
    """
    Convenience function that invokes :py:func:`jax_moseq.models.arhmm.log_prob.log_joint_likelihood`.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict, optional
        Values for each group of hyperparameters.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    return log_joint_likelihood(**data, **states, **params)


def state_cross_likelihoods(params, states, mask, **kwargs):
    """
    Calculate log likelihoods of frames assigned to each state,
    given the dynamics of each other state. See page 33 of the
    supplement (Wiltchsko, 2015) for a formal definition.
    """
    x, Ab, Q, z_all = jax.device_put(
        (states["x"], params["Ab"], params["Q"], states["z"])
    )
    n_states = Ab.shape[0]
    nlags = mask.shape[1] - z_all.shape[1]
    valid = jnp.asarray(mask[:, nlags:] > 0)

    def summed_by_state(x_i, z_i, valid_i):
        # (n_states, n_states): log likelihood under each state, summed over
        # the valid frames assigned to each state
        lls = session_log_likelihoods(x_i, Ab, Q)
        onehot = jax.nn.one_hot(z_i, n_states, dtype=lls.dtype) * valid_i[:, None]
        return onehot.T @ lls

    sums = np.asarray(mixed_map(summed_by_state)(x, z_all, valid).sum(0))

    z = np.asarray(states["z"])[np.asarray(valid)]
    changepoints = np.diff(z).nonzero()[0] + 1
    counts = np.bincount(z[changepoints])

    cross_likelihoods = np.zeros((n_states, n_states))
    for j in range(n_states):
        ll = sums[j]
        cross_likelihoods[j] = (ll - ll[j]) / (counts[j] + 1e-6)
    return cross_likelihoods


@mixed_map_jit
def marginal_log_likelihood(
    mask, x, Ab, Q, pi, *, mixed_map_iters, mixed_map_gpus, **kwargs
):
    """Marginal log likelihood of continuous latents given model parameters.

    Parameters
    ----------
    mask : jax array
        Binary indicator for which data points are valid.
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.

    Returns
    -------
    ml : float
        Marginal log likelihood.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states

    def get_mll(x_i, mask_i):
        lls = session_log_likelihoods(x_i, Ab, Q) * mask_i[nlags:, None]
        return hmm_filter(initial_distribution, pi, lls).marginal_loglik.sum()

    mlls = mixed_map(get_mll, iters=mixed_map_iters, gpus=mixed_map_gpus)(x, mask)
    return mlls.sum()
