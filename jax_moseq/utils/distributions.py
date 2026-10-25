import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from jax_moseq.utils import convert_data_precision

na = jnp.newaxis

# (rel_floor, abs_floor) of the PSD projection applied before Cholesky
# factorizations, or None (default) to factor the matrices as given.
_STABLE_CHOLESKY = None


def get_stable_cholesky():
    """Get ``(rel_floor, abs_floor)`` if the PSD projection before Cholesky
    factorizations is enabled, otherwise ``None``."""
    return _STABLE_CHOLESKY


def set_stable_cholesky(
    enabled: bool, rel_floor: float = 0.2, abs_floor: float = 1e-3
) -> None:
    """Enable or disable a float32-stabilizing PSD projection in
    :py:func:`sample_invwishart` and :py:func:`sample_mn`.

    In float32 the matrices that get factorized can have slightly negative
    eigenvalues (e.g. from cancellation), which makes ``cholesky`` return NaN.
    When enabled, they are symmetrized and their eigenvalues are clipped below
    at ``max(abs_floor, rel_floor * largest eigenvalue)`` first. Off by default.

    The setting is read when functions are traced, so changing it clears the
    jit caches to make already-compiled callers pick it up.
    """
    global _STABLE_CHOLESKY
    new = (rel_floor, abs_floor) if enabled else None
    if new != _STABLE_CHOLESKY:
        _STABLE_CHOLESKY = new
        jax.clear_caches()


def _psd_clip(S, rel_floor, abs_floor):
    S = (S + jnp.swapaxes(S, -1, -2)) / 2
    w, v = jnp.linalg.eigh(S)
    floor = jnp.maximum(abs_floor, rel_floor * jnp.max(w, axis=-1, keepdims=True))
    w = jnp.clip(w, floor, None)
    return (v * w[..., None, :]) @ jnp.swapaxes(v, -1, -2)


def _cholesky(S):
    if _STABLE_CHOLESKY is not None:
        S = _psd_clip(S, *_STABLE_CHOLESKY)
    return jnp.linalg.cholesky(S)


def sample_vonmises(seed, theta, kappa):
    return tfd.VonMises(theta, kappa).sample(seed=seed)


def sample_vonmises_fisher(seed, direction):
    kappa = jnp.sqrt((direction**2).sum(-1))
    direction = direction / kappa[..., na]
    return tfd.VonMisesFisher(direction, kappa).sample(seed=seed)


def sample_gamma(seed, a, b):
    return jr.gamma(seed, a) / b


def sample_inv_gamma(seed, a, b):
    return 1 / sample_gamma(seed, a, b)


def sample_scaled_inv_chi2(seed, degs, variance):
    return sample_inv_gamma(seed, degs / 2, degs * variance / 2)


def sample_chi2(seed, degs):
    return jr.gamma(seed, degs / 2) * 2


def sample_mn(seed, M, U, V):
    G = jr.normal(seed, M.shape)
    G = jnp.dot(_cholesky(U), G)
    G = jnp.dot(G, _cholesky(V).T)
    return M + G


def sample_invwishart(seed, S, nu):
    n = S.shape[0]

    chi2_seed, norm_seed = jr.split(seed)
    x = jnp.diag(jnp.sqrt(sample_chi2(chi2_seed, nu - jnp.arange(n))))
    x = x.at[jnp.triu_indices_from(x, 1)].set(jr.normal(norm_seed, (n * (n - 1) // 2,)))
    R = jnp.linalg.qr(x, "r")

    chol = _cholesky(S)

    T = jax.scipy.linalg.solve_triangular(R.T, chol.T, lower=True).T
    return jnp.dot(T, T.T)


def sample_niw(seed, mu, lam, nu, S):
    sigma = sample_invwishart(seed, S, nu)
    mu = jr.multivariate_normal(seed, mu, sigma / lam)
    return mu, sigma


def sample_mniw(seed, nu, S, M, K):
    sigma = sample_invwishart(seed, S, nu)
    A = sample_mn(seed, M, sigma, K)
    return A, sigma


def sample_hmm_stateseq(seed, transition_matrix, log_likelihoods, mask):
    """Sample state sequences in a Markov chain.

    Parameters
    ----------
    seed: jax.random.PRNGKey
        Random seed
    transition_matrix: jax array, shape (num_states, num_states)
        Transition matrix
    log_likelihoods: jax array, shape (num_timesteps, num_states)
        Sequence of log likelihoods of emissions given hidden state and parameters
    mask: jax array, shape (num_timesteps,)
        Sequence indicating whether to use an emission (1) or not (0)

    Returns
    -------
    log_norm: float:
        Posterior marginal log likelihood
    states: jax array, shape (num_timesteps,)
        Sequence of sampled states
    """

    num_states = transition_matrix.shape[0]
    initial_distribution = jnp.ones(num_states) / num_states

    masked_log_likelihoods = log_likelihoods * mask[:, None]
    L, z = hmm_posterior_sample(
        seed, initial_distribution, transition_matrix, masked_log_likelihoods
    )
    z = convert_data_precision(z)
    return L, z
