"""Tests for the per-session (``mixed_map``) AR-HMM steps.

The ``_ref_*`` functions are the upstream implementations (git 944322a) that
compute every session's log likelihoods before the HMM step. The public
functions must agree with them for any ``set_mixed_map_iters`` value, and must
not hold a full (N, T - nlags, num_states) log-likelihood array.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax

if os.environ.get("ARHMM_TEST_F32") != "1":
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dynamax.hidden_markov_model.inference import hmm_posterior_mode  # noqa: E402

from jax_moseq.models.arhmm.gibbs import (  # noqa: E402
    resample_discrete_stateseqs,
    stateseq_marginals,
    stateseq_mode,
)
from jax_moseq.models.arhmm.log_prob import (  # noqa: E402
    marginal_log_likelihood,
    state_cross_likelihoods,
)
from jax_moseq.utils import (  # noqa: E402
    convert_data_precision,
    get_mixed_map_iters,
    set_mixed_map_iters,
)
from jax_moseq.utils.autoregression import (  # noqa: E402
    ar_log_likelihood,
    ar_log_likelihoods,
    get_nlags,
)
from jax_moseq.utils.distributions import (  # noqa: E402
    hmm_marginal_log_likelihood,
    hmm_sampler_uses_log_space,
    hmm_smoothed_marginals,
    sample_hmm_stateseq_from_likelihoods,
    sample_hmm_stateseq_from_log_likelihoods,
)

na = jnp.newaxis


@pytest.fixture(autouse=True)
def restore_iters():
    old = get_mixed_map_iters()
    yield
    set_mixed_map_iters(old)


# ---------------------------------------------------------------- helpers


def make_model(n, t, s, d, nlags, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    ar_dim = d * nlags + 1
    x = np.cumsum(rng.standard_normal((n, t, d)), axis=1) * 0.3
    mask = np.ones((n, t))
    mask[:, 10:13] = 0
    mask[::2, -3:] = 0
    Ab = 0.3 * rng.standard_normal((s, d, ar_dim))
    A = rng.standard_normal((s, d, d))
    Q = np.einsum("sij,skj->sik", A, A) * 0.1 + 0.2 * np.eye(d)
    pi = rng.dirichlet(np.ones(s) * 2, size=s)
    return {
        k: jnp.asarray(v, dtype)
        for k, v in dict(x=x, mask=mask, Ab=Ab, Q=Q, pi=pi).items()
    }


def model_args(m):
    return m["x"], m["mask"], m["Ab"], m["Q"], m["pi"]


def iters_list(n):
    return [1, 2, 3, n, n + 4]


def assert_close(a, b, rtol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-12)


# ------------------------------------------------- reference implementations


def _ref_masked_lls(x, mask, Ab, Q):
    """(N, T - nlags, num_states), all sessions at once (upstream 944322a)."""
    lls = jax.lax.map(lambda x: ar_log_likelihoods(x, Ab, Q), x)
    return lls * mask[:, get_nlags(Ab) :, na]


def _ref_marginals(x, mask, Ab, Q, pi):
    lls = _ref_masked_lls(x, mask, Ab, Q)
    return jax.vmap(lambda ll: hmm_smoothed_marginals(pi, ll))(lls)


def _ref_mode(x, mask, Ab, Q, pi):
    lls = jax.lax.map(lambda a: ar_log_likelihood(x, a), (Ab, Q))
    lls = jnp.moveaxis(lls, 0, -1) * mask[:, get_nlags(Ab) :, na]
    init = jnp.ones(pi.shape[0]) / pi.shape[0]
    return convert_data_precision(
        jax.vmap(lambda ll: hmm_posterior_mode(init, pi, ll))(lls)
    )


def _ref_mll(x, mask, Ab, Q, pi):
    lls = _ref_masked_lls(x, mask, Ab, Q)
    return jax.vmap(lambda ll: hmm_marginal_log_likelihood(pi, ll))(lls).sum()


def _ref_sample(seed, x, mask, Ab, Q, pi):
    nlags = get_nlags(Ab)
    seeds = jr.split(seed, mask.shape[0])
    lls = _ref_masked_lls(x, mask.astype(float), Ab, Q)  # same masking as upstream
    if hmm_sampler_uses_log_space(lls.dtype):
        f = jax.vmap(sample_hmm_stateseq_from_log_likelihoods, in_axes=(0, na, 0))
        return f(seeds, pi, lls)[1]
    ll_max = lls.max(axis=-1)
    probs = jnp.exp(lls - ll_max[..., na])
    f = jax.vmap(sample_hmm_stateseq_from_likelihoods, in_axes=(0, na, 0, 0))
    return f(seeds, pi, probs, ll_max)[1]


def _ref_cross(params, states, mask):
    """Upstream numpy implementation of state_cross_likelihoods."""
    x, Ab, Q = states["x"], params["Ab"], params["Q"]
    lls = jax.lax.map(lambda a: ar_log_likelihood(x, a), (Ab, Q))
    nlags = mask.shape[1] - lls.shape[2]
    lls = np.moveaxis(np.asarray(lls), 0, 2)[mask[:, nlags:] > 0]
    z = states["z"][mask[:, nlags:] > 0]
    cp = np.diff(z).nonzero()[0] + 1
    counts = np.bincount(z[cp])
    n_states = lls.shape[1]
    out = np.zeros((n_states, n_states))
    for j in range(n_states):
        ll = lls[z == j].sum(0)
        out[j] = (ll - ll[j]) / (counts[j] + 1e-6)
    return out


# --------------------------------------------------------- 1. parity w/ ref

CASES = [(7, 1), (7, 3), (12, 1), (12, 3)]  # prime and composite N, nlags 1 and 3


@pytest.mark.parametrize("n,nlags", CASES)
def test_hmm_ops_match_reference(n, nlags):
    m = make_model(n, 50, 4, 2, nlags, seed=n + nlags)
    args = model_args(m)
    exp_marg, exp_mode, exp_mll = (
        _ref_marginals(*args),
        _ref_mode(*args),
        _ref_mll(*args),
    )
    for k in iters_list(n):
        set_mixed_map_iters(k)
        assert_close(stateseq_marginals(*args), exp_marg)
        np.testing.assert_array_equal(stateseq_mode(*args), exp_mode)
        mll = marginal_log_likelihood(m["mask"], m["x"], m["Ab"], m["Q"], m["pi"])
        assert_close(mll, exp_mll)


@pytest.mark.parametrize("n,nlags", [(7, 1), (12, 3)])
def test_state_cross_likelihoods_match_reference(n, nlags):
    s, t = 4, 50
    m = make_model(n, t, s, 2, nlags, seed=3)
    z = (np.arange(t - nlags)[None] // 5 + np.arange(n)[:, None]) % s
    params, states = {"Ab": m["Ab"], "Q": m["Q"]}, {"x": m["x"], "z": z}
    mask = np.asarray(m["mask"])
    expected = _ref_cross(params, states, mask)
    for k in (1, 3, n + 4):
        set_mixed_map_iters(k)
        assert_close(state_cross_likelihoods(params, states, mask), expected)


# ------------------------------------------------ 2. discrete state sampling


@pytest.mark.parametrize("n,nlags", CASES)
def test_resample_stateseqs_matches_reference_for_all_iters(n, nlags):
    s, t = 5, 50
    m = make_model(n, t, s, 3, nlags, seed=1)
    seed = jr.PRNGKey(11)
    args = model_args(m)
    expected = _ref_sample(seed, *args)
    for k in iters_list(n):
        set_mixed_map_iters(k)
        z = resample_discrete_stateseqs(seed, *args)
        assert z.shape == (n, t - nlags) and z.dtype == expected.dtype
        assert 0 <= int(z.min()) and int(z.max()) < s
        np.testing.assert_array_equal(z, expected)  # split(seed, N): per session


def _check_posterior(n_seeds=600):
    """Empirical state frequencies vs exact marginals, in the current dtype."""
    n, t, s = 2, 8, 2
    m = make_model(n, t, s, 2, 1, seed=5)  # d >= 2: get_nlags = ar_dim // d
    args = model_args(m)
    set_mixed_map_iters(2)
    marg = np.asarray(stateseq_marginals(*args))
    seeds = jr.split(jr.PRNGKey(0), n_seeds)
    zs = np.asarray(jax.vmap(lambda sd: resample_discrete_stateseqs(sd, *args))(seeds))
    for sess, frame in [(0, 3), (1, 5)]:
        emp = (zs[:, sess, frame] == 1).mean()
        assert abs(emp - marg[sess, frame, 1]) < 0.08, (emp, marg[sess, frame])


def test_resample_stateseqs_posterior_float64():
    assert resample_discrete_stateseqs.__name__  # probability-space branch
    assert not hmm_sampler_uses_log_space(jnp.zeros(()).dtype)
    _check_posterior()


# ------------------------------------------------------ float32 subprocess

F32_SCRIPT = """
import sys
sys.path.insert(0, {tests_dir!r})
import numpy as np, jax, jax.numpy as jnp, jax.random as jr
import test_arhmm_chunked as T
assert not jax.config.jax_enable_x64
assert T.hmm_sampler_uses_log_space(jnp.float32)
for n, nlags in [(7, 1), (12, 3)]:
    m = T.make_model(n, 40, 4, 2, nlags, seed=n, dtype=np.float32)
    assert m["x"].dtype == jnp.float32
    args = T.model_args(m)
    seed = jr.PRNGKey(4)
    exp_z = T._ref_sample(seed, *args)
    exp_marg, exp_mode, exp_mll = T._ref_marginals(*args), T._ref_mode(*args), T._ref_mll(*args)
    for k in T.iters_list(n):
        T.set_mixed_map_iters(k)
        z = T.resample_discrete_stateseqs(seed, *args)
        np.testing.assert_array_equal(z, exp_z)
        marg = T.stateseq_marginals(*args)
        assert marg.dtype == jnp.float32
        np.testing.assert_allclose(marg, exp_marg, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(T.stateseq_mode(*args), exp_mode)
        mll = T.marginal_log_likelihood(m["mask"], m["x"], m["Ab"], m["Q"], m["pi"])
        np.testing.assert_allclose(mll, exp_mll, rtol=1e-4)
T._check_posterior()
print("OK")
"""


def test_float32_log_space_branch_matches_reference_and_posterior():
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    env["ARHMM_TEST_F32"] = "1"
    script = F32_SCRIPT.format(tests_dir=str(Path(__file__).parent))
    res = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]


# --------------------------------------------------- 3. memory-shape checks


def _sub_jaxprs(v):
    if hasattr(v, "jaxpr"):  # ClosedJaxpr
        yield v.jaxpr
    elif hasattr(v, "eqns"):
        yield v
    elif isinstance(v, (list, tuple)):
        for u in v:
            yield from _sub_jaxprs(u)


def all_shapes(closed):
    """Shapes of every var in the jaxpr, incl. scan/while/pjit/jit bodies."""
    stack, shapes = [closed.jaxpr], set()
    while stack:
        jp = stack.pop()
        for var in [*jp.invars, *jp.constvars, *jp.outvars]:
            shapes.add(tuple(getattr(var.aval, "shape", ())))
        for eqn in jp.eqns:
            shapes.update(tuple(getattr(v.aval, "shape", ())) for v in eqn.outvars)
            for p in eqn.params.values():
                stack.extend(_sub_jaxprs(p))
    return shapes


# distinct sizes so shapes are unambiguous: ar_dim=4, S=5, N=7, T-nlags=40
MN, MT, MS, MD = 7, 41, 5, 3
MTN = MT - 1


def _session_count(shape):
    """Sessions a var spans: dims left after one T - nlags, num_states, latent dim."""
    rest = list(shape)
    for v in (MTN, MS, MD):  # the latent dim of a whitened-residual chunk
        if v in rest:
            rest.remove(v)
    return int(np.prod(rest))


def _full_size_lls_shapes(shapes, allowed=()):
    """Vars with num_states and T - nlags dims that span all N sessions."""
    return [
        s
        for s in shapes
        if s not in allowed and MS in s and MTN in s and _session_count(s) >= MN
    ]


def _output_layouts(k):
    """Layouts the (N, T - nlags, num_states) marginals may take, padded or not."""
    c = -(-MN // k)
    return {(MN, MTN, MS), (k * c, MTN, MS), (k, c, MTN, MS), (c, k, MTN, MS)}


@pytest.mark.parametrize("k", [2, 3, MN])
def test_no_full_loglik_array(k):
    m = make_model(MN, MT, MS, MD, 1, seed=0)
    a = model_args(m)
    set_mixed_map_iters(k)
    seed = jr.PRNGKey(0)
    cases = {
        # the (N, T, S) marginals output is legitimately full size
        "marginals": (lambda *a: stateseq_marginals(*a), _output_layouts(k)),
        "sample": (lambda *a: resample_discrete_stateseqs(seed, *a), set()),
        "mode": (lambda *a: stateseq_mode(*a), set()),
    }
    for name, (f, allowed) in cases.items():
        bad = _full_size_lls_shapes(all_shapes(jax.make_jaxpr(f)(*a)), allowed)
        assert not bad, f"{name}: full-size (N, T, num_states) arrays {bad}"
    # marginal_log_likelihood is deliberately not chunked: upstream computes all
    # segments at once (tests/test_arhmm_marginals.py), so it is not checked here.


# ------------------------------------------------- 4. global iters honored


def test_set_mixed_map_iters_is_honored_after_first_call():
    n = 12
    m = make_model(n, 30, 3, 2, 1)
    a = model_args(m)
    fns = {
        "marginals": lambda: stateseq_marginals(*a),
        "mode": lambda: stateseq_mode(*a),
        "sample": lambda: resample_discrete_stateseqs(jr.PRNGKey(0), *a),
    }  # not marginal_log_likelihood, which ignores the setting (see above)
    for name, f in fns.items():
        jaxprs = {}
        for k in (2, 4, 2):
            set_mixed_map_iters(k)
            f()  # populate jit caches before inspecting
            # fresh lambda: make_jaxpr caches traces per function object
            jaxprs[k] = str(jax.make_jaxpr(lambda: f())())
        assert jaxprs[2] != jaxprs[4], f"{name}: iters change ignored"
