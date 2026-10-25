"""Tests for the per-session (``mixed_map``) AR-HMM steps.

The ``_ref_*`` functions below are the original implementations (git c534d98),
which materialize the full (num_states, N, T) log-likelihood array. The public
functions must agree with them for any ``set_mixed_map_iters`` value and must
never build that array themselves.
"""

import functools
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dynamax.hidden_markov_model.inference import (  # noqa: E402
    hmm_filter,
    hmm_posterior_mode,
    hmm_smoother,
)

from jax_moseq.models.arhmm.gibbs import (  # noqa: E402
    resample_ar_params,
    resample_discrete_stateseqs,
    stateseq_marginals,
    stateseq_mode,
)
from jax_moseq.models.arhmm.log_prob import (  # noqa: E402
    marginal_log_likelihood,
    state_cross_likelihoods,
)
from jax_moseq.utils import (  # noqa: E402
    get_mixed_map_iters,
    pad_affine,
    psd_inv,
    psd_solve,
    set_mixed_map_iters,
)
from jax_moseq.utils.autoregression import (  # noqa: E402
    ar_log_likelihood,
    get_lags,
    get_nlags,
)
from jax_moseq.utils.distributions import (  # noqa: E402
    sample_hmm_stateseq,
    sample_mniw,
)


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


def hyp(d, nlags, dtype=np.float64):
    ar_dim = d * nlags + 1
    return dict(
        nu_0=d + 2.0,
        S_0=jnp.eye(d, dtype=dtype),
        M_0=jnp.zeros((d, ar_dim), dtype),
        K_0=jnp.eye(ar_dim, dtype=dtype),
    )


def iters_list(n):
    return [1, 2, 3, n, n + 4]


def assert_close(a, b, rtol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-12)


# ------------------------------------------------- reference implementations


def _ref_lls(x, Ab, Q):
    """(N, T - nlags, num_states) log-likelihoods, built all at once."""
    lls = jax.lax.map(functools.partial(ar_log_likelihood, x), (Ab, Q))
    return jnp.moveaxis(lls, 0, -1)


def _ref_masked_lls(x, mask, Ab, Q):
    return _ref_lls(x, Ab, Q) * mask[:, get_nlags(Ab) :, None]


def _uniform(pi):
    return jnp.ones(pi.shape[0]) / pi.shape[0]


def _ref_marginals(x, mask, Ab, Q, pi):
    lls = _ref_masked_lls(x, mask, Ab, Q)
    f = lambda ll: hmm_smoother(_uniform(pi), pi, ll).smoothed_probs
    return jax.vmap(f)(lls)


def _ref_mode(x, mask, Ab, Q, pi):
    lls = _ref_masked_lls(x, mask, Ab, Q)
    return jax.vmap(lambda ll: hmm_posterior_mode(_uniform(pi), pi, ll))(lls)


def _ref_mll(x, mask, Ab, Q, pi):
    lls = _ref_masked_lls(x, mask, Ab, Q)
    f = lambda ll: hmm_filter(_uniform(pi), pi, ll).marginal_loglik.sum()
    return jax.vmap(f)(lls).sum()


def _ref_sample(seed, x, mask, Ab, Q, pi):
    nlags = get_nlags(Ab)
    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0, None, 0, 0))(
        jr.split(seed, mask.shape[0]),
        pi,
        _ref_lls(x, Ab, Q),
        mask.astype(float)[:, nlags:],
    )
    return z


def _ref_regression(x_in, x_out, nu_0, S_0, M_0, K_0, args):
    seed, mask = args
    S_oo = jnp.einsum("ti,tj,t->ij", x_out, x_out, mask)
    S_oi = jnp.einsum("ti,tj,t->ij", x_out, x_in, mask)
    S_ii = jnp.einsum("ti,tj,t->ij", x_in, x_in, mask)
    K_0_inv = psd_inv(K_0)
    K_n_inv = K_0_inv + S_ii
    K_n = psd_inv(K_n_inv)
    M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_oi.T).T
    S_n = S_0 + S_oo + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
    return sample_mniw(seed, nu_0 + mask.sum(), S_n, M_n, K_n)


def _ref_ar_params(seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0):
    seeds = jr.split(seed, num_states)
    masks = mask[..., nlags:].reshape(1, -1) * jnp.eye(num_states)[:, z.reshape(-1)]
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
    x_out = x[..., nlags:, :].reshape(-1, x.shape[-1])
    f = functools.partial(_ref_regression, x_in, x_out, nu_0, S_0, M_0, K_0)
    return jax.lax.map(f, (seeds, masks))


def _ref_cross(params, states, mask):
    """Original numpy implementation of state_cross_likelihoods."""
    x, Ab, Q = states["x"], params["Ab"], params["Q"]
    lls = np.asarray(_ref_lls(x, Ab, Q))
    nlags = mask.shape[1] - lls.shape[1]
    lls = lls[mask[:, nlags:] > 0]
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
    m = make_model(n, 60, 4, 2, nlags, seed=n + nlags)
    args = (m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    exp_marg, exp_mode, exp_mll = (
        _ref_marginals(*args),
        _ref_mode(*args),
        _ref_mll(*args),
    )
    for k in iters_list(n):
        set_mixed_map_iters(k)
        assert_close(stateseq_marginals(*args), exp_marg)
        np.testing.assert_array_equal(stateseq_mode(*args), exp_mode)
        assert_close(marginal_log_likelihood(m["mask"], m["x"], m["Ab"], m["Q"], m["pi"]), exp_mll)


@pytest.mark.parametrize("n,nlags", CASES)
def test_state_cross_likelihoods_match_reference(n, nlags):
    s, t = 4, 60
    m = make_model(n, t, s, 2, nlags, seed=3)
    tn = t - nlags
    z = (np.arange(tn)[None] // 5 + np.arange(n)[:, None]) % s
    params, states = {"Ab": m["Ab"], "Q": m["Q"]}, {"x": m["x"], "z": z}
    mask = np.asarray(m["mask"])
    expected = _ref_cross(params, states, mask)
    for k in iters_list(n):
        set_mixed_map_iters(k)
        assert_close(state_cross_likelihoods(params, states, mask), expected)


# ------------------------------------------------ 2. discrete state sampling


@pytest.mark.parametrize("n,nlags", CASES)
def test_resample_stateseqs_matches_reference_and_is_chunk_invariant(n, nlags):
    s, t = 5, 60
    m = make_model(n, t, s, 3, nlags, seed=1)
    seed = jr.PRNGKey(11)
    args = (m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    expected = _ref_sample(seed, *args)
    for k in iters_list(n):
        set_mixed_map_iters(k)
        z = resample_discrete_stateseqs(seed, *args)
        assert z.shape == (n, t - nlags) and z.dtype == expected.dtype
        assert int(z.min()) >= 0 and int(z.max()) < s
        np.testing.assert_array_equal(z, expected)  # RNG: split(seed, N) per session


def test_resample_stateseqs_posterior_matches_marginals():
    n, t, s = 2, 8, 2
    m = make_model(n, t, s, 2, 1, seed=5)  # d >= 2: get_nlags = ar_dim // d
    args = (m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    set_mixed_map_iters(2)
    marg = np.asarray(stateseq_marginals(*args))
    seeds = jr.split(jr.PRNGKey(0), 600)
    zs = np.asarray(jax.vmap(lambda sd: resample_discrete_stateseqs(sd, *args))(seeds))
    for sess, frame in [(0, 3), (1, 5)]:
        emp = (zs[:, sess, frame] == 1).mean()
        assert abs(emp - marg[sess, frame, 1]) < 0.08


# --------------------------------------------------------- 3. AR parameters


@pytest.mark.parametrize("n,nlags", CASES)
def test_resample_ar_params_matches_reference(n, nlags):
    s, t, d = 4, 60, 2
    m = make_model(n, t, s, d, nlags, seed=2)
    z = jr.randint(jr.PRNGKey(3), (n, t - nlags), 0, s - 1)  # last state empty
    kw = dict(nlags=nlags, num_states=s, mask=m["mask"], x=m["x"], z=z, **hyp(d, nlags))
    seed = jr.PRNGKey(7)
    Ab_e, Q_e = _ref_ar_params(seed, **kw)
    for k in iters_list(n):
        set_mixed_map_iters(k)
        Ab, Q = resample_ar_params(seed, **kw)
        assert_close(Ab, Ab_e, rtol=1e-6)
        assert_close(Q, Q_e, rtol=1e-6)


# --------------------------------------------------- 4. memory-shape checks


def _sub_jaxprs(v):
    if hasattr(v, "jaxpr"):  # ClosedJaxpr
        yield v.jaxpr
    elif hasattr(v, "eqns"):
        yield v
    elif isinstance(v, (list, tuple)):
        for u in v:
            yield from _sub_jaxprs(u)


def all_shapes(closed):
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


def scan_lengths(closed):
    stack, out = [closed.jaxpr], []
    while stack:
        jp = stack.pop()
        for eqn in jp.eqns:
            if eqn.primitive.name == "scan":
                out.append(eqn.params["length"])
            for p in eqn.params.values():
                stack.extend(_sub_jaxprs(p))
    return out


# distinct sizes so shapes are unambiguous: ar_dim=4, S=5, N=7, T-nlags=40
MN, MT, MS, MD = 7, 41, 5, 3


def _mem_model():
    return make_model(MN, MT, MS, MD, 1, seed=0)


def _bad_lls_shape(shape, allowed=()):
    tn = MT - 1
    return shape not in allowed and all(v in shape for v in (MN, MS, tn))


@pytest.mark.parametrize("k", [2, 3, MN])
def test_no_full_loglik_array(k):
    m = _mem_model()
    a = (m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    set_mixed_map_iters(k)
    seed = jr.PRNGKey(0)
    cases = {
        # output (N, T, S) of marginals is legitimately full size
        "marginals": (lambda *a: stateseq_marginals(*a), a, {(MN, MT - 1, MS)}),
        "sample": (lambda *a: resample_discrete_stateseqs(seed, *a), a, set()),
        "mode": (lambda *a: stateseq_mode(*a), a, set()),
        "mll": (
            lambda x, mk, Ab, Q, pi: marginal_log_likelihood(mk, x, Ab, Q, pi),
            a,
            set(),
        ),
    }
    for name, (f, args, allowed) in cases.items():
        shapes = all_shapes(jax.make_jaxpr(f)(*args))
        bad = [s for s in shapes if _bad_lls_shape(s, allowed)]
        assert not bad, f"{name}: full (S,N,T)-sized arrays {bad}"


def test_ar_params_no_onehot_mask():
    m = _mem_model()
    tn = MT - 1
    z = jnp.zeros((MN, tn), int)
    kw = dict(nlags=1, num_states=MS, mask=m["mask"], x=m["x"], **hyp(MD, 1))
    set_mixed_map_iters(2)
    f = lambda z: resample_ar_params(jr.PRNGKey(0), z=z, **kw)
    shapes = all_shapes(jax.make_jaxpr(f)(z))
    bad = [
        s
        for s in shapes
        if MS in s and (MN * tn in s or (MN in s and tn in s)) and len(s) >= 2
    ]
    assert not bad, f"(num_states, N*T)-sized arrays {bad}"


# ------------------------------------------------- 5. global iters honored


def test_set_mixed_map_iters_is_honored_after_first_call():
    n = 12
    m = make_model(n, 30, 3, 2, 1)
    a = (m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    fns = {
        "marginals": lambda: stateseq_marginals(*a),
        "sample": lambda: resample_discrete_stateseqs(jr.PRNGKey(0), *a),
        "mll": lambda: marginal_log_likelihood(m["mask"], m["x"], m["Ab"], m["Q"], m["pi"]),
    }
    for name, f in fns.items():
        jaxprs = {}
        for k in (2, 4, 2):
            set_mixed_map_iters(k)
            f()  # call (populates jit caches) before inspecting
            # fresh lambda: make_jaxpr caches traces per function object
            jaxprs[k] = str(jax.make_jaxpr(lambda: f())())
        # the loop bound is baked into the jaxpr, so it must follow the global
        assert jaxprs[2] != jaxprs[4], f"{name}: iters change ignored"


# ------------------------------------------------------------ 6. float32


FLOAT32_SCRIPT = """
import numpy as np, jax, jax.numpy as jnp, jax.random as jr
from jax_moseq.models.arhmm.gibbs import *
from jax_moseq.models.arhmm.log_prob import marginal_log_likelihood
from jax_moseq.utils import set_mixed_map_iters
n, t, s, d, nl = 7, 40, 4, 2, 1
r = np.random.default_rng(0)
f32 = lambda a: jnp.asarray(a, jnp.float32)
x = f32(np.cumsum(r.standard_normal((n, t, d)), 1) * 0.3)
mask = f32(np.ones((n, t)))
Ab = f32(0.3 * r.standard_normal((s, d, d * nl + 1)))
Q = f32(np.tile(np.eye(d), (s, 1, 1)) * 0.5)
pi = f32(r.dirichlet(np.ones(s), size=s))
set_mixed_map_iters(3)
z = resample_discrete_stateseqs(jr.PRNGKey(0), x, mask, Ab, Q, pi)
hyp = dict(nu_0=d + 2.0, S_0=f32(np.eye(d)), M_0=f32(np.zeros((d, d * nl + 1))),
           K_0=f32(np.eye(d * nl + 1)))
outs = [stateseq_marginals(x, mask, Ab, Q, pi), marginal_log_likelihood(mask, x, Ab, Q, pi),
        *resample_ar_params(jr.PRNGKey(1), nlags=nl, num_states=s, mask=mask, x=x, z=z, **hyp)]
assert all(np.isfinite(np.asarray(o)).all() for o in outs)
assert outs[0].dtype == jnp.float32 and int(z.max()) < s
assert (np.asarray(stateseq_mode(x, mask, Ab, Q, pi)) < s).all()
print("OK")
"""


def test_float32_runs_and_is_finite():
    """Runs without x64 (the original also fails on float32 inputs under x64)."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    res = subprocess.run(
        [sys.executable, "-c", FLOAT32_SCRIPT], capture_output=True, text=True, env=env
    )
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-2000:]
