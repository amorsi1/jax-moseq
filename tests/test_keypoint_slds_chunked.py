"""Tests for the per-session (``mixed_map``) keypoint SLDS Gibbs steps.

The ``_ref_*`` functions below are the original whole-dataset implementations
(git revision c534d98); the current functions must agree with them (exactly,
or in distribution where the RNG stream legitimately changed), for every
``set_mixed_map_iters`` value. Everything runs on tiny synthetic data.
"""

import functools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.models import keypoint_slds as kslds  # noqa: E402
from jax_moseq.models.keypoint_slds import gibbs  # noqa: E402
from jax_moseq.models.keypoint_slds.alignment import (  # noqa: E402
    apply_rotation,
    estimate_aligned,
    estimate_coordinates,
    to_vanilla_slds,
    vector_to_angle,
)
from jax_moseq.utils import (  # noqa: E402
    apply_affine,
    get_mixed_map_iters,
    set_mixed_map_iters,
)
from jax_moseq.utils import distributions as dist  # noqa: E402
from jax_moseq.utils.distributions import sample_vonmises_fisher  # noqa: E402
from jax_moseq.utils.kalman import (  # noqa: E402
    ar_to_lds_dynamics,
    ar_to_lds_emissions,
    kalman_sample,
)

na = jnp.newaxis
T, K, D, L, S, NLAGS = 30, 4, 2, 3, 4, 3


@pytest.fixture(autouse=True)
def restore_iters():
    old = get_mixed_map_iters()
    yield
    set_mixed_map_iters(old)


def iters_for(n):
    return [1, 2, 3, n, n + 4]


# ----------------------------------------------------------------- data
def make_world(n, t=T, k=K, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    obs, ar_dim = (k - 1) * D, L * NLAGS
    mask = np.ones((n, t))
    mask[:, :3] = 1
    mask[1, 10:14] = 0
    mask[n - 1, 30:33] = 0
    Q = np.stack([np.eye(L) * (0.1 + 0.05 * i) for i in range(S)])
    w = dict(
        Y=rng.standard_normal((n, t, k, D)) * 3,
        mask=mask,
        x=rng.standard_normal((n, t, L)),
        v=rng.standard_normal((n, t, D)),
        h=rng.uniform(-np.pi, np.pi, (n, t)),
        s=rng.uniform(0.5, 2.0, (n, t, k)),
        z=rng.integers(0, S, (n, t - NLAGS)),
        Cd=rng.standard_normal((obs, L + 1)) * 0.5,
        sigmasq=rng.uniform(0.5, 1.5, k),
        Ab=rng.standard_normal((S, L, ar_dim + 1)) * 0.1,
        Q=Q,
        nu_s=5.0,
        s_0=1.0,
        nu_sigma=5.0,
        sigmasq_0=1.0,
        sigmasq_loc=0.5,
    )
    out = {a: jnp.asarray(b, dtype) for a, b in w.items()}
    out["z"] = jnp.asarray(w["z"])
    return out


@functools.lru_cache(maxsize=None)
def world(n):
    return make_world(n)


def seed_of(i=0):
    return jr.PRNGKey(i)


# --------------------------------------------------- reference (c534d98)
def _ref_sqerr(Y, x, v, h, Cd, mask=None):
    sqerr = ((Y - estimate_coordinates(x, v, h, Cd)) ** 2).sum(-1)
    return sqerr if mask is None else mask[..., na] * sqerr


def _ref_spread(seed, degs, variance):
    return variance / jr.gamma(seed, degs / 2, shape=variance.shape) / 2


def _ref_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kw):
    sqerr = _ref_sqerr(Y, x, v, h, Cd)
    return _ref_spread(seed, nu_s + 3, sqerr / sigmasq + s_0 * nu_s)


def _ref_obs_variance(seed, Y, mask, Cd, x, v, h, s, nu_sigma, sigmasq_0, **kw):
    sqerr = _ref_sqerr(Y, x, v, h, Cd, mask)
    degs = nu_sigma + 3 * mask.sum()
    S_y = (sqerr / s).reshape(-1, sqerr.shape[-1]).sum(0)
    return _ref_spread(seed, degs, nu_sigma * sigmasq_0 + S_y)


def _ref_heading(seed, Y, x, v, s, Cd, sigmasq, **kw):
    Y_bar = estimate_aligned(x, Cd, Y.shape[-2])
    Y_cent = Y - v[..., na, :]
    variance = s * sigmasq
    Sm = (Y_bar[..., :2, na] * Y_cent[..., na, :2] / variance[..., na, na]).sum(-3)
    mean_dir = jnp.stack([Sm[..., 0, 0] + Sm[..., 1, 1], Sm[..., 0, 1] - Sm[..., 1, 0]], -1)
    return vector_to_angle(sample_vonmises_fisher(seed, mean_dir))


def _ref_location(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc, parallel=True, **kw):
    k, d = Y.shape[-2:]
    Y_rot = apply_rotation(estimate_aligned(x, Cd, k), h)
    variance = s * sigmasq
    gammasq = 1 / (1 / variance).sum(-1, keepdims=True)
    mu = jnp.einsum("...tkd, ...tk->...td", Y - Y_rot, gammasq / variance)
    seeds = jr.split(seed, mask.shape[0])
    mdp = {"weights": jnp.eye(d), "bias": jnp.zeros(d), "cov": jnp.eye(d) * sigmasq_loc * 10}
    in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
    return jax.vmap(functools.partial(kalman_sample, parallel=parallel), in_axes)(
        seeds, mu, mask, jnp.zeros_like(mask[:, 1:], dtype=int),
        jnp.zeros(d), jnp.eye(d) * 1e4, jnp.eye(d)[na], jnp.zeros(d)[na],
        jnp.eye(d)[na] * sigmasq_loc, jnp.eye(d), jnp.zeros(d),
        jnp.repeat(gammasq, d, axis=-1), mdp, jnp.ones(d) * sigmasq.max() * 10,
    )


def _ref_x(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, jitter=1e-3, parallel=True, **kw):
    y, s, Cd, sigmasq = to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    n, obs = y.shape[0], y.shape[-1]
    ld, nl = Ab.shape[1], Ab.shape[2] // Ab.shape[1]
    m0, S0 = jnp.zeros(ld * nl), 10 * jnp.eye(ld * nl)
    C_, d_, R_, y_, _, _ = jax.vmap(ar_to_lds_emissions, in_axes=(na, 0, 0, na, na, na))(
        Cd, sigmasq * s, y, m0, S0, nl
    )
    A_, b_, Q_ = ar_to_lds_dynamics(Ab, Q)
    ez = jnp.zeros((ld * nl,) * 2).at[-ld:, -ld:].set(jnp.eye(ld))
    mdp = {"weights": ez, "bias": jnp.zeros(ld * nl), "cov": jnp.eye(ld * nl) * 10}
    in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
    x = jax.vmap(functools.partial(kalman_sample, jitter=jitter, parallel=parallel), in_axes)(
        jr.split(seed, n), y_, mask[:, nl - 1:], z, m0, S0, A_, b_, Q_, C_, d_, R_,
        mdp, jnp.ones(obs) * 10,
    )
    if nl > 1:
        x = jnp.concatenate(
            [x[:, 0, : (nl - 1) * ld].reshape(-1, nl - 1, ld), x[:, :, -ld:]], axis=1
        )
    return x


# name -> (function under test, reference, extra kwargs used by both)
CASES = {
    "scales": (gibbs.resample_scales, _ref_scales),
    "heading": (gibbs.resample_heading, _ref_heading),
    "location": (gibbs.resample_location, _ref_location),
    "x": (gibbs.resample_continuous_stateseqs, _ref_x),
    "obs_variance": (gibbs.resample_obs_variance, _ref_obs_variance),
}


# sequential message passing keeps jit times short; the parallel flag has its own test
SEQ = {"location": True, "x": True}


def call(name, w, seed=None, **extra):
    seed = seed_of() if seed is None else seed
    if name in SEQ:
        extra.setdefault("parallel_message_passing", False)
    return CASES[name][0](seed, **w, **extra)


def ref(name, w, seed=None):
    seed = seed_of() if seed is None else seed
    extra = {"parallel": False} if name in SEQ else {}
    return CASES[name][1](seed, **w, **extra)


# --------------------------------------------------------------- helpers
def walk_eqns(jaxpr):
    stack = [jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr]
    while stack:
        jp = stack.pop()
        for eqn in jp.eqns:
            yield eqn
            for val in eqn.params.values():
                stack.extend(_subs(val))


def _subs(val):
    if hasattr(val, "jaxpr") and hasattr(val, "consts"):
        yield val.jaxpr
    elif hasattr(val, "eqns"):
        yield val
    elif isinstance(val, (list, tuple)):
        for v in val:
            yield from _subs(v)


def var_shapes(name, w):
    jp = jax.make_jaxpr(lambda w_: CASES[name][0](seed_of(), **w_))(w)
    return [tuple(v.aval.shape) for e in walk_eqns(jp) for v in e.outvars if hasattr(v.aval, "shape")]


def trip_counts(name, w):
    jp = jax.make_jaxpr(lambda w_: CASES[name][0](seed_of(), **w_))(w)
    return {e.params["length"] for e in walk_eqns(jp) if e.primitive.name == "scan"}


def mean_var(name, w, n_seeds, transform=lambda a: a, fn=None):
    fn = fn or (lambda sd: call(name, w, sd))
    outs = np.stack([np.asarray(transform(fn(jr.PRNGKey(i)))) for i in range(n_seeds)])
    return outs.mean(0), outs.var(0, ddof=1)


def assert_same_distribution(a, b, n, zmax=5.0):
    (ma, va), (mb, vb) = a, b
    z = np.abs(ma - mb) / np.sqrt((va + vb) / n + 1e-300)
    assert np.nanmax(z) < zmax, f"max |z| = {np.nanmax(z):.2f}"
    # variances agree within a factor (loose; sampling error of a variance)
    ratio = (va + 1e-12) / (vb + 1e-12)
    assert np.median(ratio) == pytest.approx(1.0, rel=0.25)


# ----------------------------------------------------------------- tests
@pytest.mark.parametrize("n", [7, 12])
def test_heading_matches_reference(n):
    w = world(n)
    expected = ref("heading", {a: w[a] for a in "Y x v s Cd sigmasq".split()})
    for it in iters_for(n):
        set_mixed_map_iters(it)
        got = call("heading", w)
        assert got.shape == (n, T)
        # same RNG stream; only fp rounding (jit fusion vs eager reference) may differ
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-9)


@pytest.mark.parametrize("n", [7, 12])
@pytest.mark.parametrize("name", ["location", "x", "obs_variance"])
def test_matches_reference_stream(name, n):
    """These steps keep the original RNG stream (split(seed, N) per session /
    one draw on summed errors), so they agree with the original numerically."""
    w = world(n)
    expected = np.asarray(ref(name, w))
    for it in iters_for(n):
        set_mixed_map_iters(it)
        np.testing.assert_allclose(np.asarray(call(name, w)), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("n", [7, 12])
@pytest.mark.parametrize("name", list(CASES))
def test_chunk_size_invariance(name, n):
    w = world(n)
    set_mixed_map_iters(1)
    base = np.asarray(call(name, w))
    exact = name == "scales"
    for it in iters_for(n)[1:]:
        set_mixed_map_iters(it)
        got = np.asarray(call(name, w))
        assert got.shape == base.shape
        if exact:
            np.testing.assert_array_equal(got, base)
        else:
            np.testing.assert_allclose(got, base, rtol=1e-9, atol=1e-10)


def tiny_world():
    return make_world(3, t=12, seed=3)


def test_scales_distribution_matches_reference():
    w = tiny_world()
    sub = {a: w[a] for a in "Y x v h Cd sigmasq nu_s s_0".split()}
    n_seeds = 300
    set_mixed_map_iters(2)
    got = mean_var("scales", sub, n_seeds, jnp.log)
    jref = jax.jit(lambda sd: _ref_scales(sd, **sub))
    want = mean_var("scales", sub, n_seeds, jnp.log, fn=jref)
    assert_same_distribution(got, want, n_seeds)


@pytest.mark.parametrize("s0_kind", ["scalar", "array"])
def test_scales_prior_scalar_or_full_array(s0_kind):
    w = tiny_world()
    n, t = w["Y"].shape[:2]
    s_0 = 1.5 if s0_kind == "scalar" else jnp.asarray(
        np.random.default_rng(1).uniform(0.5, 3.0, (n, t, K))
    )
    sub = {a: w[a] for a in "Y x v h Cd sigmasq nu_s".split()} | {"s_0": s_0}
    set_mixed_map_iters(2)
    out = gibbs.resample_scales(seed_of(), **sub)
    assert out.shape == (n, t, K) and bool(jnp.isfinite(out).all()) and bool((out > 0).all())
    n_seeds = 300
    got = mean_var("scales", sub, n_seeds, jnp.log, fn=lambda sd: gibbs.resample_scales(sd, **sub))
    want = mean_var("scales", sub, n_seeds, jnp.log, fn=lambda sd: _ref_scales(sd, **sub))
    assert_same_distribution(got, want, n_seeds)


def test_x_and_location_distribution_matches_reference():
    w = tiny_world()
    n_seeds = 150
    set_mixed_map_iters(2)
    for name in ("x", "location"):
        jref = jax.jit(lambda sd, nm=name: CASES[nm][1](sd, **w, parallel=False))
        got = mean_var(name, w, n_seeds)
        want = mean_var(name, w, n_seeds, fn=jref)
        assert_same_distribution(got, want, n_seeds)


def test_location_honors_parallel_message_passing():
    w = {a: world(7)[a] for a in "Y mask x h s Cd sigmasq sigmasq_loc".split()}
    set_mixed_map_iters(2)
    par = gibbs.resample_location(seed_of(), **w, parallel_message_passing=True)
    seq = gibbs.resample_location(seed_of(), **w, parallel_message_passing=False)
    # the two samplers consume randomness differently, so each flag value must
    # match the reference run with that same flag (and the two must differ)
    for got, flag in [(par, True), (seq, False)]:
        want = _ref_location(seed_of(), **w, parallel=flag)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-6)
    assert not np.allclose(np.asarray(par), np.asarray(seq), atol=1e-3)

    def jaxpr_str(flag):
        return str(jax.make_jaxpr(
            lambda w_: gibbs.resample_location(seed_of(), **w_, parallel_message_passing=flag)
        )(w))

    assert jaxpr_str(True) != jaxpr_str(False), "flag does not change the traced program"


def tiny_model(n=3, t=20, k=K, seed=0):
    rng = np.random.default_rng(seed)
    Y = np.cumsum(rng.standard_normal((n, t, k, D)) * 0.1, axis=1) + rng.standard_normal((n, 1, k, D)) * 3
    data = {"Y": jnp.asarray(Y), "mask": jnp.ones((n, t))}
    return data, kslds.init_model(
        data=data, seed=jr.PRNGKey(seed), anterior_idxs=[0], posterior_idxs=[k - 1],
        trans_hypparams=dict(num_states=S, alpha=5.7, gamma=1e3, kappa=1e4),
        ar_hypparams=dict(latent_dim=L, nlags=NLAGS, S_0_scale=0.01, K_0_scale=10.0),
        obs_hypparams=dict(sigmasq_0=0.1, sigmasq_C=0.1, nu_sigma=1e5, nu_s=5),
        cen_hypparams=dict(sigmasq_loc=0.5),
        PCA_fitting_num_frames=1000,
    )


def test_resample_model_forwards_parallel_flag_and_is_chunk_stable(monkeypatch):
    data, model = tiny_model()
    seen = []
    real = gibbs.resample_location

    def spy(*a, **kw):
        seen.append(kw.get("parallel_message_passing"))
        return real(*a, **kw)

    monkeypatch.setattr(gibbs, "resample_location", spy)
    outs = {}
    for it, flag in [(1, False), (2, True), (3, False)]:
        set_mixed_map_iters(it)
        m = kslds.resample_model(data, **model, parallel_message_passing=flag)
        m = kslds.resample_model(data, **m, parallel_message_passing=flag)
        outs[it] = m
        for key, val in m["states"].items():
            assert val.shape == model["states"][key].shape, key
            assert bool(jnp.isfinite(val).all()), key
        assert bool(jnp.isfinite(m["params"]["Cd"]).all())
    assert seen == [False, False, True, True, False, False]
    # same flag, different chunking => same states
    for key in outs[1]["states"]:
        np.testing.assert_allclose(
            np.asarray(outs[1]["states"][key]), np.asarray(outs[3]["states"][key]), atol=1e-6
        )


def test_no_full_dataset_temporaries_in_heading():
    n, t, k = 7, 53, 5
    w = make_world(n, t=t, k=k)
    w = {a: w[a] for a in "Y x v s Cd sigmasq".split()}
    set_mixed_map_iters(3)
    shapes = var_shapes("heading", w)
    assert not [s for s in shapes if s == (n, t, k, 2, 2)]
    assert not [s for s in shapes if s[:1] == (n,) and np.prod(s) >= n * t * k * 4]


def test_no_full_dataset_temporaries_in_continuous_stateseqs():
    n, t, k = 7, 53, 5
    obs = k * D
    w = make_world(n, t=t, k=k)
    set_mixed_map_iters(3)
    shapes = var_shapes("x", w)
    bad = [s for s in shapes if s[:1] == (n,) and s[-1] == obs and len(s) == 3]
    assert not bad, bad
    assert not [s for s in shapes if len(s) == 3 and s[0] == n and s[1] in (t, t - NLAGS + 1) and s[2] == obs]


@pytest.mark.parametrize("name", ["scales", "heading", "location", "x", "obs_variance"])
def test_global_iters_changes_trip_count(name):
    w = world(12)
    set_mixed_map_iters(1)
    base = trip_counts(name, w)
    set_mixed_map_iters(4)
    four = trip_counts(name, w)
    assert 4 in four and 4 not in base, (base, four)
    set_mixed_map_iters(1)
    assert trip_counts(name, w) == base  # re-traces back


@pytest.mark.parametrize("name", list(CASES))
def test_float32_smoke(name):
    w = make_world(7, seed=5, dtype=np.float32)
    set_mixed_map_iters(2)
    out = call(name, w)
    assert out.dtype in (jnp.float32, jnp.float64)  # gamma draws may upcast under x64
    assert bool(jnp.isfinite(out).all())


# ------------------------------------------------- float32 PSD numerics
def _setter():
    if not hasattr(dist, "set_stable_cholesky"):
        pytest.skip("jax_moseq.utils.distributions.set_stable_cholesky not available")
    return dist.set_stable_cholesky


@pytest.fixture
def psd_flag():
    setter = _setter()
    yield setter
    setter(False)


def _ill_conditioned(n=6, scale=100.0, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.geomspace(1.0, 1e-3, n) * scale
    ev[-1] = -1e-4 * scale  # slightly negative: not PSD
    return jnp.asarray(q @ np.diag(ev) @ q.T, jnp.float32)


def test_psd_flag_off_matches_original_and_nans():
    S_bad = _ill_conditioned()
    nu = jnp.float32(20.0)
    out = dist.sample_invwishart(seed_of(), S_bad, nu)
    mn = dist.sample_mn(seed_of(), jnp.zeros((6, 6), jnp.float32), S_bad, S_bad)
    assert bool(jnp.isnan(out).any()) and bool(jnp.isnan(mn).any())
    # well-conditioned input: identical to the original formulas
    A = np.random.default_rng(0).standard_normal((4, 4))
    spd = jnp.asarray(A @ A.T + 4 * np.eye(4))
    chi2 = jr.split(seed_of())
    x = jnp.diag(jnp.sqrt(dist.sample_chi2(chi2[0], 9.0 - jnp.arange(4))))
    x = x.at[jnp.triu_indices_from(x, 1)].set(jr.normal(chi2[1], (6,)))
    R = jnp.linalg.qr(x, "r")
    Tm = jax.scipy.linalg.solve_triangular(R.T, jnp.linalg.cholesky(spd).T, lower=True).T
    np.testing.assert_array_equal(
        np.asarray(dist.sample_invwishart(seed_of(), spd, 9.0)), np.asarray(Tm @ Tm.T)
    )


def test_psd_flag_on_is_finite(psd_flag):
    S_bad = _ill_conditioned()
    psd_flag(True)
    out = dist.sample_invwishart(seed_of(), S_bad, jnp.float32(20.0))
    mn = dist.sample_mn(seed_of(), jnp.zeros((6, 6), jnp.float32), S_bad, S_bad)
    assert bool(jnp.isfinite(out).all()) and bool(jnp.isfinite(mn).all())
    # a clean SPD input is (nearly) unaffected by the projection
    A = np.random.default_rng(0).standard_normal((4, 4))
    spd = jnp.asarray(A @ A.T + 4 * np.eye(4))
    on = dist.sample_invwishart(seed_of(), spd, 9.0)
    psd_flag(False)
    off = dist.sample_invwishart(seed_of(), spd, 9.0)
    np.testing.assert_allclose(np.asarray(on), np.asarray(off), rtol=1e-6)
