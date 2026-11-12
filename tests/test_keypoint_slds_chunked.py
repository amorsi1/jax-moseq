"""Tests for the per-session (``mixed_map``) keypoint SLDS Gibbs steps.

The ``_ref_*`` functions are the original whole-dataset implementations
(upstream 944322a); the current functions must agree with them (numerically,
or in distribution where the RNG stream legitimately changed: scales), for
every ``set_mixed_map_iters`` value. Everything runs on tiny synthetic data.
"""

import functools
import os
import subprocess
import sys
from pathlib import Path

import jax

# the float32 smoke test re-imports this module with KPMS_F32=1
jax.config.update("jax_enable_x64", os.environ.get("KPMS_F32") != "1")

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow_probability.substrates.jax.distributions as tfd  # noqa: E402

from jax_moseq.models import keypoint_slds as kslds  # noqa: E402
from jax_moseq.models.keypoint_slds import gibbs  # noqa: E402
from jax_moseq.models.keypoint_slds.alignment import (  # noqa: E402
    apply_rotation,
    estimate_aligned,
    estimate_coordinates,
    to_vanilla_slds,
)
from jax_moseq.utils import get_mixed_map_iters, set_mixed_map_iters  # noqa: E402
from jax_moseq.utils.distributions import sample_vonmises  # noqa: E402
from jax_moseq.utils.kalman import (  # noqa: E402
    ar_to_lds_dynamics,
    ar_to_lds_emissions,
    kalman_sample,
)

na = jnp.newaxis
T, K, D, L, S, NLAGS = 40, 4, 2, 3, 4, 3
F64 = os.environ.get("KPMS_F32") != "1"


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
    mask[1, 10:14] = 0
    mask[n - 1, 30:33] = 0
    mask[0, :2] = 0
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
        Q=np.stack([np.eye(L) * (0.1 + 0.05 * i) for i in range(S)]),
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


# --------------------------------------------------- reference (944322a)
def _gamma_spread(seed, degs, variance):
    a = jnp.asarray(degs / 2, variance.dtype)
    g = tfd.Gamma(a, jnp.ones((), variance.dtype)).sample(variance.shape, seed=seed)
    return variance / g / 2


def _ref_sqerr(Y, x, v, h, Cd, mask=None):
    sqerr = ((Y - estimate_coordinates(x, v, h, Cd)) ** 2).sum(-1)
    return sqerr if mask is None else mask[..., na] * sqerr


def _ref_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kw):
    sqerr = _ref_sqerr(Y, x, v, h, Cd)
    return _gamma_spread(seed, nu_s + 3, sqerr / sigmasq + s_0 * nu_s)


def _ref_obs_variance(seed, Y, mask, Cd, x, v, h, s, nu_sigma, sigmasq_0, **kw):
    sqerr = _ref_sqerr(Y, x, v, h, Cd, mask)
    degs = nu_sigma + 3 * mask.sum()
    S_y = (sqerr / s).reshape(-1, sqerr.shape[-1]).sum(0)
    return _gamma_spread(seed, degs, nu_sigma * sigmasq_0 + S_y)


def _ref_heading(seed, Y, x, v, s, Cd, sigmasq, **kw):
    Y_bar = estimate_aligned(x, Cd, Y.shape[-2])
    Y_cent = Y - v[..., na, :]
    variance = s * sigmasq
    Sm = (Y_bar[..., :2, na] * Y_cent[..., na, :2] / variance[..., na, na]).sum(-3)
    kc = Sm[..., 0, 0] + Sm[..., 1, 1]
    ks = Sm[..., 0, 1] - Sm[..., 1, 0]
    return sample_vonmises(seed, jnp.arctan2(ks, kc), jnp.sqrt(kc**2 + ks**2))


def _ref_location(
    seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc, parallel_message_passing=True, **kw
):
    k, d = Y.shape[-2:]
    Y_rot = apply_rotation(estimate_aligned(x, Cd, k), h)
    variance = s * sigmasq
    gammasq = 1 / (1 / variance).sum(-1, keepdims=True)
    mu = jnp.einsum("...tkd, ...tk->...td", Y - Y_rot, gammasq / variance)
    seeds = jr.split(seed, mask.shape[0])
    mdp = {
        "weights": jnp.eye(d),
        "bias": jnp.zeros(d),
        "cov": jnp.eye(d) * sigmasq_loc * 10,
    }
    R = gammasq[..., na] * jnp.eye(d, dtype=gammasq.dtype)
    in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
    fn = functools.partial(kalman_sample, parallel=parallel_message_passing)
    return jax.vmap(fn, in_axes)(
        seeds,
        mu,
        mask,
        jnp.zeros_like(mask[:, 1:], dtype=int),
        jnp.zeros(d),
        jnp.eye(d) * 1e4,
        jnp.eye(d)[na],
        jnp.zeros(d)[na],
        jnp.eye(d)[na] * sigmasq_loc,
        jnp.eye(d),
        jnp.zeros(d),
        R,
        mdp,
        jnp.eye(d, dtype=sigmasq.dtype) * sigmasq.max() * 10,
    )


def _ref_x(
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
    **kw
):
    y, s, Cd, sigmasq = to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    n, obs = y.shape[0], y.shape[-1]
    ld = Ab.shape[1]
    nl = Ab.shape[2] // ld
    m0, S0 = jnp.zeros(ld * nl), 10 * jnp.eye(ld * nl)
    C_, d_, R_, y_, _, _ = jax.vmap(
        ar_to_lds_emissions, in_axes=(na, 0, 0, na, na, na)
    )(Cd, sigmasq * s, y, m0, S0, nl)
    A_, b_, Q_ = ar_to_lds_dynamics(Ab, Q)
    ez = jnp.zeros((ld * nl,) * 2).at[-ld:, -ld:].set(jnp.eye(ld))
    mdp = {"weights": ez, "bias": jnp.zeros(ld * nl), "cov": jnp.eye(ld * nl) * 10}
    in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
    fn = functools.partial(kalman_sample, jitter=jitter, parallel=parallel_message_passing)
    x = jax.vmap(fn, in_axes)(
        jr.split(seed, n),
        y_,
        mask[:, nl - 1 :],
        z,
        m0,
        S0,
        A_,
        b_,
        Q_,
        C_,
        d_,
        R_,
        mdp,
        jnp.ones(obs) * 10,
    )
    if nl > 1:
        x = jnp.concatenate(
            [x[:, 0, : (nl - 1) * ld].reshape(-1, nl - 1, ld), x[:, :, -ld:]], axis=1
        )
    return x


# name -> (function under test, reference)
CASES = {
    "scales": (gibbs.resample_scales, _ref_scales),
    "heading": (gibbs.resample_heading, _ref_heading),
    "location": (gibbs.resample_location, _ref_location),
    "x": (gibbs.resample_continuous_stateseqs, _ref_x),
    "obs_variance": (gibbs.resample_obs_variance, _ref_obs_variance),
}
# sequential message passing keeps jit times short; the flag has its own test
SEQ = ("location", "x")


def call(name, w, seed=None, **extra):
    if name in SEQ:
        extra.setdefault("parallel_message_passing", False)
    return CASES[name][0](seed_of() if seed is None else seed, **w, **extra)


def ref(name, w, seed=None, **extra):
    if name in SEQ:
        extra.setdefault("parallel_message_passing", False)
    return CASES[name][1](seed_of() if seed is None else seed, **w, **extra)


# --------------------------------------------------------------- helpers
def _subs(val):
    if hasattr(val, "jaxpr") and hasattr(val, "consts"):
        yield val.jaxpr
    elif hasattr(val, "eqns"):
        yield val
    elif isinstance(val, (list, tuple)):
        for v in val:
            yield from _subs(v)


def walk_eqns(closed):
    stack = [closed.jaxpr if hasattr(closed, "jaxpr") else closed]
    while stack:
        jp = stack.pop()
        for eqn in jp.eqns:
            yield eqn
            for val in eqn.params.values():
                stack.extend(_subs(val))


def jaxpr_of(name, w, **extra):
    """Fresh lambda each time: make_jaxpr caches by function object."""
    if name in SEQ:
        extra.setdefault("parallel_message_passing", False)
    fn = CASES[name][0]
    return jax.make_jaxpr(lambda w_: fn(seed_of(), **w_, **extra))(w)


def var_shapes(name, w, **extra):
    return [
        tuple(v.aval.shape)
        for e in walk_eqns(jaxpr_of(name, w, **extra))
        for v in e.outvars
        if hasattr(v.aval, "shape")
    ]


def scan_lengths(name, w):
    return {e.params["length"] for e in walk_eqns(jaxpr_of(name, w)) if e.primitive.name == "scan"}


def sample_stats(fn, n_seeds, transform=lambda a: a):
    outs = np.stack([np.asarray(transform(fn(jr.PRNGKey(i)))) for i in range(n_seeds)])
    return outs.mean(0), outs.var(0, ddof=1)


def assert_same_distribution(a, b, n, zmax=5.0):
    (ma, va), (mb, vb) = a, b
    z = np.abs(ma - mb) / np.sqrt((va + vb) / n + 1e-300)
    assert np.nanmax(z) < zmax, f"max |z| = {np.nanmax(z):.2f}"
    assert np.median((va + 1e-12) / (vb + 1e-12)) == pytest.approx(1.0, rel=0.25)


# ------------------------------------------------------ 1. parity vs ref
@pytest.mark.parametrize("n", [7, 12])
@pytest.mark.parametrize("name", ["heading", "location", "x", "obs_variance"])
def test_matches_reference(name, n):
    w = world(n)
    expected = np.asarray(ref(name, w))
    tol = dict(atol=1e-9) if name == "heading" else dict(rtol=1e-7, atol=1e-8)
    for it in iters_for(n):
        set_mixed_map_iters(it)
        got = np.asarray(call(name, w))
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, **tol, err_msg=f"iters={it}")


# --------------------------------------------------------- 3. invariance
@pytest.mark.parametrize("n", [7, 12])
@pytest.mark.parametrize("name", list(CASES))
def test_chunk_size_invariance(name, n):
    w = world(n)
    set_mixed_map_iters(1)
    base = np.asarray(call(name, w))
    for it in iters_for(n)[1:]:
        set_mixed_map_iters(it)
        got = np.asarray(call(name, w))
        assert got.shape == base.shape
        np.testing.assert_allclose(got, base, rtol=1e-9, atol=1e-10, err_msg=f"iters={it}")


# ------------------------------------------------------------ 2. scales
def tiny_world():
    return make_world(3, t=12, seed=3)


def _scales_inputs(s_0):
    w = tiny_world()
    return {a: w[a] for a in "Y x v h Cd sigmasq nu_s".split()} | {"s_0": s_0}


@pytest.mark.parametrize("s0_kind", ["scalar", "full", "broadcast"])
def test_scales_distribution_matches_reference(s0_kind):
    n, t = 3, 12
    s_0 = {
        "scalar": 1.5,
        "full": jnp.asarray(np.random.default_rng(1).uniform(0.5, 3.0, (n, t, K))),
        "broadcast": jnp.asarray(np.random.default_rng(2).uniform(0.5, 3.0, (1, 1, K))),
    }[s0_kind]
    sub = _scales_inputs(s_0)
    set_mixed_map_iters(2)
    out = gibbs.resample_scales(seed_of(), **sub)
    assert out.shape == (n, t, K) and bool(jnp.isfinite(out).all()) and bool((out > 0).all())
    n_seeds = 200
    got = sample_stats(lambda sd: gibbs.resample_scales(sd, **sub), n_seeds, jnp.log)
    jref = jax.jit(lambda sd: _ref_scales(sd, **sub))
    want = sample_stats(jref, n_seeds, jnp.log)
    assert_same_distribution(got, want, n_seeds)


@pytest.mark.parametrize("s0_kind", ["scalar", "full"])
def test_scales_chunk_invariant_for_each_prior(s0_kind):
    s_0 = 1.5 if s0_kind == "scalar" else jnp.asarray(
        np.random.default_rng(1).uniform(0.5, 3.0, (3, 12, K))
    )
    sub = _scales_inputs(s_0)
    outs = []
    for it in (1, 2, 3, 7):
        set_mixed_map_iters(it)
        outs.append(np.asarray(gibbs.resample_scales(seed_of(3), **sub)))
    for o in outs[1:]:
        np.testing.assert_allclose(o, outs[0], rtol=1e-9, atol=1e-12)


def test_x_and_location_distribution_matches_reference():
    w = make_world(3, t=14, seed=3)
    n_seeds = 100
    set_mixed_map_iters(2)
    for name in ("x", "location"):
        jref = jax.jit(lambda sd, nm=name: ref(nm, w, sd))
        got = sample_stats(lambda sd, nm=name: call(nm, w, sd), n_seeds)
        want = sample_stats(jref, n_seeds)
        assert_same_distribution(got, want, n_seeds)


# ------------------------------------------- 4. parallel flag, end-to-end
@pytest.mark.parametrize("name", ["location", "x"])
def test_parallel_message_passing_flag(name):
    w = world(7)
    set_mixed_map_iters(3)
    outs = {}
    for flag in (True, False):
        got = np.asarray(CASES[name][0](seed_of(), **w, parallel_message_passing=flag))
        want = np.asarray(CASES[name][1](seed_of(), **w, parallel_message_passing=flag))
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=f"parallel={flag}")
        outs[flag] = got
    # distinct samplers consume randomness differently
    assert not np.allclose(outs[True], outs[False], atol=1e-3)


def tiny_model(n=3, t=20, k=K, seed=0):
    rng = np.random.default_rng(seed)
    Y = np.cumsum(rng.standard_normal((n, t, k, D)) * 0.1, axis=1)
    Y = Y + rng.standard_normal((n, 1, k, D)) * 3
    data = {"Y": jnp.asarray(Y), "mask": jnp.ones((n, t))}
    return data, kslds.init_model(
        data=data,
        seed=jr.PRNGKey(seed),
        anterior_idxs=[0],
        posterior_idxs=[k - 1],
        trans_hypparams=dict(num_states=S, alpha=5.7, gamma=1e3, kappa=1e4),
        ar_hypparams=dict(latent_dim=L, nlags=NLAGS, S_0_scale=0.01, K_0_scale=10.0),
        obs_hypparams=dict(sigmasq_0=0.1, sigmasq_C=0.1, nu_sigma=1e5, nu_s=5),
        cen_hypparams=dict(sigmasq_loc=0.5),
        PCA_fitting_num_frames=1000,
    )


def test_resample_model_end_to_end_chunk_stable():
    data, model = tiny_model()
    outs = {}
    for it in (1, 2, 3):
        set_mixed_map_iters(it)
        m = model
        for _ in range(2):
            m = kslds.resample_model(
                data, **m, parallel_message_passing=False, resample_global_noise_scale=True
            )
        outs[it] = m
        for key, val in m["states"].items():
            assert val.shape == model["states"][key].shape, key
            assert bool(jnp.isfinite(val).all()), key
        assert bool(jnp.isfinite(m["params"]["sigmasq"]).all())
    for it in (2, 3):
        for key in outs[1]["states"]:
            np.testing.assert_allclose(
                np.asarray(outs[it]["states"][key]),
                np.asarray(outs[1]["states"][key]),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{key} iters={it}",
            )
        np.testing.assert_allclose(
            np.asarray(outs[it]["params"]["sigmasq"]),
            np.asarray(outs[1]["params"]["sigmasq"]),
            rtol=1e-6,
        )


# ---------------------------------------------- 5. memory-shape regression
def _big_world():
    n, t, k = 7, 53, 5
    return (n, t, k), make_world(n, t=t, k=k)


def _n_leading_large(shapes, n, floor):
    return [s for s in shapes if s[:1] == (n,) and int(np.prod(s)) >= floor]


def test_no_full_dataset_temporaries_in_heading():
    (n, t, k), w = _big_world()
    w = {a: w[a] for a in "Y x v s Cd sigmasq".split()}
    set_mixed_map_iters(3)
    shapes = var_shapes("heading", w)
    assert (n, t, k, 2, 2) not in shapes
    assert not _n_leading_large(shapes, n, n * t * k * D)


def test_no_full_dataset_temporaries_in_location():
    (n, t, k), w = _big_world()
    w = {a: w[a] for a in "Y mask x h s Cd sigmasq sigmasq_loc".split()}
    set_mixed_map_iters(3)
    shapes = var_shapes("location", w)
    assert not _n_leading_large(shapes, n, n * t * k * D)


def test_no_full_dataset_temporaries_in_continuous_stateseqs():
    (n, t, k), w = _big_world()
    set_mixed_map_iters(3)
    shapes = var_shapes("x", w)
    assert not _n_leading_large(shapes, n, n * t * k * D)
    obs = k * D
    assert not [s for s in shapes if len(s) == 3 and s[0] == n and s[1] in (t, t - NLAGS + 1) and s[2] == obs]


def test_no_full_dataset_temporaries_in_scales_and_obs_variance():
    (n, t, k), w = _big_world()
    set_mixed_map_iters(3)
    # the (N, T, k) sqerr/scales arrays are inherent outputs/inputs; only
    # anything larger (e.g. (N,T,k,d) coordinates estimates) is a regression
    for name in ("scales", "obs_variance"):
        assert not _n_leading_large(var_shapes(name, w), n, n * t * k * D), name


# ------------------------------------------ 6. global iters is honored
@pytest.mark.parametrize("name", list(CASES))
def test_global_iters_changes_trip_count(name):
    w = world(12)
    set_mixed_map_iters(1)
    base = scan_lengths(name, w)
    set_mixed_map_iters(4)
    four = scan_lengths(name, w)
    assert 4 in four and 4 not in base, (base, four)
    set_mixed_map_iters(1)
    assert scan_lengths(name, w) == base  # re-traces back


# --------------------------------------------------------- 7. float32
@pytest.mark.skipif(not F64, reason="driver runs under x64")
def test_float32_smoke_subprocess():
    code = (
        "import numpy as np, jax.numpy as jnp\n"
        "import test_keypoint_slds_chunked as m\n"
        "from jax_moseq.utils import set_mixed_map_iters\n"
        "assert not m.jax.config.jax_enable_x64\n"
        "w = m.make_world(7, seed=5, dtype=np.float32)\n"
        "set_mixed_map_iters(2)\n"
        "for name in m.CASES:\n"
        "    ok_ref = bool(jnp.isfinite(m.ref(name, w)).all())\n"
        "    out = m.call(name, w)\n"
        "    ok = bool(jnp.isfinite(out).all())\n"
        "    print(name, ok_ref, ok)\n"
        "    assert ok or not ok_ref, name\n"
        "    assert out.dtype == jnp.float32, (name, out.dtype)\n"
    )
    here = Path(__file__).resolve().parent
    env = {**os.environ, "KPMS_F32": "1", "PYTHONPATH": f"{here.parent}:{here}"}
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, res.stdout + res.stderr
