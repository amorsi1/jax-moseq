"""Tests for ``jax_moseq.utils.mixed_map``.

``mixed_map`` must be semantically identical to ``jax.vmap`` for row-independent
functions, for any axis size N >= 1 and any ``iters`` >= 1, while never making
full-size copies of its inputs. Everything runs on tiny synthetic arrays.
"""

import os
import subprocess
import sys
import textwrap

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.utils import (  # noqa: E402
    get_mixed_map_iters,
    mixed_map,
    set_mixed_map_iters,
)

NS = [1, 2, 7, 12, 13, 31]
ITERS = [1, 2, 3, 5, "N", "N+5"]
D_IN, D_OUT = 5, 3


def resolve_iters(iters, n):
    return {"N": n, "N+5": n + 5}.get(iters, iters)


def assert_tree_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_allclose(np.asarray(x, float), np.asarray(y, float), atol=1e-10)


def make_data(n, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((n, D_IN)))
    w = jnp.asarray(rng.standard_normal((D_IN, D_OUT)))
    y = jnp.asarray(rng.standard_normal((n, 4, D_IN)))
    return x, w, y


def single_out(x, w, y):
    return (x @ w) * jnp.sum(y)


def multi_out(x, w, y):
    h = x @ w
    return h, jnp.sum(y * x, axis=-1), (jnp.sum(h) > 0), jnp.argmax(h)


@pytest.fixture
def restore_iters():
    old = get_mixed_map_iters()
    yield
    set_mixed_map_iters(old)


@pytest.mark.parametrize("fun", [single_out, multi_out])
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("iters", ITERS)
def test_parity_with_vmap(fun, n, iters):
    x, w, y = make_data(n)
    axes = (0, None, 0)
    expected = jax.vmap(fun, axes)(x, w, y)
    got = mixed_map(fun, axes, iters=resolve_iters(iters, n), gpus=1)(x, w, y)
    assert_tree_equal(got, expected)


@pytest.mark.parametrize("iters", [1, 3, 20])
def test_mapped_axis_not_zero_and_out_axes(iters):
    n = 13
    x, w, y = make_data(n)
    xt = x.T  # (D_IN, n), map axis 1

    def fun(xr, w, yr):
        h = xr @ w
        return h, jnp.sum(yr, axis=0)

    axes, out_axes = (1, None, 0), (1, 0)
    expected = jax.vmap(fun, axes, out_axes)(xt, w, y)
    got = mixed_map(fun, axes, out_axes, iters=iters, gpus=1)(xt, w, y)
    assert got[0].shape == (D_OUT, n)
    assert_tree_equal(got, expected)


def test_dtypes_preserved():
    n = 7
    x = jnp.arange(n * 2, dtype=jnp.int32).reshape(n, 2)

    def fun(r):
        return r.astype(jnp.float32) * 0.5, r + 1, r > 3, jnp.sum(r) % 2 == 0

    got = mixed_map(fun, iters=3, gpus=1)(x)
    expected = jax.vmap(fun)(x)
    for g, e in zip(got, expected):
        assert g.dtype == e.dtype
    assert [g.dtype for g in got] == [jnp.float32, jnp.int32, jnp.bool_, jnp.bool_]
    assert_tree_equal(got, expected)


def test_per_row_prng_chunk_invariant():
    n = 13
    keys = jr.split(jr.PRNGKey(0), n)

    def fun(key, scale):
        k1, k2 = jr.split(key)
        return jr.normal(k1, (4,)) * scale, jr.randint(k2, (), 0, 100)

    expected = jax.vmap(fun, (0, None))(keys, 2.0)
    for iters in [1, 2, 3, 5, n, n + 5]:
        got = mixed_map(fun, (0, None), iters=iters, gpus=1)(keys, 2.0)
        assert_tree_equal(got, expected)


def test_reuse_different_n_and_arg_counts():
    f = mixed_map(lambda a: a * 2 + 1, iters=3, gpus=1)
    for n in [5, 12, 7]:
        a = jnp.arange(n * 2.0).reshape(n, 2)
        assert_tree_equal(f(a), a * 2 + 1)

    g = mixed_map(lambda *a: sum(a), iters=3, gpus=1)  # in_axes=None
    a = jnp.arange(10.0)
    assert_tree_equal(g(a), a)
    assert_tree_equal(g(a, a * 2), a * 3)
    assert_tree_equal(g(a, a * 2, a * 3), a * 6)
    assert_tree_equal(g(a), a)

    h = mixed_map(lambda a: (a, a + 1), iters=2, gpus=1)  # tuple then reuse
    assert_tree_equal(h(a), (a, a + 1))
    assert_tree_equal(h(a[:3]), (a[:3], a[:3] + 1))


def _trip_counts(closed):
    return [e.params["length"] for e in _walk_eqns(closed) if e.primitive.name == "scan"]


def test_global_iters_honored_and_overridden(restore_iters):
    n = 12
    x, w, y = make_data(n)
    axes = (0, None, 0)
    expected = jax.vmap(single_out, axes)(x, w, y)
    f = mixed_map(single_out, axes, gpus=1)
    for k in [1, 3, 4]:
        set_mixed_map_iters(k)
        assert_tree_equal(f(x, w, y), expected)
        trips = _trip_counts(jax.make_jaxpr(f)(x, w, y))
        if trips:  # loop primitive form may vary; only check when visible
            assert k in trips or k == 1
    set_mixed_map_iters(4)
    explicit = mixed_map(single_out, axes, iters=2, gpus=1)
    assert_tree_equal(explicit(x, w, y), expected)
    trips = _trip_counts(jax.make_jaxpr(explicit)(x, w, y))
    if trips:
        assert 4 not in trips


def test_jit_static_iters():
    n = 13
    x, w, y = make_data(n)
    axes = (0, None, 0)
    expected = jax.vmap(multi_out, axes)(x, w, y)

    @jax.jit
    def run_static(x, w, y, iters):
        return mixed_map(multi_out, axes, iters=iters, gpus=1)(x, w, y)

    run = jax.jit(run_static.__wrapped__, static_argnames="iters")
    for iters in [2, 4, 20]:
        assert_tree_equal(run(x, w, y, iters=iters), expected)


def _sub_jaxprs(value):
    if hasattr(value, "jaxpr") and hasattr(value, "consts"):  # ClosedJaxpr
        yield value.jaxpr
    elif hasattr(value, "eqns"):  # Jaxpr
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _sub_jaxprs(v)


def _walk_eqns(closed):
    stack = [closed.jaxpr if hasattr(closed, "jaxpr") else closed]
    while stack:
        jaxpr = stack.pop()
        for eqn in jaxpr.eqns:
            yield eqn
            for v in eqn.params.values():
                stack.extend(_sub_jaxprs(v))


def test_no_full_size_copies():
    n, iters = 13, 4  # 13 not divisible by chunk size 4
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((n, 17, 6)))
    z = jnp.asarray(rng.standard_normal((n, 9)))
    w = jnp.asarray(rng.standard_normal((6, 8)))

    def fun(xr, wr, zr):
        return jnp.sum(xr @ wr, axis=0), zr * 2.0

    f = mixed_map(fun, (0, None, 0), iters=iters, gpus=1)
    expected = jax.vmap(fun, (0, None, 0))(x, w, z)
    assert_tree_equal(f(x, w, z), expected)

    bad = []
    for eqn in _walk_eqns(jax.make_jaxpr(f)(x, w, z)):
        name = eqn.primitive.name
        if name in ("concatenate", "pad"):
            shapes = [v.aval.shape for v in [*eqn.invars, *eqn.outvars] if hasattr(v, "aval")]
            if any(s and s[0] >= n for s in shapes):
                bad.append((name, shapes))
        elif name == "transpose":
            if any(v.aval.shape and v.aval.shape[0] == n for v in eqn.invars):
                bad.append((name, [v.aval.shape for v in eqn.invars]))
    assert not bad, f"full-size copies found: {bad}"


def test_multi_device_pmap_subprocess():
    script = textwrap.dedent(
        """
        import jax, jax.numpy as jnp, numpy as np
        jax.config.update("jax_enable_x64", True)
        from jax_moseq.utils import mixed_map
        assert jax.local_device_count() == 2, jax.local_device_count()
        fun = lambda x, w: (x @ w, jnp.sum(x) > 0)
        w = jnp.arange(12.0).reshape(4, 3)
        for n in (6, 7):
            x = jnp.arange(n * 4.0).reshape(n, 4) - 5
            ref = jax.vmap(fun, (0, None))(x, w)
            for iters in (1, 2, 3):
                out = mixed_map(fun, (0, None), iters=iters, gpus=2)(x, w)
                for o, r in zip(out, ref):
                    assert o.dtype == r.dtype
                    np.testing.assert_allclose(np.asarray(o, float), np.asarray(r, float))
        print("OK")
        """
    )
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2", "JAX_PLATFORMS": "cpu"}
    res = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr


def test_invalid_iters_raises():
    f = mixed_map(lambda a: a + 1, iters=0, gpus=1)
    with pytest.raises(ValueError):
        f(jnp.arange(4.0))
    with pytest.raises(ValueError):
        mixed_map(lambda a: a + 1, iters=-2, gpus=1)(jnp.arange(4.0))


def test_in_axes_length_mismatch_raises():
    f = mixed_map(lambda a, b: a + b, (0, 0, 0), iters=2, gpus=1)
    a = jnp.arange(4.0)
    with pytest.raises((AssertionError, ValueError)):
        f(a, a)
