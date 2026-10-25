import numpy as np
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
from jax.scipy.linalg import cho_factor, cho_solve
from textwrap import fill
import functools
from math import ceil
from typing import Optional

_MIXED_MAP_ITERS = 1
_MIXED_MAP_GPUS = 1


def concatenate_stateseqs(stateseqs, mask=None):
    """
    Concatenate state sequences, optionally applying a mask.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    Returns
    -------
    stateseqs_flat: ndarray
        1d array containing all state sequences
    """
    if isinstance(stateseqs, dict):
        stateseq_flat = np.hstack(list(stateseqs.values()))
    elif isinstance(stateseqs, list):
        stateseq_flat = np.hstack(stateseqs)
    elif mask is not None:
        stateseq_flat = stateseqs[mask[:, -stateseqs.shape[1] :] > 0]
    else:
        stateseq_flat = stateseqs.flatten()
    return stateseq_flat


def get_durations(stateseqs, mask=None):
    """
    Get durations for a batch of state sequences.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    Returns
    -------
    durations: 1d array
        The duration of each each state (across all state sequences)

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])
    }
    >>> get_durations(stateseqs)
    array([2, 3, 1, 3, 1])
    """
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)
    stateseq_padded = np.hstack([[-1], stateseq_flat, [-1]])
    changepoints = np.diff(stateseq_padded).nonzero()[0]
    return changepoints[1:] - changepoints[:-1]


def get_frequencies(stateseqs, mask=None, num_states=None, runlength=True):
    """
    Get state frequencies for a batch of state sequences.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    num_states: int, default=None
        Number of different states. If None, the number of states will
        be set to `max(stateseqs)+1`.

    runlength: bool, default=True
        Whether to count frequency by the number of instances of each
        state (True), or by the number of frames in each state (False).

    Returns
    -------
    frequencies: 1d array
        Frequency of each state across all state sequences

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])}
    >>> get_frequencies(stateseqs, runlength=True)
    array([0.2, 0.4, 0.2, 0.2])
    >>> get_frequencies(stateseqs, runlength=False)
    array([0.3, 0.3, 0.3, 0.1])
    """
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)

    if runlength:
        state_onsets = np.pad(np.diff(stateseq_flat).nonzero()[0] + 1, (1, 0))
        stateseq_flat = stateseq_flat[state_onsets]

    counts = np.bincount(stateseq_flat, minlength=0 if num_states is None else num_states)
    frequencies = counts / counts.sum()
    return frequencies


def symmetrize(A):
    """Symmetrize a matrix."""
    return (A + A.swapaxes(-1, -2)) / 2


def psd_solve(A, B, diagonal_boost=1e-6):
    """
    Solves the linear system Ax=B, assuming A is positive semi-definite.

    Uses Cholesky decomposition for improved numerical stability and
    efficiency. A is symmetrized and diagonal elements are boosted by
    ``diagonal_boost`` to ensure positive definiteness.

    Parameters
    ----------
    A: jax array, shape (n,n)
        A positive semi-definite matrix
    b: jax array, shape (...,n)

    Returns
    -------
    x: jax array, shape (...,n)
        Solution of the linear system Ax=b
    """
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), B)
    return x


def psd_inv(A, diagonal_boost=1e-6):
    """
    Invert a positive semi-definite matrix.

    Uses :py:func:`jax_moseq.utils.psd_solve` for numerical stability
    and ensures that the inverse matrix is symmetric.

    Parameters
    ----------
    A: jax array, shape (n,n)
        A positive semi-definite matrix

    Returns
    -------
    Ainv: jax array, shape (n,n)
        The inverse of A
    """
    Ainv = psd_solve(A, jnp.eye(A.shape[-1]), diagonal_boost=diagonal_boost)
    return symmetrize(Ainv)


def jax_io(fn):
    """
    Converts a function involving numpy arrays to one that inputs and
    outputs jax arrays.
    """
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs))
    )


def device_put_as_scalar(x):
    as_scalar = lambda arr: arr.item() if arr.shape == () else arr
    return jax.tree_util.tree_map(as_scalar, jax.device_put(x))


def apply_affine(x, Ab):
    return jnp.einsum("...ij, ...j->...i", Ab, pad_affine(x))


def pad_affine(x):
    """
    Pad ``x`` with 1's so that it can be affine transformed with matrix
    multiplication.
    """
    padding = jnp.ones((*x.shape[:-1], 1))
    xpadded = jnp.concatenate((x, padding), axis=-1)
    return xpadded


def fit_pca(Y, mask, PCA_fitting_num_frames=1000000, verbose=False, **kwargs):
    """
    Fit a PCA model to transformed keypoint coordinates.

    Parameters
    ----------
    Y: jax array, shape (..., d)
        Keypoint coordinates
    mask: jax array
        Binary indicator for which elements of ``Y`` are valid
    PCA_fitting_num_frames: int, default=1000000
        Maximum number of frames to use for PCA. Frames will be sampled
        randomly if the input data exceed this size.
    verbose: bool, default=False
        Whether to print the number of sampled frames.
    Returns
    -------
    pca, sklearn.decomposition._pca.PCA
        An sklearn PCA model fit to Y
    """
    Y_flat = Y[mask > 0]

    N = Y_flat.shape[0]
    N_sample = min(PCA_fitting_num_frames, N)
    rng = np.random.default_rng(42)
    sample = rng.choice(N, N_sample, replace=False)
    Y_sample = np.array(Y_flat)[sample]

    if verbose:
        print(f"PCA: Fitting PCA model to {N_sample} data points")
    pca = PCA(random_state=42).fit(Y_sample)
    return pca


def wrap_angle(x):
    """
    Wrap an angle to the range [-pi, pi].
    """
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def pad_along_axis(arr, pad_widths, axis=0, value=0):
    """
    Pad an array along a single axis

    Parameters
    -------
    arr: ndarray, Array to be padded
    pad_widths: tuple (int,int), Amount of padding on either end
    axis: int, Axis along which to add padding
    value: float, Value of padded array elements

    Returns
    _______
    padded_arr: ndarray
    """
    pad_left_shape = list(arr.shape)
    pad_right_shape = list(arr.shape)

    pad_left_shape[axis] = pad_widths[0]
    pad_right_shape[axis] = pad_widths[1]

    padding_left = jnp.ones(pad_left_shape, dtype=arr.dtype) * value
    padding_right = jnp.ones(pad_right_shape, dtype=arr.dtype) * value

    padded_arr = jnp.concatenate([padding_left, arr, padding_right], axis=axis)
    return padded_arr


def unbatch(data, keys, bounds):
    """
    Invert :py:func:`jax_moseq.utils.batch`

    Parameters
    ----------
    data: ndarray, shape (num_segs, seg_length, ...)
        Stack of segmented time-series

    keys: list or array of str, length num_segs
        Name of the time-series that each segment came from

    bounds: ndarray, shape (num_segs, 2)
        Start and end times for each segment, reflecting
        how the segments were extracted from the original
        time-series.

    Returns
    -------
    data_dict: dict
        Dictionary mapping names to reconstructed time-series
    """
    data_dict = {}
    for key in set(list(keys)):
        length = bounds[keys == key, 1].max()
        seq = np.zeros((int(length), *data.shape[2:]), dtype=data.dtype)
        for (s, e), d in zip(bounds[keys == key], data[keys == key]):
            seq[s:e] = d[: e - s]
        data_dict[key] = seq
    return data_dict


def batch(data_dict, keys=None, seg_length=None, seg_overlap=30):
    """
    Stack time-series data of different lengths into a single array for batch
    processing, optionally breaking up the data into fixed length segments. The
    data is padded so that the stacked array isn't ragged. The padding
    repeats the last frame of each time-series until the end of the segment.

    Parameters
    ----------
    data_dict: dict {str : ndarray}
        Dictionary mapping names to ndarrays, where the first dim
        represents time. All data arrays must have the same shape except
        for the first dim.

    keys: list of str, default=None
        Optional list of names specifying which datasets to include in
        the output and what order to put them in. Each name must be a
        key in ``data_dict``. If ``keys=None``, names will be sorted
        alphabetically.

    seg_length: int, default=None
        Break each time-series into segments of this length. If
        ``seg_length=None``, the final stacked array will be as long
        as the longest time-series.

    seg_overlap: int, default=30
        Amount of overlap between segments. For example, setting
        ``seg_length=N`` and ``seg_overlap=M`` will result in segments
        with start/end times (0, N+M), (N, 2*N+M), (2*N, 3*N+M),...

    Returns
    -------
    data: ndarray, shape (N, seg_length, ...)
        Stacked data array

    mask: ndarray, shape (N, seg_length)
        Binary indicator specifying which elements of ``data`` are not
        padding (``mask==0`` in padded locations)

    metadata: tuple (keys, bounds)
        Metadata for the rows of `data`, as a tuple with an array of keys
        and an array of (start,end) times.
    """
    if keys is None:
        keys = sorted(data_dict.keys())
    Ns = [len(data_dict[key]) for key in keys]
    if seg_length is None:
        seg_length = np.max(Ns)

    stack, mask, keys_out, bounds = [], [], [], []
    for key, N in zip(keys, Ns):
        for start in range(0, N, seg_length):
            arr = data_dict[key]
            end = min(start + seg_length + seg_overlap, N)
            pad_length = seg_length + seg_overlap - (end - start)
            padding = np.repeat(arr[end - 1 : end], pad_length, axis=0)
            mask.append(np.hstack([np.ones(end - start), np.zeros(pad_length)]))
            stack.append(np.concatenate([arr[start:end], padding], axis=0))
            keys_out.append(key)
            bounds.append((start, end))

    stack = np.stack(stack)
    mask = np.stack(mask)
    metadata = (np.array(keys_out), np.array(bounds))
    return stack, mask, metadata


def get_mixed_map_iters():
    """Get the number of iterations to use for jax.lax.map in
    :py:func:`jax_moseq.utils.mixed_map`."""
    return _MIXED_MAP_ITERS


def set_mixed_map_iters(iters):
    """Set the number of iterations to use for jax.lax.map in
    :py:func:`jax_moseq.utils.mixed_map`."""
    global _MIXED_MAP_ITERS
    _MIXED_MAP_ITERS = iters


def get_mixed_map_gpus():
    """Get the number of GPUs to use for jax.pmap in
    :py:func:`jax_moseq.utils.mixed_map`."""
    return _MIXED_MAP_GPUS


def set_mixed_map_gpus(gpus):
    """Set the number of GPUs to use for jax.pmap in
    :py:func:`jax_moseq.utils.mixed_map`."""
    global _MIXED_MAP_GPUS
    _MIXED_MAP_GPUS = gpus


def _as_axes(axes, n: int, name: str) -> tuple:
    """Broadcast an int/None `axes` spec to a tuple of length `n`."""
    if axes is None or isinstance(axes, int):
        return (0 if axes is None else axes,) * n
    assert len(axes) == n, f"`{name}` should be a tuple with length {n}, got {len(axes)}"
    return tuple(axes)


def mixed_map(
    fun,
    in_axes=None,
    out_axes=None,
    iters: Optional[int] = None,
    gpus: Optional[int] = None,
):
    """
    Memory-efficient drop-in replacement for `jax.vmap(fun, in_axes, out_axes)`.

    The mapped axis (of size N) is processed in serial chunks with
    `jax.lax.fori_loop`; each chunk is evaluated with `jax.vmap` (and
    `jax.pmap` across `gpus` devices if `gpus > 1`). Rows must be independent,
    i.e. `fun` must not mix information across the mapped axis.

    Parameters
    ----------
    iters: int, optional
        Target number of serial steps (>= 1). Each step processes
        `ceil(N / iters)` rows (rounded up to a multiple of `gpus`). Defaults
        to :py:func:`jax_moseq.utils.get_mixed_map_iters`, read at call time.
    gpus: int, optional
        Number of devices for `jax.pmap` within each step. Defaults to
        :py:func:`jax_moseq.utils.get_mixed_map_gpus`, read at call time.
        If rounding a chunk up to a multiple of `gpus` would exceed N, a
        single N-row step is used when `gpus` divides N, otherwise `gpus=1`.

    Inputs are sliced inside the loop and outputs are written into buffers
    allocated once, so no padded or reshaped copies of the full arrays are
    made. When N is not a multiple of the step size, the last step overlaps
    the previous one (recomputing identical rows) instead of padding.
    """

    @functools.wraps(fun)
    def mixed_map_f(*args):
        n_iters = get_mixed_map_iters() if iters is None else iters
        n_gpus = get_mixed_map_gpus() if gpus is None else gpus
        if n_iters < 1 or n_gpus < 1:
            raise ValueError(f"`iters` and `gpus` must be >= 1, got {n_iters}, {n_gpus}")

        axes = _as_axes(in_axes, len(args), "in_axes")
        mapped = [i for i, ax in enumerate(axes) if ax is not None]
        if not mapped:
            raise ValueError("at least one argument must be mapped")
        N = args[mapped[0]].shape[axes[mapped[0]]]

        # step = rows per serial iteration = vmap_size * n_gpus <= N
        vmap_size = ceil(ceil(N / n_iters) / n_gpus)
        if vmap_size * n_gpus > N:
            n_gpus = n_gpus if N % n_gpus == 0 else 1
            vmap_size = N // n_gpus
        step = vmap_size * n_gpus
        n_steps = ceil(N / step)
        mapped_axes = tuple(axes[i] for i in mapped)

        def call(*mapped_args):
            full = list(args)
            for i, a in zip(mapped, mapped_args):
                full[i] = a
            return fun(*full)

        def run_chunk(start):
            chunk = [
                jax.lax.dynamic_slice_in_dim(args[i], start, step, axis=ax)
                for i, ax in zip(mapped, mapped_axes)
            ]
            if n_gpus > 1:
                chunk = [
                    a.reshape(*a.shape[:ax], n_gpus, vmap_size, *a.shape[ax + 1 :])
                    for a, ax in zip(chunk, mapped_axes)
                ]
            outs = chunk_fn(*chunk)
            if n_gpus > 1:  # merge (n_gpus, vmap_size) at each out axis
                outs = jax.tree_util.tree_map(
                    lambda o, ax: o.reshape(*o.shape[:ax], step, *o.shape[ax + 2 :]),
                    outs,
                    out_axes_tree,
                )
            return outs

        # output structure/axes are only known after tracing `fun`
        out_struct = jax.eval_shape(
            jax.vmap(call, in_axes=mapped_axes),
            *[
                jax.ShapeDtypeStruct(
                    args[i].shape[:ax] + (1,) + args[i].shape[ax + 1 :], args[i].dtype
                )
                for i, ax in zip(mapped, mapped_axes)
            ],
        )
        n_out = len(out_struct) if isinstance(out_struct, (tuple, list)) else 1
        out_axes_tree = _as_axes(out_axes, n_out, "out_axes")
        if not isinstance(out_struct, (tuple, list)):
            out_axes_tree = out_axes_tree[0]
        else:
            out_axes_tree = type(out_struct)(out_axes_tree)

        vmapped = jax.vmap(call, in_axes=mapped_axes, out_axes=out_axes_tree)
        if n_gpus > 1:
            chunk_fn = jax.pmap(
                vmapped,
                in_axes=mapped_axes,
                out_axes=out_axes_tree,
            )
        else:
            chunk_fn = vmapped

        if n_steps == 1:  # step == N: no loop or buffers needed
            return run_chunk(0)

        def alloc(spec, ax):
            # `spec` was traced with a size-1 mapped axis at position 0
            shape = list(spec.shape[1:])
            shape.insert(ax % (len(shape) + 1), N)
            return jnp.zeros(shape, spec.dtype)

        bufs = jax.tree_util.tree_map(alloc, out_struct, out_axes_tree)

        def body(i, bufs):
            start = jnp.minimum(i * step, N - step)
            outs = run_chunk(start)
            return jax.tree_util.tree_map(
                lambda b, o, ax: jax.lax.dynamic_update_slice_in_dim(b, o, start, axis=ax),
                bufs,
                outs,
                out_axes_tree,
            )

        return jax.lax.fori_loop(0, n_steps, body, bufs)

    return mixed_map_f


def mixed_map_jit(fun=None, *, static_argnames=()):
    """`jax.jit` that follows the global `mixed_map` settings.

    A jitted function that calls `mixed_map` reads `get_mixed_map_iters()` once
    at trace time, so later calls to `set_mixed_map_iters` would be silently
    ignored. This decorator passes the current iters/gpus to `fun` as the
    static keyword arguments `mixed_map_iters` and `mixed_map_gpus` (forward
    them to `mixed_map(..., iters=..., gpus=...)`), so changing the globals
    re-traces instead.
    """

    def decorator(fun):
        jitted = jax.jit(
            fun, static_argnames=(*static_argnames, "mixed_map_iters", "mixed_map_gpus")
        )

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return jitted(
                *args,
                mixed_map_iters=get_mixed_map_iters(),
                mixed_map_gpus=get_mixed_map_gpus(),
                **kwargs,
            )

        return wrapper

    return decorator if fun is None else decorator(fun)
