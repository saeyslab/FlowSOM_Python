# https://github.com/numba/numba/issues/1269

import numba as nb
import numpy as np


@nb.njit
def apply_along_axis_0(func1d, arr):
    """Like calling func1d(arr, axis=0)."""
    if arr.size == 0:
        raise RuntimeError("Must have arr.size > 0")
    ndim = arr.ndim
    if ndim == 0:
        raise RuntimeError("Must have ndim > 0")
    elif 1 == ndim:
        return func1d(arr)
    else:
        result_shape = arr.shape[1:]
        out = np.empty(result_shape, arr.dtype)
        _apply_along_axis_0(func1d, arr, out)
        return out


@nb.njit
def _apply_along_axis_0(func1d, arr, out):
    """Like calling func1d(arr, axis=0, out=out). Require arr to be 2d or bigger."""
    ndim = arr.ndim
    if ndim < 2:
        raise RuntimeError("_apply_along_axis_0 requires 2d array or bigger")
    elif ndim == 2:  # 2-dimensional case
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:  # higher dimensional case
        for i, out_slice in enumerate(out):
            _apply_along_axis_0(func1d, arr[:, i], out_slice)


@nb.njit
def nb_mean_axis_0(arr):
    """Like calling np.mean(arr, axis=0)."""
    return apply_along_axis_0(np.mean, arr)


@nb.njit
def nb_std_axis_0(arr):
    """Like calling np.std(arr, axis=0)."""
    return apply_along_axis_0(np.std, arr)


@nb.njit
def nb_amax_axis_0(arr):
    """Like calling np.amax(arr, axis=0)."""
    return apply_along_axis_0(np.amax, arr)


@nb.njit
def nb_median_axis_0(arr):
    """Like calling np.median(arr, axis=0)."""
    return apply_along_axis_0(np.median, arr)
