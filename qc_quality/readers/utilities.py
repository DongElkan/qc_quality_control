"""
This module provides utilities for the package.
"""
import numpy as np
import numba as nb


@nb.njit("UniTuple(int64, 2)(float64[:], float64, float64)")
def match_array(x, v, t):
    """
    Match values in array in the range.

    Args:
        x: Array for match.
        v: Value.
        t: Tolerance of value to match.

    Returns:
        int: Index of lower bound value.
        int: Index of upper bound value.

    """
    vl = v - t
    vr = v + t
    lo = np.searchsorted(x, vl)
    hi = np.searchsorted(x, vr)
    n = x.size
    if hi < n and vr == x[hi]:
        # occasionally, the right side value equals to upper bound
        hi += 1
    return lo, hi


@nb.njit("int64[:](int64[:], int64)")
def match_index(x, k) -> np.ndarray:
    """
    Matches array to get index.

    Args:
        x: Array matched against.
        k: Value to find.

    Returns:
        array: Index.

    """
    n = x.size
    index = np.zeros(n, dtype=x.dtype)
    j = 0
    for i in range(n):
        if x[i] == k:
            index[j] = i
            j += 1
    return index[:j]
