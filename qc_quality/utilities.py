"""
This module provides utilities for the package.
"""
import numba as nb
import numpy as np


@nb.njit("float64[int64, int64]", fastmath=True)
def log_binomial_coefficient(n, k):
    """
    Logarithm transformed binomial coefficients.

    Args:
        n: Power.
        k: No. of elements.

    Returns:
        float: Coefficient.

    """
    if k < 0 or k > n:
        return -np.inf
    if k == 0 or k == n:
        return 0.

    k = min(k, n - k)
    c = 0.
    for i in range(k):
        c += np.log(n - i) - np.log(i + 1)

    return c


@nb.njit("float64[:](float64[:], float64[:], int64)", fastmath=True)
def element_isotope_dist2(mass, prob, n):
    """
    Calculates isotopic distribution for element with two isotopes.

    Args:
        mass: Mass of isotopic atom.
        prob: Probability of occurrence of isotopic atom.
        n: Number of the element.

    Returns:
        Array: isotopic distribution.

    """
    dist_mass = np.zeros(n + 1, dtype=nb.float64)
    dist_int = np.zeros(n + 1, dtype=nb.float64)
    for i in range(n + 1):
        dist_mass[i] = mass[0] * (n - i) + mass[1] * i
        c = log_binomial_coefficient(n, n - i)
        p = (n - i) * np.log(prob[0]) + i * np.log(prob[1])
        dist_int[i] = np.exp(c + p)

    # keep top 10 peaks
    k = dist_int.argmax()
    j0 = max(k - 5, 0)
    j1 = min(n + 1, k + 5)
    return dist_mass[j0: j1], dist_int[j0: j1]


@nb.njit("float64[:](int64[:,:], int64)", fastmath=True)
def multinomial_coef_log(p, n):
    """
    Calculates multinomial coefficients, log transformed.

    Args:
        p: Matrix of No. of elements.
        n: No.

    Returns:
        Array: coefficients.

    """
    # cumulative sum of log transformed range
    logn = np.zeros(n, dtype=nb.float64)
    for i in range(1, n):
        logn[i] = logn[i - 1] + np.log(i + 1)
    tn = logn[-1]

    r, c = p.shape
    coef = np.zeros(r, dtype=nb.float64)
    for i in range(r):
        sk = 0.
        for j in range(c):
            nk = p[i][j]
            if nk > 0:
                sk += logn[nk]
        coef[i] = tn - sk

    return coef
