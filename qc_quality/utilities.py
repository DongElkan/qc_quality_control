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
    return dist_mass, dist_int



