"""
This module provides natural cubic spline smoothing with automatic
parameter selection using improved AIC.

References:
    [1] Eubank RL. Nonparametric Regression and Spline Smoothing. 2nd
        Ed. New York; Basel: Marcel Dekker. 1999.
    [2] Green PJ, Silverman BW. Nonparametric Regression and
        Generalized Linear Models: A roughness penalty approach.
        Chapman and Hall/CRC. 1993.
    [3] Hurvich CM, Simonoff JS, Tsai CL. Smoothing Parameter Selection
        in Nonparametric Regression Using an Improved Akaike
        Information Criterion. J R Statist Soc B. 1998, 60, 271-293.
    [4] Lee TCM. Smoothing parameter selection for smoothing splines: a
        simulation study. Comput Stat Data Anal. 2003, 42, 139–148.

"""
import numpy as np
import numba as nb


@nb.njit("UniTuple(float64[:,:], 2)(float64[:])")
def cubic_splines(x):
    """
    Generates cubic spline matrix.

    Args:
        x: Knots

    Returns:
        array: Spline matrix

    """
    n = x.size
    h = x[1:] - x[:-1]
    h2 = 1. / h
    r = np.zeros((n-2, n-2), dtype=np.float64)
    q = np.zeros((n, n-2), dtype=np.float64)
    for i in range(1, n-1):
        j = i - 1
        q[j][i] = h2[j]
        q[i][i] = -(h2[j] + h2[i])
        q[i+1][i] = h2[i]

    for i in range(1, n-2):
        j = i - 1
        r[j][j] = (h[j] + h[i]) / 3
        r[j][i] = h[i] / 6
        r[i][j] = h[i] / 6
