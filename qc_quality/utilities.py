"""
This module provides utilities for the package.
"""
import numba as nb
import numpy as np


@nb.njit("UniTuple(float64[:], 2)(float64[:], float64[:], float64)")
def _group_mass(mass, dist, tol):
    """
    Groups mass and sums up value in dist for same mass group.

    Args:
        mass: Array of the mass
        dist: Array of distribution values.
        tol: Mass tolerance for grouping.

    Returns:
        Array: Grouped mass.
        Array: Grouped distribution values.

    """
    n = mass.size
    ix = mass.argsort()
    k = 0

    g_mass = np.zeros(n, dtype=nb.float64)
    g_dist = np.zeros(n, dtype=nb.float64)
    g_mass[k] = mass[ix[k]]
    g_dist[k] = dist[ix[k]]
    for i in range(1, n):
        j = ix[i]
        if mass[j] - g_mass[k] > tol:
            # if mass is changed, move to next one and update
            k += 1
            g_mass[k] = mass[j]
        g_dist[k] += dist[j]

    return g_mass[:k + 1], g_dist[:k + 1]


@nb.njit("UniTuple(int64, 2)(float64[:], float64[:], float64, float64)")
def _get_mass_range(mass, intensity, mass_tol, thr):
    """
    Determines the range of mass for acceptance.

    Args:
        mass: Mass.
        intensity: Intensity.
        mass_tol: Tolerance to determine the range.
        thr: Relative intensity threshold to determine the range.

    Returns:
        Left and right index.

    """
    k = intensity.argmax()
    m = mass[k]
    mt = intensity[k]
    j0 = np.searchsorted(mass, m - mass_tol)
    j1 = np.searchsorted(mass, m + mass_tol)
    val_j0 = j0
    for i in range(j0, k):
        if intensity[i] / mt >= thr:
            break
        val_j0 = i

    val_j1 = j1
    for i in range(k, j1):
        if intensity[i] / mt >= thr:
            val_j1 = i

    return val_j0, min(val_j1 + 1, j1)


@nb.njit(["float64(int64[:], float64[:])", "float64(float64[:], float64[:])"],
         fastmath=True)
def dot(x, y):
    n = x.size
    s = 0.
    for i in range(n):
        s += x[i] * y[i]
    return s


@nb.njit("float64[:](int64[:,:], int64)", fastmath=True)
def multinomial_coef_log(p, n):
    """
    Calculates multinomial coefficients, log transformed.

    Args:
        p: Matrix of No. of atoms.
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
                sk += logn[nk - 1]
        coef[i] = tn - sk

    return coef


@nb.njit("float64(int64, int64)", fastmath=True)
def binomial_coef_log(n, k):
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


@nb.njit("UniTuple(float64[:], 2)(float64[:], float64[:], int64)",
         fastmath=True)
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
        c = binomial_coef_log(n, n - i)
        p = (n - i) * np.log(prob[0]) + i * np.log(prob[1])
        dist_int[i] = np.exp(c + p)

    # keep [-20, 20] around mass of top peak intensity and relative intensity
    # higher than 1e-8.
    j0, j1 = _get_mass_range(dist_mass, dist_int, 20., 1e-10)
    return dist_mass[j0: j1], dist_int[j0: j1]


@nb.njit("UniTuple(float64[:], 2)(float64[:], float64[:,:], float64[:,:], "
         "int64[:,:])", fastmath=True)
def element_isotope_distn(coef, mass, prob, parts):
    """
    Calculates isotopic dist for elements with more than 3 isotopes.

    Args:
        coef: Multinomial coefficients, n unique coefficients by 1, log
            transformed.
        mass: Mass of elemental isotopes, m permutations by k isotopes.
        prob: Probability of elemental isotopes, m permutations by k
            isotopes.
        parts: Partitions of m atoms, n by k.

    Returns:
        Array: Mass of isotopic variants.
        Array: Intensity of isotopic variants.

    """
    # Pre-allocation of arrays using max mass of the distribution.
    n = coef.size
    m, c = mass.shape

    # log probability
    log_prob = np.zeros((m, c), dtype=nb.float64)
    for i in range(m):
        for j in range(c):
            log_prob[i][j] = np.log(prob[i][j])

    iso_mass = np.zeros(n * m, dtype=nb.float64)
    iso_int = np.zeros(n * m, dtype=nb.float64)
    k = 0
    for i in range(n):
        for j in range(m):
            iso_mass[k] = dot(parts[i], mass[j])
            lp = dot(parts[i], log_prob[j])
            iso_int[k] = np.exp(lp + coef[i])
            k += 1

    # get final dist with mass [-20, 20] around mass of max intensity peak.
    f_iso_mass, f_iso_int = _group_mass(iso_mass, iso_int, 1e-6)
    j0, j1 = _get_mass_range(f_iso_mass, f_iso_int, 20., 1e-12)
    return f_iso_mass[j0: j1], f_iso_int[j0: j1]


@nb.njit("UniTuple(float64[:], 2)(float64[:], float64[:], float64[:], "
         "float64[:])", fastmath=True)
def element_dist_combine(dist_mass1, dist_int1, dist_mass2, dist_int2):
    """
    Combines distribution of two kinds of elements for the distribution
    of molecule.

    Args:
        dist_mass1: Mass of distribution of element 1.
        dist_int1: Intensity of distribution of element 1.
        dist_mass2: Mass of distribution of element 2.
        dist_int2: Intensity of distribution of element 2.

    Returns:
        Array: Mass of distribution of combined element.
        Array: Intensity of distribution of combined element.

    """
    n1 = dist_mass1.size
    n2 = dist_mass2.size
    dist_mass = np.zeros(n1 * n2, dtype=nb.float64)
    dist_int = np.zeros(n1 * n2, dtype=nb.float64)
    k = 0
    for i in range(n1):
        for j in range(n2):
            dist_mass[k] = dist_mass1[i] + dist_mass2[j]
            dist_int[k] = dist_int1[i] * dist_int2[j]
            k += 1

    f_dist_mass, f_dist_int = _group_mass(dist_mass, dist_int, 1e-6)

    # selective peaks
    j0, j1 = _get_mass_range(f_dist_mass, f_dist_int, 30., 1e-8)
    return f_dist_mass[j0: j1], f_dist_int[j0: j1]


@nb.njit("UniTuple(float64[:], 2)(float64[:], float64[:])", fastmath=True)
def centroid_mass(mass, dist_int):
    """
    Centroids mass. All mass are firstly rounded to the nearest
    integers. All intensities with mass rounded to an integer are
    summed up. The accurate mass is weighted average of all mass
    rounded to that integer.

    Args:
        mass: Mass.
        dist_int: Intensity of distribution.

    Returns:
        Array: centroided (accurate) mass.
        Array: Intensity.

    """
    # pre-allocate arrays for mass and intensity, with weight sum.
    m = mass.size
    cent_mass = np.zeros(m, dtype=nb.float64)
    cent_int = np.zeros(m, dtype=nb.float64)
    round_mass = np.zeros(m, dtype=nb.int64)

    # sor the mass
    ix = mass.argsort()
    k = 0
    j = ix[k]
    cent_mass[k] = mass[j] * dist_int[j]
    cent_int[k] = dist_int[j]
    round_mass[k] = round(mass[j])
    for i in ix[1:]:
        mk = round(mass[i])
        if mk - round_mass[k] > 0:
            # update for next integer
            k += 1
            cent_mass[k] = mass[i] * dist_int[i]
            cent_int[k] = dist_int[i]
            round_mass[k] = mk
        else:
            cent_int[k] += dist_int[i]
            cent_mass[k] += mass[i] * dist_int[i]

    # accurate mass
    for i in range(k + 1):
        cent_mass[i] /= cent_int[i]

    return cent_mass[:k+1], cent_int[:k+1]
