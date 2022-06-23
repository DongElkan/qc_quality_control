"""
This module provides functions to extract XIC from LC-MS data.

The algorithm is based on reference [1].

References:
    [1] Myers OD, et al., One Step Forward for Reducing False Positive
        and False Negative Compound Identifications from Mass
        Spectrometry Metabolomics Data: New Algorithms for Constructing
        Extracted Ion Chromatograms and Detecting Chromatographic
        Peaks. Anal Chem. 2017, 89, 8696–8703.

"""
import numba as nb
import numpy as np


@nb.njit("UniTuple(float64[:], 2)(float64[:], float64[:], float64)")
def centroid(mz, intensity, tol):
    """
    Centroids mass spectrum.

    Args:
        mz: Mass spectrum m/z.
        intensity: Mass spectrum intensity.
        tol: Tolerance, in ppm.

    Returns:
        array: Centroided m/z.
        array: Centroided intensity.

    """
    six = intensity.argsort()
    n = mz.size

    checked = np.zeros(n, dtype=nb.int64)
    cent_mz = np.zeros(n, dtype=np.float64)
    cent_intensity = np.zeros(n, dtype=np.float64)
    k = 0
    for i in six:
        if checked[i] == 1:
            # has been assigned
            continue
        mzk = mz[i]

        cent_mz[k] = mzk
        cent_intensity[k] = intensity[i]
        k += 1

        err = mzk * tol / 1e6
        # search left
        for jl in range(i, -1, -1):
            if mzk - mz[jl] > err:
                break
            checked[jl] = 1
        # search right
        for jr in range(i + 1, n):
            if mz[jr] - mzk > err:
                break
            checked[jr] = 1

        checked[i] = 1

    return cent_mz[:k], cent_intensity[:k]


@nb.njit("Tuple((int64[:], int64))(float64[:], float64, float64)")
def _check_in_mz(mz_arr, mz, tol):
    """ Checks whether a m/z in a m/z array. """
    n = mz_arr.size
    ix = np.zeros(n, dtype=nb.int64)
    k = 0
    for i in range(n):
        if abs(mz_arr[i] - mz) <= tol:
            ix[k] = i
            k += 1
    return ix[:k], k


@nb.njit(
    "UniTuple(float64[:], 2)(float64[:,:], float64[:,:], float64, float64)")
def extract_top_mz(ms1_mz, ms1_intensity, tol, thr):
    """ Extracts XIC in top intensity. """
    nsel = 1000
    sel_mz = np.zeros(nsel, dtype=nb.float64)
    sel_int = np.zeros(nsel, dtype=nb.float64)
    nms, nmz = ms1_mz.shape
    k = -1
    min_int = 0.
    max_int = 0.
    for i in range(nms):
        for j in range(nmz):
            mzk = ms1_mz[i][j]
            ink = ms1_intensity[i][j]
            if ink == 0:
                break

            if ink < thr or ink <= min_int:
                continue

            kend = k + 1
            err = mzk * tol
            ix, nix = _check_in_mz(sel_mz[:kend], mzk, err)
            if ink > max_int:
                ki = k
                max_int = ink
            else:
                ki = np.searchsorted(sel_int[:kend], ink)
                if sel_int[ki] == ink:
                    continue

            if nix > 0:
                # if is found in the m/z array, adjust the selected array
                if ix[-1] > ki:
                    # if the matched m/z in selected m/z array has higher
                    # intensity, ignore this.
                    continue

                # otherwise, this intensity is the highest among
                # all matched m/z, then adjust the arrays
                kj = 0
                for ij in range(ix[0], kend):
                    if ij in ix:
                        kj += 1
                    else:
                        sel_mz[ij - kj] = sel_mz[ij]
                        sel_int[ij - kj] = sel_int[ij]
                sel_mz[kend - kj: kend] = 0.
                sel_int[kend - kj: kend] = 0.
                k = kend - kj
                kend = k + 1

            if k == nsel - 1:
                sel_mz[:ki] = sel_mz[1: nsel]
                sel_int[:ki] = sel_int[1: nsel]
                min_int = sel_int[0]
            else:
                sel_mz[ki + 1: kend + 1] = sel_mz[ki:kend]
                sel_int[ki + 1: kend + 1] = sel_int[ki:kend]
                k += 1
            sel_mz[ki] = mzk
            sel_int[ki] = ink

    return sel_mz[:k + 1], sel_int[:k + 1]


@nb.njit("float64[:](float64[:,:], float64[:,:], float64, float64)",
         parallel=True)
def extract_xic(ms1_mz, ms1_intensity, sel_mz, tol):
    """
    Extracts XIC from matrix.

    Args:
        ms1_mz: m/z matrix of MS1, No. ms by No. m/z values.
        ms1_intensity: Intensity matrix of MS1.
        sel_mz: Selected m/z for extracting XIC
        tol: Tolerance for matching m/z, in ppm.

    Returns:
        A matrix of XIC.

    """
    ns = sel_mz
    nms, nmz = ms1_mz.shape
    xic = np.zeros((ns, nms), dtype=nb.float64)

    # extract XIC
    for i in range(nms):
        ix = np.searchsorted(ms1_mz[i], sel_mz)
        for j in range(ns):
            jk = ix[j]
            mzk = sel_mz[j]
            ink = 0.
            # left side
            for ji in range(jk - 1, -1, -1):
                if mzk - ms1_mz[i][ji] > tol:
                    break
                if ms1_intensity[i][ji] > ink:
                    ink = ms1_intensity[i][ji]
            # right side
            for ji in range(jk, nmz):
                if ms1_mz[i][ji] - mzk > tol:
                    break
                if ms1_intensity[i][ji] > ink:
                    ink = ms1_intensity[i][ji]
            xic[j][i] = ink
    return xic
