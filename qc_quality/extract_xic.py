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

    cent_mz = cent_mz[:k]
    cent_intensity = cent_intensity[:k]

    ix = cent_mz.argsort()

    return cent_mz[ix], cent_intensity[ix]


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


@nb.njit("UniTuple(float64[:], 2)(float64[:,:], "
         "float64[:,:], float64, float64)")
def extract_top_mz(ms1_mz, ms1_intensity, tol, thr):
    """ Extracts XIC in top intensity. """
    nsel = 1000
    sel_mz = np.zeros(nsel, dtype=nb.float64)
    sel_int = np.zeros(nsel, dtype=nb.float64)
    nms, nmz = ms1_mz.shape
    k = -1
    max_int = 0.
    for i in range(nms):
        for j in range(nmz):
            mzk = ms1_mz[i][j]
            ink = ms1_intensity[i][j]
            if ink == 0:
                break

            if (ink <= sel_int[0] and k == nsel - 1) or ink < thr:
                continue

            err = mzk * tol / 1e6
            ix, nix = _check_in_mz(sel_mz[:k + 1], mzk, err)
            if nix > 0:
                # if is found in the m/z array, adjust the selected array
                if sel_int[ix[-1]] >= ink:
                    # if the matched m/z in selected m/z array has higher
                    # intensity, ignore this.
                    continue

                # otherwise, this intensity is the highest among
                # all matched m/z, then adjust the arrays
                a = 0
                for b in range(ix[0], k + 1):
                    if b in ix:
                        a += 1
                    else:
                        sel_mz[b - a] = sel_mz[b]
                        sel_int[b - a] = sel_int[b]
                sel_mz[k - a + 1: k + 1] = 0.
                sel_int[k - a + 1: k + 1] = 0.
                k -= a

            if ink > max_int:
                h = k + 1
                max_int = ink
            else:
                h = np.searchsorted(sel_int[:k + 1], ink)
                if sel_int[h] == ink:
                    continue

            # assign to the sorted arrays
            if k == nsel - 1:
                sel_int[:h - 1] = sel_int[1:h]
                sel_mz[:h - 1] = sel_mz[1:h]
                sel_int[h - 1] = ink
                sel_mz[h - 1] = mzk
            else:
                k += 1
                sel_mz[h + 1:k + 1] = sel_mz[h:k]
                sel_int[h + 1:k + 1] = sel_int[h:k]
                sel_mz[h] = mzk
                sel_int[h] = ink

    return sel_mz[:k + 1], sel_int[:k + 1]


@nb.njit("UniTuple(float64[:,:], 3)(float64[:,:], float64[:,:], int64[:],"
         " float64[:], float64)", parallel=True)
def extract_xic(ms1_mz, ms1_intensity, nrecords, sel_mz, tol):
    """
    Extracts XIC from matrix.

    Args:
        ms1_mz: m/z matrix of MS1, No. ms by No. m/z values.
        ms1_intensity: Intensity matrix of MS1.
        nrecords: Number of m/z in each mass spectrum.
        sel_mz: Selected m/z for extracting XIC, have been sorted based on
            their intensities from highest to lowest.
        tol: Tolerance for matching m/z, in ppm.

    Returns:
        A matrix of XIC.

    """
    ns = sel_mz.size
    nms, nmz = ms1_mz.shape
    xic = np.zeros((ns, nms), dtype=np.float64)

    # errors
    errs = np.zeros(ns)
    for i in range(ns):
        errs[i] = sel_mz[i] * tol / 1e6

    # extract XIC
    for i in range(nms):
        ix = np.searchsorted(ms1_mz[i][:nrecords[i]], sel_mz)
        for j in range(ns):
            a = ix[j]
            mzk = sel_mz[j]
            err = errs[j]
            ink = 0.
            # left side
            jl = -1
            for h in range(a - 1, -1, -1):
                if ms1_mz[i][h] > 0:
                    if mzk - ms1_mz[i][h] > err:
                        break
                    if ms1_intensity[i][h] > ink:
                        ink = ms1_intensity[i][h]
                    jl = h
            # right side
            jr = -1
            for h in range(a, nrecords[i]):
                if ms1_mz[i][h] > 0:
                    if ms1_mz[i][h] - mzk > err:
                        break
                    if ms1_intensity[i][h] > ink:
                        ink = ms1_intensity[i][h]
                    jr = h + 1
            xic[j][i] = ink

            # if matched, clear the intensity that have been matched
            if jl >= 0 or jr >= 0:
                if jl >= 0:
                    j0 = jl
                else:
                    j0 = a
                if jr >= 0:
                    j1 = jr
                else:
                    j1 = a

                ms1_mz[i][j0:j1] = 0.
                ms1_intensity[i][j0:j1] = 0.

    return xic, ms1_mz, ms1_intensity


@nb.njit("Tuple((float64[:,:], float64[:,:], int64[:]))(float64[:,:], "
         "float64[:,:])")
def clear_mz_intensity_matrix_zeros(mz, intensity):
    """
    Clears the zeros in the matrix after extraction of XIC.

    Args:
        mz: m/z matrix.
        intensity: Intensity matrix.

    Returns:
        array: Updated m/z matrix.
        array: Updated intensity matrix.
        array: No. of valid m/z values each mass spectrum.

    """
    nms, nmz = mz.shape
    mz2 = np.zeros((nms, nmz), dtype=np.float64)
    intensity2 = np.zeros((nms, nmz), dtype=np.float64)
    num_records = np.zeros(nms, dtype=np.int64)
    mk = 0
    for i in range(nms):
        k = 0
        for j in range(nmz):
            if mz[i][j] > 0:
                mz2[i][k] = mz[i][j]
                intensity2[i][k] = intensity[i][j]
                k += 1
        if k > mk:
            mk = k
        num_records[i] = k
    return mz2[:, :mk], intensity2[:, :mk], num_records
