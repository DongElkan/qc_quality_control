"""
This module extracts peptide XIC from .mgf or mzML files.
"""
import numpy as np
import numba as nb
import tqdm

from typing import List, Tuple

from csaps import csaps
from scipy.optimize import OptimizeWarning

from .readers.base import XIC
from . import peak_analysis


@nb.njit("int64[:,:](float64[:])")
def peak_detection(peaks) -> np.ndarray:
    """
    Detects peaks using first derivatives.

    Args:
        peaks: peaks.

    Returns:
        array: Index of detected peak edges.

    """
    intensity_diff = peaks[1:] - peaks[:-1]
    n = intensity_diff.size

    # peak edges
    peak_edges = np.zeros((n, 2), dtype=nb.int64)
    k = 0

    # starting point
    i0 = 0
    for i in range(n):
        if intensity_diff[i] > 0:
            i0 = i
            break

    if i0 == n - 1:
        return peak_edges[:k]

    # gets peak edges
    i1 = i0
    while i0 < n:
        v0 = intensity_diff[i0]
        i1 = i0 + 1
        for i in range(i0 + 1, n):
            if intensity_diff[i] >= 0 > v0:
                i1 = i
                break
            v0 = intensity_diff[i]

        if i1 >= n - 1:
            break

        if i1 - i0 >= 5:
            peak_edges[k][0] = i0
            peak_edges[k][1] = i1 + 1
            k += 1
        i0 = i1

    if i1 - i0 >= 5:
        peak_edges[k][0] = i0
        peak_edges[k][1] = i1 + 1
        k += 1

    return peak_edges[:k]


@nb.njit("float64(float64[:], float64[:])", fastmath=True)
def calculate_area(x, y):
    """
    Calculates areas between curve using Trapezoidal rule.

    Args:
        x: x axis.
        y: y axis.

    Returns:
        float: area.

    """
    n = x.size
    a = 0.
    for i in range(n-1):
        d = x[i + 1] - x[i]
        u = (y[i + 1] + y[i]) / 2
        a += d * u

    # correct baseline
    c = abs(y[0] + y[n - 1]) * (x[n - 1] - x[0]) / 2

    return a - c


def _order_time_xic(xic):
    """ Sorts times of XIC.
    Makes sure the xdata satisfy the condition: x1 < x2 < ... < xN
    for spline smoothing.
    """
    xic_times = xic[:, 0]
    if (np.diff(xic_times) <= 0).any():
        six = np.argsort(xic_times)
        xic = xic[six, :]
        xic_times = xic_times[six]
        # zero differenes between adjacent time point
        zero_diff_idx, = np.where(np.diff(xic_times) == 0)

        if zero_diff_idx.size == 0:
            return xic

        n = xic_times.size
        # interpolate the time if it is same with previous time point.
        tmp_times = xic_times.copy()
        for i in zero_diff_idx:
            if i < n - 2:
                tmp_times[i + 1] = (tmp_times[i] + tmp_times[i + 2]) / 2
            else:
                tmp_times[i + 1] += 0.001
        xic[:, 0] = tmp_times

    return xic


def extract_features(xics: List[XIC], thr: float)\
        -> List[Tuple[float, float, float]]:
    """
    Extracts features from XICs.

    Args:
        xics: List of XIC.
        thr: Intensity threshold.

    Returns:
        list: List of features.

    """
    # extracts peaks for identified peptides
    features = []
    for xic in tqdm.tqdm(xics, desc="Extracting features"):
        # smooth of intensity
        sm_intensity = csaps(xic.rt, xic.intensity, xic.rt, smooth=0.99999)
        # detects peaks using first derivatives
        peaks = peak_detection(sm_intensity.astype(np.float64))

        # maps identified peptide retention times to XIC to extract
        for edge in peaks:
            il = edge[0]
            ir = edge[1]
            tt_int = xic.intensity[il: ir]
            # intensity of peak must be higher than 0.1% max peak intensity
            if tt_int.max() >= thr:
                tt_rt = xic.rt[il: ir]
                # fit the peak to get peak area, and check is valid or not
                try:
                    int_fit, param = peak_analysis.fit_curve(tt_rt, tt_int)
                except (RuntimeError, TypeError,
                        RuntimeWarning, OptimizeWarning):
                    continue

                # some are not converged
                if (np.diff(int_fit) < 0).any():
                    # peak areas
                    rt_fit = np.arange(tt_rt[0], tt_rt[-1], 0.0001)
                    int_fit = peak_analysis._emg(rt_fit, *param)
                    # TODO: It may be questionable to calculate peak area
                    #       using fitted peak, instead of original data points.
                    a = calculate_area(rt_fit, int_fit)
                    t = rt_fit[int_fit.argmax()]
                    features.append((xic.mz, a, t))

    return features
