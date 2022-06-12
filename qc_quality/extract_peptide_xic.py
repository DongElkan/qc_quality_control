"""
This module extracts peptide XIC from .mgf or mzML files.
"""
import sys
import collections
import numpy as np
import tqdm

from operator import itemgetter

from csaps import csaps
from scipy.optimize import OptimizeWarning

import peak_analysis


def peak_detection(peaks):
    """ Peak detection using first derivatives.
    """
    diff_peaks = np.diff(peaks)
    n = diff_peaks.size

    # starting point
    i0 = 0
    for i, v in enumerate(diff_peaks):
        if v > 0:
            i0 = i
            break

    if i0 == n - 1:
        return None

    # gets peaks
    peaks = []
    while i0 < n:
        v0 = diff_peaks[i0]
        for i, v in enumerate(diff_peaks[i0+1:]):
            if v > 0 > v0:
                break
            v0 = v

        i1 = i + i0 + 1
        if i1 >= n - 1:
            break
        peaks.append((i0, i1 + 1))
        i0 = i1

    if i1 > i0:
        peaks.append((i0, i1 + 1))

    # quality control
    return [(i, j) for i, j in peaks if j - i >= 5]


def area(x, y):
    """ Calculates areas between curve.
    """
    dx = np.diff(x)
    sy = (y[1:] + y[:-1]) / 2
    t = (dx * sy).sum()
    # correction if baseline appears
    c = (y[0] + y[-1]) * (x[-1] - x[0]) / 2
    return t - c


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


# def extract_peptide_rt(xics, psms):
#     """ Extracts retention times for peptides.
#     """
#     # groups PSMs based on peptides and charge states
#     peps_spec_rts, pep_seqs = collections.defaultdict(set), dict()
#     for p in tqdm.tqdm(psms, desc="Group identified peptides"):
#         pk = merge_seq_mods(p.seq, p.mods)
#         raw = p.spec_id.split(".")[0]
#         peps_spec_rts[(raw, pk, p.charge)].add(p.retention_time)
#         pep_seqs[pk] = p.seq
#
#     # extracts peaks for identified peptides
#     pep_retentions = []
#     for (raw, p, c), xic in tqdm.tqdm(xics.items(), desc="Detect peptide RTs"):
#         if (raw, p, c) not in peps_spec_rts:
#             continue
#
#         pep_times = np.array(sorted(peps_spec_rts[(raw, p, c)]))
#
#         max_int = xic[:, 1].max()
#         # smooth of intensity
#         sm_intensity = csaps(xic[:, 0], xic[:, 1], xic[:, 0], smooth=0.99)
#         # detects peaks using first derivatives
#         peaks = _peak_detection(sm_intensity)
#
#         # maps identified peptide retention times to XIC to extract peaks
#         peak_pep_nums = collections.defaultdict()
#         for i, j in peaks:
#             tt_int = xic[i:j, 1]
#             # intensity of peak must be higher than 0.1% max peak intensity
#             if tt_int.max() >= max_int * 0.001:
#                 t0, t1 = xic[i, 0], xic[j, 0]
#                 peak_pep_nums[(i, j)] = ((pep_times >= t0)
#                                          & (pep_times <= t1)).sum()
#
#         # max number of peptides identified
#         nrt = max(peak_pep_nums.values())
#         # justifies the peaks
#         xic_times = xic[:, 0]
#         peak_areas = []
#         for (i, j), nk in peak_pep_nums.items():
#             if nk < nrt:
#                 continue
#
#             t0, t1 = xic[i, 0], xic[j, 0]
#             idx = (xic_times >= t0) & (xic_times <= t1)
#             rt, intensity = xic[idx, 0], xic[idx, 1]
#
#             # fits the peak to get peak properties: peak area, is valid or not
#             try:
#                 int_fit, param = peak_analysis.fit_curve(rt, intensity)
#             except (RuntimeError, TypeError, RuntimeWarning, OptimizeWarning):
#                 continue
#
#             # some are not converged
#             if (np.diff(int_fit) < 0).any():
#                 # peak areas
#                 rt_fit = np.arange(t0, t1, 0.0001)
#                 int_fit = peak_analysis._emg(rt_fit, *param)
#                 a = _area(rt_fit, int_fit)
#                 rt_pep = rt_fit[int_fit.argmax()]
#                 peak_areas.append((a, rt_pep))
#
#         # max peak area
#         if not peak_areas:
#             continue
#
#         area, pep_rt = max(peak_areas, key=itemgetter(0))
#         pep_retentions.append((raw, p, pep_seqs[p], c, area, pep_rt))
#
#     return pep_retentions
