"""
This module extracts peptide XIC from .mgf or mzML files.
"""

import os
import sys
import re
import collections
import numpy as np
import tqdm

from operator import attrgetter, itemgetter

from csaps import csaps
from scipy.optimize import OptimizeWarning

import peak_analysis

from readers.mzml_reader import mzMLReader


def _peak_detection(peaks):
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


def _area(x, y):
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


def extract_peptide_rt(xics, psms):
    """ Extracts retention times for peptides.
    """
    # groups PSMs based on peptides and charge states
    peps_spec_rts, pep_seqs = collections.defaultdict(set), dict()
    for p in tqdm.tqdm(psms, desc="Group identified peptides"):
        pk = merge_seq_mods(p.seq, p.mods)
        raw = p.spec_id.split(".")[0]
        peps_spec_rts[(raw, pk, p.charge)].add(p.retention_time)
        pep_seqs[pk] = p.seq

    # extracts peaks for identified peptides
    pep_retentions = []
    for (raw, p, c), xic in tqdm.tqdm(xics.items(), desc="Detect peptide RTs"):
        if (raw, p, c) not in peps_spec_rts:
            continue

        pep_times = np.array(sorted(peps_spec_rts[(raw, p, c)]))

        max_int = xic[:, 1].max()
        # smooth of intensity
        sm_intensity = csaps(xic[:, 0], xic[:, 1], xic[:, 0], smooth=0.99)
        # detects peaks using first derivatives
        peaks = _peak_detection(sm_intensity)

        # maps identified peptide retention times to XIC to extract peaks
        peak_pep_nums = collections.defaultdict()
        for i, j in peaks:
            tt_int = xic[i:j, 1]
            # intensity of peak must be higher than 0.1% max peak intensity
            if tt_int.max() >= max_int * 0.001:
                t0, t1 = xic[i, 0], xic[j, 0]
                peak_pep_nums[(i, j)] = ((pep_times >= t0)
                                         & (pep_times <= t1)).sum()

        # max number of peptides identified
        nrt = max(peak_pep_nums.values())
        # justifies the peaks
        xic_times = xic[:, 0]
        peak_areas = []
        for (i, j), nk in peak_pep_nums.items():
            if nk < nrt:
                continue

            t0, t1 = xic[i, 0], xic[j, 0]
            idx = (xic_times >= t0) & (xic_times <= t1)
            rt, intensity = xic[idx, 0], xic[idx, 1]

            # fits the peak to get peak properties: peak area, is valid or not
            try:
                int_fit, param = peak_analysis.fit_curve(rt, intensity)
            except (RuntimeError, TypeError, RuntimeWarning, OptimizeWarning):
                continue

            # some are not converged
            if (np.diff(int_fit) < 0).any():
                # peak areas
                rt_fit = np.arange(t0, t1, 0.0001)
                int_fit = peak_analysis._emg(rt_fit, *param)
                a = _area(rt_fit, int_fit)
                rt_pep = rt_fit[int_fit.argmax()]
                peak_areas.append((a, rt_pep))

        # max peak area
        if not peak_areas:
            continue

        area, pep_rt = max(peak_areas, key=itemgetter(0))
        pep_retentions.append((raw, p, pep_seqs[p], c, area, pep_rt))

    return pep_retentions


def extract_xic(target_dir, conf_thr):
    """ Extracts XICs from target dir.
    """
    tol = 0.00002
    # peptide summary file
    pep_summary = [f for f in os.listdir(target_dir) if "PeptideSummary" in f]
    # get ProteinPilot search results
    print("Read search results...")
    res = read_peptide_summary(f"{target_dir}/{pep_summary[0]}")
    # get mass spectra
    spec_file = [f for f in os.listdir(target_dir) if f.endswith(".mgf")][0]
    spectra = _get_mgf(f"{target_dir}/{spec_file}")

    # get peptide precursor m/z
    print("Get identified peptides' precursor m/z...")
    # positive PSMs
    pos_res = [p for p in res if p.confidence >= conf_thr]
    pep_prec_mzs = collections.defaultdict(dict)
    for p in pos_res:
        j = p.spec_id.split(".")[0]
        pk = merge_seq_mods(p.seq, p.mods)
        if ((pk, p.charge) not in pep_prec_mzs[j]
                and not any(m.mass is None for m in p.mods)):
            pep_prec_mzs[j][(pk, p.charge)] = p.peptide.mz
    print(f"Total {sum(len(px) for px in pep_prec_mzs.values())} peptides")

    # prefix
    raw_prefix = {}
    for spec in spectra:
        spid, spraw = re.findall(r'Locus:(.*) File:"(.*)"', spec.id)[0]
        raw_prefix[spid.split(".")[0]] = spraw.split(".wiff")[0]

    # deal with different data files
    print("Map peptides to spectra to extract XIC...")
    pep_xics = collections.defaultdict()
    for j, raw in raw_prefix.items():
        if j not in pep_prec_mzs:
            continue
        # theoretical peptide precursor m/z
        curr_prec_mzs = pep_prec_mzs[j]

        # load mass spectra
        print(f"Read mass spectra from .mzML files for {raw}")
        raw_files = [f for f in os.listdir(target_dir)
                     if f.split("-")[0] == raw and f.endswith("mzML")]
        raw_specs = []
        for mzml_raw in raw_files:
            tmp_mzml = mzMLReader(f"{target_dir}/{mzml_raw}")
            raw_specs += [sp for sp in tmp_mzml._spectra if sp.ms_level == 1]
        raw_specs = sorted(raw_specs, key=attrgetter("rt"))
        del tmp_mzml

        # sort mass spectra
        for spec in raw_specs:
            spec.spectrum = spec.spectrum[spec.spectrum[:, 0].argsort(), :]

        # extract XIC
        for (pk, c), pmz in tqdm.tqdm(curr_prec_mzs.items(),
                                      total=len(curr_prec_mzs),
                                      desc=f"Extract XIC from {raw}"):
            # upper and lower bound of precursor m/z
            pmz_lw = pmz - (pmz * c * tol)
            pmz_hg = pmz + (pmz * c * tol)
            # extract XIC
            xic = []
            for spec in raw_specs:
                il = np.searchsorted(spec.spectrum[:, 0], pmz_lw, side="left")
                ih = np.searchsorted(spec.spectrum[:, 0], pmz_hg, side="right")
                xic.append([spec.rt, 0] if il == ih
                           else [spec.rt, spec.spectrum[il:ih, 1].max()])

            # check the order of retention times to make sure they are in
            # ascending order and not same between adjacent points.
            pep_xics[(j, pk, c)] = _order_time_xic(np.array(xic))

    return pep_xics, pos_res


def save_results(extracted_peptide_info, file_name):
    """ Saves the extracted peptide retentions.
    """
    # save to file
    with open(file_name, "w") as f:
        f.write("Raw Index\tPeptide\tSequence\tCharge\t"
                "Retention Time\tPeak Area\n")
        for k, p, s, c, a, t in tqdm.tqdm(extracted_peptide_info,
                                          desc="Save results"):
            f.write(f"{k}\t{p}\t{s}\t{c}\t{round(t, 2)}\t{round(a, 4)}\n")


def extract_peptide_retentions(target, fdr_threshold):
    """ Extracts peptide retention times.
    """
    # XICs
    xics, psms = extract_xic(target, fdr_threshold)
    # peptide retention times
    pep_retentions = extract_peptide_rt(xics, psms)
    # save the peptides
    save_results(pep_retentions, f"{target}_peptide_retentions.txt")


if __name__ == "__main__":
    args = sys.argv
    extract_peptide_retentions(args[1], float(args[2]))
