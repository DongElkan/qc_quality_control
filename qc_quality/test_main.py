"""
This module debugs and tests the QC quality control package.
"""
import os
import sys
import logging
import tqdm
import pickle

import numpy as np

from .readers import mzml_reader

from .extract_xic import (centroid,
                          extract_top_mz,
                          extract_xic,
                          clear_mz_intensity_matrix_zeros)
from .extract_features import extract_features
from .readers.base import XIC


logging.basicConfig(level="INFO",
                    format="%(asctime)s [%(levelname)s]  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
THR = 0.001
TOL = 10.  # ppm


def main():
    """
    Main function.

    """
    mzml_file = r"E:\data\qc_quality_control\DBT Project\20200824-NEG-S1.mzML"
    logging.info(f"Reading mzML data from: {mzml_file}...")
    mzml = mzml_reader.mzMLReader(mzml_file)

    logging.info("Constructing MS1 data matrix...")
    nms1 = len(mzml._ms1_index)
    nmz_max = max(mzml._spectra[i].peak_num for i in mzml._ms1_index)
    # m/z matrix
    ms1_mz = np.zeros((nms1, nmz_max), dtype=np.float64)
    ms1_intensity = np.zeros((nms1, nmz_max), dtype=np.float64)
    rts = np.zeros(nms1, dtype=np.float64)
    max_mz_int = 0.
    nrecords = np.zeros(nms1, dtype=np.int64)
    for j, i in tqdm.tqdm(enumerate(mzml._ms1_index),
                          desc="Create m/z-intensity matrix",
                          total=nms1):
        spec = mzml._spectra[i]

        # centroid the m/z
        mz, intens = centroid(spec.mz, spec.intensity, 10.)
        nk = mz.size
        ms1_mz[j][:nk] = mz
        ms1_intensity[j][:nk] = intens
        nrecords[j] = nk
        rts[j] = spec.rt
        # base peak intensity
        if spec.max_intensity > max_mz_int:
            max_mz_int = spec.max_intensity

    logging.info(f"Created m/z and intensity matrix with size {ms1_mz.shape}.")

    int_thr = max_mz_int * THR
    logging.info(f"Set intensity threshold at {'%.6f' % int_thr} for highest "
                 f"peak intensity {'%.4f' % max_mz_int} at threshold {THR}.")

    logging.info("Extracting XIC...")
    xic_collection = []
    n_iter = 0
    nxic = 0
    while 1:
        n_iter += 1
        logging.info(f"Iterating {n_iter}...")
        logging.info("Extracting m/z with top intensities...")
        sel_mz, sel_int = extract_top_mz(ms1_mz, ms1_intensity, TOL, int_thr)

        if sel_mz.size == 0:
            logging.info("Intensity threshold reached, stop iteration.")
            break

        # logging.info(f"Top intensity selected: {'%.4f' % sel_int[-1]}")
        # assert (np.diff(sel_int) > 0).all(), "Something wrong here..."

        sel_int = sel_int[::-1]
        sel_mz = sel_mz[::-1]

        logging.info("Extracting XIC...")
        logging.info(f"Selected top intensity {'%.2f' % sel_int[0]} at m/z "
                     f"{'%.4f' % sel_mz[0]}.")
        kk = np.unravel_index(ms1_intensity.argmax(), ms1_intensity.shape)
        logging.info(f"In data matrix, top intensity "
                     f"{'%.2f' % ms1_intensity[kk]}, "
                     f"at m/z {'%.4f' % ms1_mz[kk]}.")
        xic, ms1_mz, ms1_intensity = extract_xic(
            ms1_mz, ms1_intensity, nrecords, sel_mz, TOL)
        logging.info(f"Extracted XIC with size: {xic.shape}.")
        logging.info(f"Max intensity in XICs: {'%.4f' % xic.max()}.")

        # reorganize the m/z and intensity
        logging.info("Reorganizing m/z and intensity...")
        ms1_mz, ms1_intensity, nrecords = clear_mz_intensity_matrix_zeros(
            ms1_mz, ms1_intensity)
        logging.info(f"Updated m/z-intensity matrix to size {ms1_mz.shape}.")
        logging.info(f"Max No. of records {nrecords.max()}.")
        mm = ms1_intensity.max()
        logging.info(f"Updated max intensity {'%.2f' % mm}.")

        xic_collection.append((xic, sel_mz))
        nxic += xic.shape[0]

    # combine extracted XIC
    xic_mtr = []
    for xic, xmz in xic_collection:
        for x, mz in zip(xic, xmz):
            xic_mtr.append(XIC(rt=rts, intensity=x, mz=mz, tol=TOL))
    logging.info(f"Total {nxic} XICs were extracted.")

    logging.info("Caching extracted XICs...")
    with open("tmp.pkl", "wb") as f:
        pickle.dump(xic_mtr, f)

    logging.info("Extracting features...")
    features = extract_features(xic_mtr, int_thr)
    logging.info(f"Total {len(features)} features were extracted.")

    logging.info("Saving to file...")
    data_name, _ = os.path.splitext(os.path.basename(mzml_file))
    with open(f"{data_name}_features.csv", "w") as f:
        f.write("m/z,retention time,peak area\n")
        for mz, a, t in features:
            f.write(f"{'%.8f' % mz},{'%.4f' % t},{'%.2f' % a}\n")
    logging.info(f"Saved to {data_name}_features.csv.")
    logging.info("End analysis.")


if __name__ == "__main__":
    main()
