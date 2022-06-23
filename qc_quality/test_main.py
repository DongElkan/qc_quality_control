"""
This module debugs and tests the QC quality control package.
"""
import sys
import logging
import tqdm

import numpy as np

from qc.readers import mzml_reader

from qc.extract_xic import (centroid,
                            extract_top_mz,
                            extract_xic,
                            clear_mz_intensity_matrix_zeros)


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
    for j, i in tqdm.tqdm(enumerate(mzml._ms1_index),
                          desc="Create m/z-intensity matrix",
                          total=nms1):
        spec = mzml._spectra[i]

        # centroid the m/z
        mz, intens = centroid(spec.mz, spec.intensity, 10.)
        nk = mz.size
        ms1_mz[j][:nk] = mz
        ms1_intensity[j][:nk] = intens
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
    while 1:
        n_iter += 1
        logging.info(f"Iterating {n_iter}...")
        logging.info("Extracting m/z with top intensities...")
        sel_mz, sel_int = extract_top_mz(ms1_mz, ms1_intensity, TOL, int_thr)
        msel = sel_int[-1]
        logging.info(f"Top intensity selected: {'%.4f' % msel}")
        # print(sel_int)
        assert (np.diff(sel_int) > 0).all(), "Something wrong here..."

        if sel_mz.size == 0:
            logging.info("Intensity threshold reached, stop iteration.")
            break

        logging.info("Extracting XIC...")
        sel_mz = sel_mz[::-1]  # from highest to lowest intensity
        xic, ms1_mz, ms1_intensity = extract_xic(
            ms1_mz, ms1_intensity, sel_mz, TOL)
        logging.info(f"Extracted XIC with size: {xic.shape}.")
        logging.info(f"Max intensity in XICs: {'%.4f' % xic.max()}.")

        # reorganize the m/z and intensity
        logging.info("Reorganizing m/z and intensity...")
        ms1_mz, ms1_intensity = clear_mz_intensity_matrix_zeros(
            ms1_mz, ms1_intensity)
        logging.info(f"Updated m/z-intensity matrix to size {ms1_mz.shape}.")
        mm = ms1_intensity.max()
        logging.info(f"Updated max intensity {'%.4f' % mm}.")

        xic_collection.append(xic)


if __name__ == "__main__":
    main()
