"""
This module calculates isotopic distributions including isotopic fine
structure of molecules using FFT and various scaling 'tricks'. Easily
adopted to molecules of any elemental composition (by altering
MAX_ELEMENTS and the nuclide matrix A). To simulate spectra, convolute
with peak shape using FFT.

(C) 1999 by Magnus Palmblad, Division of Ion Physics, Uppsala Univ.

Acknowledgements:
Lars Larsson-Cohn, Dept. of Mathematical Statistics, Uppsala Univ.,
for help on theory of convolutions and FFT.
Jan Axelsson, Div. of Ion Physics, Uppsala Univ. for comments and ideas

Contact Magnus Palmblad (magnus.palmblad@gmail.com)

Converted to Python 1/10/08 by
Brian H. Clowers (bhclowers@gmail.com)

"""
import numpy as np
from numpy import fft


def next2pow(v):
    return 2 ** int(np.ceil(np.log(x) / np.log(2.0)))


MAX_ELEMENTS = 5 + 1  # add 1 due to mass correction 'element'
MAX_ISOTOPES = 4    # maximum No. of isotopes for one element
CUTOFF = 1e-8       # relative intensity cutoff for plotting

WINDOW_SIZE = 500

# RESOLUTION=input('Resolution (in Da) ----> ');  % mass unit used in vectors
RESOLUTION = 0.5
if RESOLUTION < 0.00001:  # minimal mass step allowed
  RESOLUTION = 0.00001
elif RESOLUTION > 0.5:  # maximal mass step allowed
  RESOLUTION = 0.5

# R is used to scale nuclide masses (see below)
R = 0.00001/RESOLUTION

WINDOW_SIZE = WINDOW_SIZE / RESOLUTION  # convert window size to new mass units
# fast radix-2 fast-Fourier transform algorithm
WINDOW_SIZE = next2pow(WINDOW_SIZE)

if WINDOW_SIZE < np.round(496708 * R) + 1:
    # just to make sure window is big enough
    WINDOW_SIZE = next2pow(np.round(496708 * R) + 1)

# H378 C254 N65 O75 S6
# empiric formula, e.g. bovine insulin
m = np.array([378, 254, 65, 75, 6, 0])

# isotopic abundances stored in matrix A (one row for each element)
A = np.zeros((MAX_ELEMENTS, MAX_ISOTOPES, 2))

A[0][0, :] = [100783, 0.9998443]  # 1H
A[0][1, :] = [201410, 0.0001557]  # 2H
A[1][0, :] = [100000, 0.98889]  # 12C
A[1][1, :] = [200336, 0.01111]  # 13C
A[2][0, :] = [100307, 0.99634]  # 14N
A[2][1, :] = [200011, 0.00366]  # 15N
A[3][0, :] = [99492, 0.997628]  # 16O
A[3][1, :] = [199913, 0.000372]  # 17O
A[3][2, :] = [299916, 0.002000]  # 18O
A[4][0, :] = [97207, 0.95018]  # 32S
A[4][1, :] = [197146, 0.00750]  # 33S
A[4][2, :] = [296787, 0.04215]  # 34S
A[4][2, :] = [496708, 0.00017]  # 36S
# for shifting mass so that Mmi is near left limit of window
A[5][0, :] = [100000, 1.00000]

mmi = np.fromiter([np.round(100783 * R),
                   np.round(100000 * R),
                   np.round(100307 * R),
                   np.round(99492 * R),
                   np.round(97207 * R),
                   0], np.float64) * m
# (Virtual) monoisotopic mass in new units
mmi = mmi.sum()

# shift distribution to 1 Da from lower window limit:
m[MAX_ELEMENTS - 1] = np.ceil(((WINDOW_SIZE - 1)
                               - np.mod(mmi, WINDOW_SIZE - 1)
                               + np.round(100000 * R)) * RESOLUTION)
# correction for 'virtual' elements and mass shift
MASS_REMOVED = np.array([0, 11, 13, 15, 31, -1]) * m
MASS_REMOVED = MASS_REMOVED.sum()

ptA = np.ones(WINDOW_SIZE)
t_fft = 0
t_mult = 0

for i in range(MAX_ELEMENTS):
    tA = np.zeros(WINDOW_SIZE)
    for j in range(MAX_ISOTOPES):
        if A[i][j, 0] != 0:
            # removed +1 after R+1 --we're using python
            # put isotopic distribution in tA
            tA[np.round(A[i][j, 0] * R)] = A[i][j, 1]

    # FFT along elements isotopic distribution O(nlogn)
    tA = fft.fft(tA)
    tA = tA ** m[i]  # O(n)
    #################
    ptA = ptA * tA  # O(n) this is where it is messing UP
    #################

ptA = fft.ifft(ptA).real  # O(nlogn)
