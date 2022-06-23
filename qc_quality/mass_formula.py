"""
This module provides functions to infer molecular formula from accurate
mass in mass spectrometry based on 7 golden rules.

References:
    Kind T, et al. Seven Golden Rules for heuristic filtering of
    molecular formulas obtained by accurate mass spectrometry.
    BMC Bioinformatics. 2007, 8.

"""
import numba as nb
import numpy as np


# Maximum number of element: CHONSP
MAX_NUM = np.array([
    [39, 72, 20, 20, 8, 9],
    [78, 126, 27, 25, 14, 9],
    [156, 236, 63, 32, 14, 9],
    [162, 208, 78, 48, 9, 6]
], dtype=np.int64)


@nb.njit("int64[:](float64, float64)", fastmath=True)
def mass2formula(mass, tol):
    """
    Infers formula from the accurate mass.

    Args:
        mass: Accurate mass.
        tol: Maximum error between mass of the formula and mass,
            in ppm.

    Returns:
        array: Number of atoms in the order of CHON.

    """
    mc = 12.0000000
    mh = 1.00782503223
    mo = 15.99491461957
    mn = 14.00307400443
    if mass < 500:
        max_num = MAX_NUM[0]
    elif mass < 1000:
        max_num = MAX_NUM[1]
    elif mass < 2000:
        max_num = MAX_NUM[2]
    else:
        max_num = MAX_NUM[3]

    err = mass * tol / 1e6

    max_nc = min(int(mass / mc), max_num[0]) + 1
    max_nn = min(int(mass / mn), max_num[0]) + 1
    max_no = min(int(mass / mo), max_num[0]) + 1
    fm = np.zeros(4, dtype=nb.int64)
    for i in range(1, max_nc):
        m = mc * i
        j = 0
        for j in range(1, max_no):
            m += mo
            if j / i > 3 or m > mass + err:
                # according to the rule, O/C <= 3, and m less than mass
                break
            k = 0
            for k in range(1, max_nn):
                m += mn
                if k / i > 4 or m > mass + err:
                    # N/C <= 4, and m less than mass
                    break
                nh = round((mass - m) / mh)
                if 0.1 <= nh / i <= 6:
                    # H/C <= 6
                    mk = m + nh * mh
                    if abs(mk - mass) <= err:
                        fm[0] = i
                        fm[1] = nh
                        fm[2] = j
                        fm[3] = k
                        return fm
            m -= mn * k  # back to oxygen
        m -= mo * j  # back to carbon

    return fm


# if __name__ == "__main__":
#     accurate_mass = 344.329051
#     fm = mass2formula(accurate_mass, 5.)
#     print(fm)
