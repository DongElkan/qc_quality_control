"""
Baseline correction for chromatograms.
"""
import numpy as np
from numpy import linalg


def _adaptive_reweight(w, d, i, k, p):
    w[d >= 0] = 0
    # re-weight current baseline
    neg_d = d[d < 0]
    neg_d /= neg_d.sum()
    w[d < 0] = np.exp(i * neg_d)
    # compensation for start and end
    w[:k] = p
    w[-k:] = p
    return w


def _asymmetric_reweight(d):
    neg_d = d[d < 0]
    m = neg_d.mean()
    s = neg_d.std()
    f = np.clip(2 * (d - (2 * s - m)) / s, None, 1e2)
    return 1 / (1 + np.exp(f))


def rpls(x, method="adaptive", lambda_=100, derivative=2,
         max_iter=100, tol=1e-3, wep=0.1, p=0.5):
    """
    Adaptive iteratively re-weighted penalized least squares
    for baseline correction.

    Parameters
    ----------
    x: np.ndarray
        Chromatographic peak
    method: str
        Method for baseline correction. Two re-weighted penalized least
        squares methods are are currently implemented:
        "adaptive": adaptive iteratively
        "asymmetric": asymmetrically
        Default is "adaptive".
    lambda_: int
        Penalty for smoothing.
    derivative: int
        Derivative order, default is 2.
    max_iter: int
        Maximum iteration.
    tol: float
        Tolerance to stop iteration.
    wep: float
        The first and last wep * n data points to compensate,
        where n is the number of points in x.
    p: float
        Weights for starting and ending points defined by wep

    Returns
    -------
    z: np.ndarray
        Fitted baseline.

    References
    ----------
    [1] Zhang ZM, Chen S, Liang YZ. Baseline correction using adaptive
        iteratively reweighted penalized least squares. Analyst. 2010,
        135, 1138-1146.
    [2] Baek SJ, Park A, Ahn YJ, Choo J. Baseline correction using
        asymmetrically re-weighted penalized least squares smoothing.
        Analyst. 2015, 140, 250-257

    """
    n = x.shape[0]
    w = np.ones(n)

    k = max(1, int(n * wep))

    # smoothing matrix
    e = np.eye(n)
    d = np.diff(e, n=derivative, axis=0)
    b = lambda_ * np.dot(d.T, d)

    d0, z, s0 = x, np.ones(n) * np.amin(x), linalg.norm(x)
    for i in range(max_iter):
        # solve background using least squares
        c = b.copy()
        c[np.diag_indices_from(c)] += w
        z = linalg.solve(c, w * x)

        # remove peak signal from the background for next iteration
        d = x - z
        s1 = linalg.norm(d - d0)
        if s1 / s0 <= tol:
            if i == max_iter - 1:
                print("WARING: max iteration reached!")
            break

        if method == "adaptive":
            w = _adaptive_reweight(w, d, i + 1, k, p)
        elif method == "asymmetric":
            w = _asymmetric_reweight(d)
        else:
            raise ValueError("Unrecognized method. It should be `adaptive` or"
                             " `asymmetric`.")
        # update for tolerance
        s0, d0 = linalg.norm(d), d

    return z
