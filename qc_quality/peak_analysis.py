"""
Analysis of chromatographic peaks
"""
import numpy as np
import numba as nb
import math

from typing import Tuple
from scipy import optimize


_ = np.seterr(all='warn', over='raise', invalid='ignore')


@nb.njit("float64[:](float64[:], float64, float64, float64, float64)",
         fastmath=True)
def _emg(x, h, sigma, mu, tau):
    """
    Fit exponentially modified gaussian function to simulate the
    shape of chromatographic peaks.

    Args:
        x: The peak for fitting
        h: The amplitude of Gaussian
        sigma: Gaussian variance
        mu: Gaussian mean
        tau: Exponent relaxation time

    Returns:
        np.ndarray: Fitted peak.

    References:
        [1] wiki/Exponentially_modified_Gaussian_distribution
        [2] Kalambet Y, et al. J Chemometrics. 2011; 25, 352–356.
        [3] Jeansonne MS, et al. J Chromatogr Sci. 1991, 29, 258-266.

    """
    n = x.size
    y = np.zeros(n, dtype=nb.float64)
    c = sigma / tau
    sq2 = np.sqrt(2)
    lk = np.log(h * c * np.sqrt(np.pi / 2))
    for i in range(n):
        m = (x[i] - mu) / sigma
        z = (c - m) / sq2
        if z > 6.71e7:
            # to avoid overflow
            s = np.log(h / (1 - m / c)) - m * m / 2
        else:
            if z < 0:
                g = c * c / 2 - (m * c)
            else:
                g = -m * m / 2
            e = lk + np.log(math.erfc(z))
            s = g + e

        if s >= -50:
            y[i] = np.exp(min(s, 50))

    return y


@nb.njit("float64[:](float64[:], float64, float64, float64)", fastmath=True)
def _gaussian(x, h, mu, sigma):
    """ Fit gaussian curve. """
    n = x.size
    y = np.zeros(n, dtype=x.dtype)
    for i in range(n):
        s = -((x[i] - mu) / sigma) ** 2 / 2
        if s > -50:
            y[i] = h * np.exp(min(s, 50))
    return y


def fit_curve(rt: np.ndarray, intensity: np.ndarray, shape: str = 'emg')\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the curve using scipy's curve_fit optimization.

    Args:
        rt: Retention time.
        intensity: Peak intensities.
        shape: Function for fitting the peak. {`emg`, `gaussian`}.
            emg: exponentially modified gaussian function
            gaussian: gaussian function
            Defaults to `emg`.

    Returns:
        np.ndarray: Fitted peaks.
        np.ndarray: Parameters for fitting the peak.

    """
    # initialization
    j = np.argmax(intensity)
    h0, m0 = intensity[j], rt[j]
    s0 = np.std(rt)
    t0 = 1

    # fit curve using optimization
    if shape == 'emg':
        param, _ = optimize.curve_fit(_emg, rt, intensity, p0=(h0, s0, m0, t0))
        return _emg(rt, *param), param

    if shape == 'gaussian':
        param, _ = optimize.curve_fit(_gaussian, rt, intensity, p0=(h0, m0, s0))
        return _gaussian(rt, *param), param

    raise ValueError("Unrecognized shape for fitting. Expected `emg` or "
                     f"`gaussian`, got {shape}.")


def get_peak_param(rt: np.ndarray, intensity: np.ndarray)\
        -> Tuple[float, float, float]:
    """
    Get peak parameters, including peak width at half height,
    peak width at base and peak height.

    Args:
        rt: Retention time.
        intensity: Peak intensities.

    Returns:
        tuple: A tuple of peak height, peak width at half height,
            peak width at base.

    """
    h = intensity.max()
    # half peak height
    h2 = h/2
    # approximated peak width at half height
    rt_h2 = rt[intensity >= h2]
    wh = rt_h2.max() - rt_h2.min()
    # approximated peak width at base
    rt_base = rt[intensity >= h * 0.001]
    wb = rt_base.max() - rt_base.min()
    return h, wh, wb


def whittaker_smoothing(x, r, d=2):
    """ Whittacker smoother """
    n = x.shape[0]
    e = np.eye(n)
    d = np.diff(e, n=d, axis=0)
    b = r * np.dot(d.T, d)

    b[np.diag_indices_from(b)] += 1.
    z = np.linalg.solve(b, x)
    return z
