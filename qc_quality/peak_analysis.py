"""
Analysis of chromatographic peaks
"""
import numpy as np
from scipy import optimize, special


_ = np.seterr(all='warn', over='raise', invalid='ignore')


def _emg(x, h, sigma, mu, tau):
    """
    Fit exponentially modified gaussian function to simulate the
    shape of chromatographic peaks.

    Parameters
    ----------
    x: np.ndarray
        The peak for fitting
    h: float
        The amplitude of Gaussian
    sigma: float
        Gaussian variance
    mu: float
        Gaussian mean
    tau: float
        Exponent relaxation time

    Returns
    -------
    y: np.ndarray
        Fitted peak

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    [2] Kalambet Y, et al. J Chemometrics. 2011; 25, 352–356.
    [3] Jeansonne MS, et al. J Chromatogr Sci. 1991, 29, 258-266.
    """
    # set up parameters
    m = (x - mu) / sigma
    c = sigma / tau
    z = (c - m) / np.sqrt(2)
    y = np.empty(z.shape)
    ix = np.zeros(z.shape, dtype=bool)
    # using log transform to calculate the values
    if (z < 0).any():
        ix[z < 0] = True
        # gaussian
        g = c * c / 2 - (m[ix] * c)
        # exponential
        e = np.log(h * c * np.sqrt(np.pi / 2)) + np.log(special.erfc(z[ix]))
        y[ix] = g + e

    if (z > 6.71e7).any():
        ix[z > 6.71e7] = True
        m2 = m[z > 6.71e7]
        y[z > 6.71e7] = np.log(h / (1 - m2 / c)) - m2 * m2 / 2

    ix = np.logical_not(ix)
    if ix.any() > 0:
        m2 = m[ix]
        g = -m2 * m2 / 2
        e = np.log(h * c * np.sqrt(np.pi / 2)) + np.log(special.erfcx(z[ix]))
        y[ix] = g + e

    return np.exp(np.clip(y, -200, 200))


def _gaussian(x, h, mu, sigma):
    """ Fit gaussian curve. """
    y = ((x - mu) / sigma) ** 2 / 2
    # avoid underflow error
    return h * np.exp(-np.clip(y, -200, 200))


def fit_curve(rt, intensity, shape='emg'):
    """
    Fit the curve using scipy's curve_fit optimization.

    Parameters
    ----------
    rt: np.ndarray
        Retention time
    intensity: np.ndarray
        Peak intensities
    shape: str
        Function for fitting the peak. Accepted functions are:
        "emg": exponentially modified gaussian function
        "gau": gaussian function
        Default is "emg".

    Returns
    -------
    param: np.ndarray
        Parameters for fitting the peak

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

    raise ValueError("Unrecognized shape for fitting. The shape must be "
                     "`emg` or `gaussian`")


def get_peak_param(rt, intensity):
    """
    Get peak parameters, including peak width at half height,
    peak width at base and peak height.

    Parameters
    ----------
    rt: np.ndarray
        Retention time
    intensity: np.ndarray
        Peak intensities

    Returns
    -------
    param: tuple
        A tuple of peak height, peak width at half height,
        peak width at base

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
