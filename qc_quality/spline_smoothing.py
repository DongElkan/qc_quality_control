"""
This module provides natural cubic spline smoothing with automatic
parameter selection using improved AIC.

References:
    [1] Reinsch CH. Smoothing by spline functions. Numer Math. 1967, 10,
        177–183.
    [2] Eubank RL. Nonparametric Regression and Spline Smoothing. 2nd
        Ed. New York; Basel: Marcel Dekker. 1999.
    [3] Green PJ, Silverman BW. Nonparametric Regression and
        Generalized Linear Models: A roughness penalty approach.
        Chapman and Hall/CRC. 1993.
    [4] Hurvich CM, Simonoff JS, Tsai CL. Smoothing Parameter Selection
        in Nonparametric Regression Using an Improved Akaike
        Information Criterion. J R Statist Soc B. 1998, 60, 271-293.
    [5] Lee TCM. Smoothing parameter selection for smoothing splines: a
        simulation study. Comput Stat Data Anal. 2003, 42, 139–148.
    [6] Clarke B, Fokoue E, Zhang HH. Principles and Theory for Data
        Mining and Machine Learning. Springer New York, NY. 2009.
    [7] Hutchinson MF, de Hoog FR. Smoothing Noisy Data with Spline
        Functions. Numer Math. 1985, 47, 99-106.

"""
import numpy as np
import numba as nb

from typing import List, Optional, Tuple, Dict


@nb.njit("float64[:](float64[:,:], float64[:])")
def _q_y(q, y) -> np.ndarray:
    """ Calculates q by y. """
    n, _ = q.shape
    z = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v = 0
        for j in range(i, i+3):
            v += q[i][j] * y[j]
        z[i] = v
    return z


@nb.njit("float64[:](float64[:,:], float64[:,:], float64)")
def hat_matrix_diagonal(q, b, a) -> np.ndarray:
    """
    Calculates diagonal elements of hat matrix
        I - a * Q' * inv(R + a * Q * Q') * Q
    Where
        R + a * Q * Q'
    is the Reinsch coefficient matrix. The inverse can be calculated
    through LDL Cholesky decomposition, by
        B^-1 = D^-1 * L^-1 + (I - L') * B^-1

    Args:
        q: Matrix Q.
        b: Inverse of Reinsch coefficient matrix.
        a: Smoothing parameter.

    Returns:
        array: Diagonal elements of hat matrix

    """
    n = q.shape[1]
    s = np.zeros(n, dtype=np.float64)

    # s[0]
    s[0] = q[0][0] * q[0][0] * b[0][0]
    # s[i], n-1 > i > 0
    for i in range(1, n - 1):
        qi = q[:, i]
        v = 0
        i0 = max(i - 2, 0)
        i1 = min(i + 1, n - 2)
        for j in range(i0, i1):
            vi = 0
            for k in range(i0, i1):
                vi += qi[k] * b[k][j]
            v += qi[j] * vi
        s[i] = v
    # s[n-1]
    s[n - 1] = q[n - 3][n - 1] * q[n - 3][n - 1] * b[n - 3][n - 3]

    return 1 - a * s


@nb.njit("Tuple((float64[:,:], float64[:]))(float64[:,:])", fastmath=True)
def ldl_decompose(a):
    """
    Calculates LDL Cholesky decomposition.

    Args:
        a: Banded matrix

    Returns:
        array: Lower angular matrix.
        array: Diagonal array of matrix D.

    """
    n = a.shape[0]
    ml = np.zeros((n, n), dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    d[0] = a[0][0]
    t0 = a[1][0] / d[0]
    d[1] = a[1][1] - (t0 * t0 * d[0])
    ml[1][0] = t0
    for i in range(2, n):
        i0 = i - 2
        i1 = i - 1
        t0 = a[i][i0] / d[i0]
        t1 = (a[i][i1] - (ml[i1][i0] * t0 * d[i0])) / d[i1]
        d[i] = a[i][i] - (t1 * t1 * d[i1]) - (t0 * t0 * d[i0])
        ml[i][i0] = t0
        ml[i][i1] = t1

    for i in range(n):
        ml[i][i] = 1

    return ml, d


@nb.njit("float64[:](float64[:,:], float64[:], float64[:])")
def solve_linear(ml, d, z) -> np.ndarray:
    """
    Solves linear equation system by LDL Cholesky decomposition.
    For linear system Ax = z, and LDL decomposition of A:
        A = DLD'
    Solve the equations sequentially:
        Lu = z
        Dv = u
        L'x = v

    Args:
        ml: Lower diagonal matrix L.
        d: Diagonal of matrix D, 1D array.
        z: Dependent variable.

    Returns:
        array: Solved array.

    """
    n = d.size
    u = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    # solve Lu = z
    u[0] = z[0]
    u[1] = z[1] - ml[1][0] * u[0]
    for i in range(2, n):
        u[i] = z[i] - ml[i][i-1] * u[i-1] - ml[i][i-2] * u[i-2]

    # solve Dv = u
    v = u / d

    # solve L'x = v
    x[n-1] = v[n-1]
    x[n-2] = v[n-2] - ml[n-1][n-2] * x[n-1]
    for i in range(n-3, -1, -1):
        x[i] = v[i] - ml[i+1][i] * x[i+1] - ml[i+2][i] * x[i+2]

    return x


@nb.njit("float64[:,:](float64[:,:], float64[:])")
def inverse_reinsch_coef_matrix(ml, d) -> np.ndarray:
    """
    Calculates the inverse of Reinsch coefficient matrix by LDL
    decomposition.

    Args:
        ml: Lower tri-angular matrix L.
        d: Diagonal of matrix D.

    Returns:
        array: Inverse matrix.

    """
    n = d.size
    d2 = 1 / d
    b = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1, -1, -1):
        d0 = d2[i]
        for k in range(i + 1, min(n, i + 8)):
            v = 0
            for j in range(i + 1, min(n, i + 8)):
                v = v - ml[j][i] * b[j][k]
            b[i][k] = v
            b[k][i] = v

            d0 = d0 - ml[k][i] * v

        b[i][i] = d0

    return b


@nb.njit("float64[:,:](float64[:,:], float64[:,:], float64)", fastmath=True)
def reinsch_coef_matrix(q, r, a) -> np.ndarray:
    """
    Calculates coefficient matrix of Reinsch algorith.

    Args:
        q: Triangular matrix.
        r: Triangular matrix.
        a: Smooth parameter.

    Returns:
        array: hat matrix.

    """
    n, _ = q.shape
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        qi = q[i]
        i1 = i + 3
        for j in range(max(0, i - 2), min(n, i1)):
            qj = q[j]
            v = 0
            for k in range(max(i, j), min(i1, j+3)):
                v += qi[k] * qj[k]
            c[i, j] = v * a

        for j in range(max(0, i - 1), min(n, i + 2)):
            c[i, j] += r[i, j]

    return c


@nb.njit("UniTuple(float64[:,:], 2)(float64[:])")
def cubic_splines(x):
    """
    Generates cubic spline matrices.

    Args:
        x: Knots

    Returns:
        array: Spline matrix

    """
    n = x.size
    h = x[1:] - x[:-1]
    h2 = 1. / h
    r = np.zeros((n-2, n-2), dtype=np.float64)
    q = np.zeros((n-2, n), dtype=np.float64)
    for i in range(1, n-1):
        j = i - 1
        q[j][j] = h2[j]
        q[j][i] = -(h2[j] + h2[i])
        q[j][i+1] = h2[i]

    for i in range(1, n-2):
        j = i - 1
        r[j][j] = (h[j] + h[i]) / 3
        r[j][i] = h[i] / 6
        r[i][j] = h[i] / 6
    r[i][i] = (h[i] + h[i-1]) / 3

    return q, r


@nb.njit("float64[:,:](float64[:], float64[:], float64[:])")
def spline_coef(x, z, g) -> np.ndarray:
    """
    Calculates spline coefficients.

    Args:
        x: Interval edges.
        z: Smoothed values.
        g: Second derivatives.

    Returns:
        array: Coefficients.

    """
    n = x.size
    g2 = np.zeros(n, dtype=np.float64)
    g2[1:n-1] = g

    c = np.zeros((4, n-1), dtype=np.float64)
    c[0] = z[:-1]
    c[2][1:] = g / 2
    # coefficients b
    df = z[1:] - z[:-1]
    dx = x[1:] - x[:-1]
    b = np.zeros(n-1, dtype=np.float64)
    for i in range(n-1):
        b[i] = df[i] / dx[i] - dx[i] / 6 * (2 * g2[i] + g2[i+1])
    c[1] = b
    # coefficients d
    c[3] = (g2[1:] - g2[:-1]) / dx / 6

    return c


@nb.njit("float64[:](float64[:], float64[:,:], float64[:], float64[:])")
def coef_predict(x, coefs, intervals, g) -> np.ndarray:
    """
    Predicts x using coefficients.

    Args:
        x: x for prediction.
        coefs: Coefficients.
        intervals: Interval edges for smoothing.
        g: Smoothed array.

    Returns:
        array: Predicted array.

    """
    n = x.size
    nk = intervals.size
    ix = np.searchsorted(intervals, x) - 1
    p = np.zeros(n, dtype=np.float64)
    for i in range(n):
        j = ix[i]
        if j < 0:
            p[i] = g[0]
        elif j == nk - 1:
            p[i] = g[-1]
        else:
            v = 0
            for m in range(4):
                v += coefs[m][j] * (x[i] - intervals[j]) ** m
            p[i] = v
    return p


@nb.njit("float64[:](float64[:], float64[:], float64[:], float64[:])")
def predict(x, g, d, intervals) -> np.ndarray:
    """
    Predicts x.

    Args:
        x: x for prediction.
        g: Smoothed array.
        d: 2nd derivative, with 0 padded at two terminal sides of the
            array.
        intervals: Interval edges for fitting.

    Returns:
        array: Predicted array.

    """
    n = intervals.size
    nx = x.size
    h = intervals[1:] - intervals[:-1]
    ix = np.searchsorted(intervals, x)
    p = np.zeros(nx, dtype=np.float64)
    for i in range(nx):
        jr = ix[i]
        jl = jr - 1
        if jr == 0:
            p[i] = g[0]
        elif jr == n:
            p[i] = g[-1]
        else:
            d_lo = x[i] - intervals[jl]
            d_hi = intervals[jr] - x[i]
            t = (d_lo * g[jr] + d_hi * g[jl]) / h[jl]
            a = - d_lo * d_hi / 6 * ((1 + d_lo/h[jl]) * d[jr]
                                     + (1 + d_hi/h[jl]) * d[jl])
            p[i] = t + a
    return p


class SplineSmoothing:
    """
    This class performs spline smoothing with a series of smoothing
    parameters, with optimized one selected using improved AIC and
    generalized cross validation.

    Args:
        smooth_params (optional, list): Smoothing parameters.
            Defaulted to 1 to 10^6.
        criteria (str): Criterion for selection of parameters,
            {`aicc`, `gcv`, `cv`}
            `aicc`: Improved AIC, this is the default.
            `gcv`: Generalized cross validation.
            `cv`: Cross validation.

    """
    def __init__(self,
                 smooth_params: Optional[List[float]] = None,
                 criteria: str = "aicc"):

        self.smooth_params = [10 ** v for v in range(8)]
        self.criteria = criteria

        if smooth_params is not None:
            self.smooth_params = smooth_params

        self._Q: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None
        self._score: Dict[str, List[float]] = {"aicc": [], "cv": [], "gcv": []}
        self._best_index: Optional[int] = None
        self._x: Optional[np.ndarray] = None
        self._coefficients: Optional[np.ndarray] = None
        self._d2: Optional[np.ndarray] = None
        self._best_fit: Optional[np.ndarray] = None

        self._check_params()

    @property
    def best_smoothing_param(self) -> float:
        """
        Returns the best smoothing parameters.

        Raises:
            ValueError

        """
        if self._best_index is None:
            raise ValueError("The smoothing is not performed.")
        return self.smooth_params[self._best_index]

    @property
    def best_criterion(self) -> float:
        """
        Returns the lowest criterion value.

        Raises:
            ValueError

        """
        if self._best_index is None:
            raise ValueError("The smoothing is not performed.")
        return self._score[self.criteria][self._best_index]

    @property
    def cv_scores(self) -> List[Tuple[float, float]]:
        """ Returns cross validation values. """
        return list(zip(self.smooth_params, self._score["cv"]))

    @property
    def gcv_scores(self) -> List[Tuple[float, float]]:
        """ Returns generalized cross validation scores. """
        return list(zip(self.smooth_params, self._score["gcv"]))

    @property
    def aicc_scores(self) -> List[Tuple[float, float]]:
        """ Returns AICC scores. """
        return list(zip(self.smooth_params, self._score["aicc"]))

    @property
    def interval_edges(self) -> Optional[np.ndarray]:
        """ Returns interval edges. """
        if self._best_index is None:
            return None
        return self._x

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """ Returns coefficients of splines. """
        if self._best_index is None:
            return None
        return self._coefficients

    @property
    def q(self):
        """ Returns matrix Q. """
        return self._Q

    @property
    def r(self):
        """ Returns matrix R. """
        return self._R

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates spline smoothing.

        Args:
            x: Knots, must be in strictly increasing order.
            y: Responses corresponding to x.

        Returns:
            array: Smoothed values.

        """
        self._check_array(x, y)
        return self._fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the smoothing curve.

        Args:
            x: Array for evaluation.

        Returns:
            array: Evaluated curve.

        """
        if self._coefficients is None:
            raise RuntimeError("The coefficients are not estimated. Should "
                               "run the fit in advance.")
        x = self._check_pred_x(x)

        return predict(x, self._best_fit, self._d2, self._x)

    def _fit(self, x, y) -> np.ndarray:
        """ Smooths the arrays. """
        self._initialize_scores()

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # Q, R matrices
        q, r = cubic_splines(x)
        self._Q = q
        self._R = r
        self._x = x

        # Q * y
        z = _q_y(q, y)

        # selection of smoothing parameters
        fits = []
        devs = []
        for p in self.smooth_params:
            s = reinsch_coef_matrix(q, r, p)
            # DLD Cholesky decomposition
            ml, d = ldl_decompose(s)
            # gamma by solving linear system
            m = solve_linear(ml, d, z)
            # diagonal elements of hat matrix
            b = inverse_reinsch_coef_matrix(ml, d)
            diag_hat = hat_matrix_diagonal(q, b, p)
            # fitted curve
            err = p * np.dot(m, q)
            fv = y - err

            self._calculate_criterion(diag_hat, err)
            fits.append(fv)
            devs.append(m)

        i = np.argmin(self._score[self.criteria])
        best_fit = fits[i]
        self._best_index = i
        self._best_fit = best_fit

        # coefficients
        coef = spline_coef(x, best_fit, devs[i])
        self._coefficients = coef

        d2 = np.zeros_like(x)
        d2[1:-1] = devs[i]
        self._d2 = d2

        return best_fit

    def _calculate_criterion(self, hat_diagonals, residuals):
        """
        Calculates criterion: GCV, AICC, CV

        Args:
            hat_diagonals: Diagonal elements of hat matrix.
            residuals: Residuals of fitting

        """
        n = residuals.size

        # CV
        loo_err = residuals / (1 - hat_diagonals)
        cv = (loo_err * loo_err).mean()

        # GCV
        t = hat_diagonals.sum()
        mse = (residuals * residuals).mean()
        gcv = mse / ((1 - t / n) ** 2)

        # AICC
        aicc = np.log(mse) + 1 + 2 * (1 + t) / (n - t - 2)

        self._score["gcv"].append(gcv)
        self._score["cv"].append(cv)
        self._score["aicc"].append(aicc)

    def _initialize_scores(self):
        """ Initializes scores for new fitting. """
        self._score = {"aicc": [], "cv": [], "gcv": []}
        self._best_index = None
        self._coefficients = None

    def _check_params(self):
        """ Check whether parameters are valid. """
        if min(self.smooth_params) <= 0:
            raise ValueError("The smooth parameters must be non-negative.")

        if self.criteria not in ("aicc", "gcv", "cv"):
            raise ValueError(f"Expected 'aicc' or 'gcv', got {self.criteria}.")

    @staticmethod
    def _check_array(x, y):
        """ Checks whether the knots array is valid. """
        x_shape = x.shape
        y_shape = y.shape
        if x_shape != y_shape:
            raise ValueError("Shapes of x and y must be consistent.")

        if x.ndim != 1:
            raise ValueError("Currently only 1D is accepted.")

        if x.size < 3:
            raise ValueError(f"The number of knots must be larger than 3.")

        xd = np.diff(x)
        if (xd <= 0).any():
            raise ValueError("The knots array x must be strictly increasing.")

    def _check_pred_x(self, x):
        """
        Checks x for prediction. This prediction only allow all x
        values in the range of x for constructing smoothing splines.

        Raises:
            ValueError

        """
        x0 = self._x.min(initial=None)
        xm = self._x.max(initial=None)
        if x.min(initial=None) < x0 or x.max(initial=None) > xm:
            raise ValueError("Values in array for prediction are out of "
                             "range for fitting, which is not currently "
                             "implemented.")

        return x.astype(np.float64)

#
# if __name__ == "__main__":
#
#     import matplotlib.pyplot as plt
#
#     areas = []
#     with open(r"../tests/curve_data.txt", "r") as f:
#         for line in f:
#             areas.append(float(line.rstrip()))
#     areas = np.fromiter(areas, np.float64)
#     areas /= np.median(areas)
#
#     xx = np.arange(areas.size)
#     jx = np.fromiter([*[i * 10 for i in range(17)], 165], np.int64)
#
#     spline = SplineSmoothing(criteria="aicc")
#     yp = spline.fit(jx, areas[jx])
#
#     fig = plt.figure(1)
#     ax = fig.add_subplot()
#     scores = np.array([list(s) for s in spline.gcv_scores])
#     ax.plot(np.log10(scores[:, 0]), scores[:, 1], c="firebrick")
#
#     yp_2 = spline.predict(xx)
#
#     fig = plt.figure(2)
#     ax = fig.add_subplot()
#     ax.plot(xx, areas, ".", ms=3., c="k")
#     ax.plot(jx, yp, "o", linestyle="none", c="firebrick", mfc="none")
#     ax.plot(xx, yp_2, c="darkgreen", lw=2.)
#     plt.show()
