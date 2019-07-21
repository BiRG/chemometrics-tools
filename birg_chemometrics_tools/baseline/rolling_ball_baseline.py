
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
import math


class RollingBallBaseline(BaseEstimator, TransformerMixin):
    """Rolling ball baseline correction
    RollingBallBaseline implements the rolling-ball baseline method based on the ideas from Kneen and Annegarn, but with
    fixed window width as in the R baseline package.

    Parameters
    ----------
    min_max_window : int
        Width of local window for minimization/maximization

    smoothing_window : int
        Width of local window for smoothing.

    Attributes
    ----------
    baseline_ : [n_spectra, n_points] (axis=0) or [n_points, n_spectra] (axis=1)

    References
    ----------
    M.A. Kneen, H.J. Annegarn: Algorithm for fitting XRF, SEM and PIXE X-ray spectra backgrounds. Nuclear Instruments
    and Methods in Physics Research Section B: Beam Interactions with Materials and Atoms. 109-110 1996.
     https://doi.org/10.1016/0168-583X(95)00908-6

    Kristian Hovde Liland and Bjrn-Helge Mevik (2011). baseline: Baseline Correction of Spectra.
    R package version 1.0-1. http://CRAN.R-project.org/package=baseline
    """
    def __init__(self, min_max_window, smoothing_window):
        self.min_max_window = min_max_window
        self.smoothing_window = smoothing_window
        self.baseline_ = None

    def _fit_one_baseline(self, x):
        # x is 1D array (a column or row of X)
        wm = self.min_max_window
        ws = self.smoothing_window
        m = x.size
        T1 = np.zeros(x.shape)
        T2 = np.zeros(x.shape)
        baseline = np.zeros(x.shape)

        # minimize
        u1 = math.ceil(((wm+1) / 2))
        T1[0] = np.min(x[:u1+1])

        # start of spectrum
        for i in range(1, wm):
            u2 = u1 + 1 + ((i + 1) % 2)
            T1[i] = min(np.min(x[u1+1:u2+1]), T1[i-1])
            u1 = u2

        # middle of spectrum
        for i in range(wm, m - wm):
            T1[i] = x[u1+1] if (x[u1+1] <= T1[i-1] and x[u1-wm] != T1[i-1]) else np.min(x[i-wm:i+wm+1])
            u1 += 1

        # end of spectrum
        u1 = m - 2*wm - 2
        for i in range(m-wm, m):
            u2 = u1 + 1 + (i % 2)
            T1[i] - T1[i-1] if np.min(x[u1:u2]) > T1[i-1] else np.min(x[u2:])
            u1 = u2

        # maximize
        u1 = math.ceil((wm + 1) / 2)

        # start of spectrum
        T2[0] = np.max(T1[:u1+1])
        for i in range(1, wm):
            u2 = u1 + 1 + ((i + 1) % 2)
            T2[i] = max(np.max(T1[u1+1:u2+1]), T2[i-1])
            u1 = u2

        # middle of spectrum
        for i in range(wm+2, m-wm+1):
            T2[i] = T1[u1+1] if (T1[u1+1] >= T2[i-1] and T1[u1-wm] != T2[i-1]) else np.max(T1[i-wm:i+wm+1])
            u1 += 1

        # end of spectrum
        u1 = m - 2*wm - 2
        for i in range(m-wm, m):
            u2 = u1 + 1 + (i % 2)
            T2[i] = T2[i-1] if np.max(T1[u1:u2]) < T2[i-1] else np.max(T1[u2:])
            u1 = u2

        np.savetxt('C:/Users/Daniel Foose/T1.csv', T1, delimiter=',')
        np.savetxt('C:/Users/Daniel Foose/T2.csv', T2, delimiter=',')
        # smoothing
        u1 = math.ceil(ws/2) - 1

        # start of spectrum
        v = np.sum(T2[:u1+1])
        for i in range(ws):
            u2 = u1 + 1 + ((i + 1) % 2)
            v += np.sum(T2[u1+1:u2+1])
            baseline[i] = v / u2
            u1 = u2

        # middle of spectrum
        v = np.sum(T2[:2*ws+2])
        baseline[ws] = v / (2*ws+1)
        for i in range(ws+1, m-ws):
            v = v - T2[i-ws-1] + T2[i+ws]
            baseline[i] = v / (2 * ws + 1)
        u1 = m - 2 * ws
        v -= T2[u1]  # sum so far
        baseline[m-ws] = v / (2 * ws)  # mean so far

        # end of spectrum
        for i in range(m-ws+1, m):
            u2 = u1 + 1 + (i % 2)
            v -= np.sum(T2[u1:u2])
            baseline[i] = v / (m - u2)
            u1 = u2

        return baseline

    def fit(self, X, axis=0):
        """Estimate baselines for a collection of spectra

        Parameters
        ----------
        X : [n_spectra, n_points] (axis=0) or [n_points, n_spectra] (axis=1)
            A collection of spectra in rows or columns.
        axis : int
            0 if spectra are in rows, 1 if spectra are in columns

        """
        X = check_array(X)
        baselines = [self._fit_one_baseline(np.ravel(x)) for x in np.split(X, np.size(X, axis), axis)]
        self.baseline_ = np.stack(baselines, axis)
        return self

    def transform(self, X):
        """Correct a collection of spectra with the fit baselines

        Parameters
        ----------
        X : [n_spectra, n_points] (axis=0 in fit()) or [n_points, n_spectra] (axis=1 in fit())
            The spectra to baseline-correct. Note: this should be the same X as in fit.
        """
        return X - self.baseline_
