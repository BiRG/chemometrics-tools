from typing import Iterable

from scipy.optimize import minimize_scalar
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import Normalizer
from sklearn.utils import check_array
import numpy as np
from sklearn.utils.validation import check_is_fitted


class SumNormalizer(Normalizer):
    """
    A scaled L1 norm, which transforms data so that the sum of each row is a specified sum.

    Parameters
    ----------
    new_sum: float
        The value each row of the transformed data will sum to.

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).
    """
    def __init__(self, new_sum=1.0, copy=True):
        super().__init__('l1', copy)
        self.new_sum = new_sum

    def transform(self, X, copy=None):
        X = check_array(X, copy=copy)
        X = self.new_sum * super().transform(X, copy)
        return X


class MinMaxNormalizer(Normalizer):
    """
    Like max norm, but with the data shifted so that the minimum is zero prior to normalization, fixing data points to
    [0, 1]

    Parameters
    ----------
    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).
    """
    def __init__(self, copy=True):
        super().__init__('max', copy)

    def transform(self, X, copy=None):
        X = check_array(X, copy=copy)
        X = X - np.min(X, axis=1).reshape(-1, 1)
        return super().transform(X, copy)


class PeakNormalizer(BaseEstimator, TransformerMixin):
    """
    Set the value of the maximum within a window to 1 and scale other points accordingly

    Parameters
    ----------
    abscissa_min: minimum value of X specifying scale by window

    abscissa_max: maximum value of X specifying scale by window

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).

    Notes
    -----
    Unlike with other sklearn transformers, this transforms "y" rather than "X"
    """
    def __init__(self, abscissa_min, abscissa_max, copy):
        self.abscissa_min = abscissa_min
        self.abscissa_max = abscissa_max
        self.copy = copy
        self.peak_maxes_ = None

    def fit(self, X, abscissa):
        """
        Learn the maxes of the specified range.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like, 2D, features
        abscissa : the feature labels
        """
        X = np.ravel(check_array(X, accept_sparse='csr', copy=True))
        abscissa = check_array(abscissa, copy=True)
        X = X - np.min(X, axis=1).reshape(-1, 1)
        inds = np.where((abscissa >= self.abscissa_min) & (abscissa <= self.abscissa_max))
        self.peak_maxes_ = np.max(X[:, inds].reshape(X.shape[0], -1), axis=1)
        return self

    def transform(self, X, copy=None):
        """Scale each non zero row of X to unit norm

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
            X in this function is y in the previous function
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy)
        check_is_fitted(self, 'peak_maxes_')
        X = X - np.min(X, axis=1).reshape(-1, 1)
        X = X / self.peak_maxes_.reshape(X.shape[0], -1)
        return X


class ProbabilisticQuotientNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_type = 'median'):
        self.reference_type = reference_type
        self.reference_spectrum = None

    def fit(self, X):
        """
        here X is the QC data
        """
        X = check_array(X)
        if self.reference_type == 'mean':
            self.reference_spectrum = np.mean(X, axis=0)
        elif self.reference_type == 'median':
            self.reference_spectrum = np.median(X, axis=0)
        else:
            raise ValueError(f'Cannot calculate reference spectrum with reference_type "{self.reference_type}"')

    def quotients(self, X):
        """determine the quotients"""
        return X / self.reference_spectrum

    def transform(self, X, copy=True):
        X = check_array(X, copy=copy)
        X = X / np.median(X / self.reference_spectrum)
        return X / np.median(X / self.reference_spectrum)


class HistogramNormalizer(BaseEstimator, TransformerMixin):
    """Perform Torgrip's histogram normalization on spectra

    Parameters
    ----------
    n_bins: int
        The number of histogram bins to use.

    n_std: int
        The number of standard deviations to set as the threshold for inclusion in the histogram.
        Any points below n_std * noise_std_ will be ignored.

    noise_ind: int or array/list of int
        If an integer, the number of points at the beginning of the spectra to consider noise.
        If an array or list, the indices to use as noise. This is applied on a row by row basis.


    References
    ----------
    R. J. O. Torgrip, K. M. Aberg, E. Alm, I. Schuppe-Koistinen and J. Lindberg. Metabolomics (2008) 4:114-121
    DOI: 10.1007/s11306-007-0102-2

    """
    def __init__(self, n_bins=60, n_std=5, noise_ind=35):
        self.n_bins = n_bins
        self.n_std = n_std
        if isinstance(noise_ind, int):
            self.noise_ind = np.array([i for i in range(0, noise_ind)])
        elif isinstance(noise_ind, Iterable):
            self.noise_ind = noise_ind
        else:
            raise ValueError('noise_ind must be integer or iterable.')

    def fit(self, X):
        return self

    def transform(self, X, target_spectrum=None):
        X = check_array(X, copy=True)
        Z = np.log2(X)
        target_spectrum = np.log2(target_spectrum) if target_spectrum is not None else np.nanmedian(Z, axis=0)

        noise_std = X[:, self.noise_ind].std()
        bin_edges = np.histogram_bin_edges(Z[(X < noise_std) & np.isfinite(Z)], self.n_bins)
        target_histogram, _ = np.histogram(target_spectrum, bin_edges)

        def hist_err(mult, z_vals):
            test_hist, _ = np.histogram(mult * z_vals, bin_edges)
            return np.sum((target_histogram - test_hist) ** 2)

        def optimize_spectrum(X_i, Z_i):
            hist, _ = np.histogram(Z_i[X_i < noise_std], bin_edges)
            # initial search bounds are
            low_b = np.min(target_histogram) / np.max(target_histogram)
            up_b = np.max(target_histogram) / (np.min(target_histogram) or 1.0)
            n_steps = np.ceil(np.log2(up_b/low_b)).astype(int)
            bound_mults = [low_b * (2 ** i) for i in range(0, n_steps)]
            errs = np.array([hist_err(bound_mult, Z_i) for bound_mult in bound_mults])
            # indices of two elements that bound first interval of minimum error
            try:
                min_err_idx = np.where(errs == np.min(errs))[0].item()
            except:
                min_err_idx = 0
            try:
                low_b_idx = np.where(errs[:min_err_idx] > errs[min_err_idx])[-1].item()
            except ValueError:  # throws when np.where returns empty
                low_b_idx = 0
            try:
                up_b_idx = np.where(errs[min_err_idx+1:] > errs[min_err_idx])[0].item()
            except ValueError:
                up_b_idx = len(errs) - 1
            up_b = bound_mults[up_b_idx] if bound_mults[up_b_idx] < up_b else up_b
            low_b = bound_mults[low_b_idx] if bound_mults[low_b_idx] > low_b else low_b
            mult = minimize_scalar(hist_err, bounds=(low_b, up_b), args=[Z_i], method='bounded').x
            print(f'mult: {mult}')
            return X_i * mult

        return np.row_stack([
            optimize_spectrum(X_i, Z_i)
            for X_i, Z_i in zip(np.vsplit(X, X.shape[0]), np.vsplit(Z, X.shape[0]))
        ])