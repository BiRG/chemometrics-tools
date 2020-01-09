"""
The HistogramNormalization class contains source code from scikit-image

Copyright (C) 2019, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from typing import Iterable

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
    def __init__(self, abscissa_min, abscissa_max, copy=True):
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
    def __init__(self, reference_type='median'):
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
    """Perform histogram normalization on spectra. Inspired by Horgrip, but using an estimate of the cumulative
     distribution function as in scikit-image.

    Parameters
    ----------
    aggregate: str or function
        An aggregate function to use. 'median' and 'mean' are acceptable, as is a callable that has "axis" as a kwarg,
        like an NumPy aggregate function


    References
    ----------
    R. J. O. Torgrip, K. M. Aberg, E. Alm, I. Schuppe-Koistinen and J. Lindberg. Metabolomics (2008) 4:114-121
    DOI: 10.1007/s11306-007-0102-2

    """
    def __init__(self, aggregate='median'):
        """
        self.n_std = n_std
        if isinstance(noise_ind, int):
            self.noise_ind = np.array([i for i in range(0, noise_ind)])
        elif isinstance(noise_ind, Iterable):
            self.noise_ind = noise_ind
        else:
            raise ValueError('noise_ind must be integer or iterable.')
        """
        if callable(aggregate):
            self.aggregate_function = aggregate
        elif aggregate == 'mean':
            self.aggregate_function = np.mean
        elif aggregate == 'median':
            self.aggregate_function = np.median
        else:
            raise ValueError(f'Cannot interpret aggregate function {aggregate}. Acceptable aggregate functions are '
                             f'\'mean\' and \'median\', or you may pass a callable with "axis" as a kwarg such as '
                             f'NumPy aggregate functions.')
        self.reference_spectrum_ = None

    @staticmethod
    def _match_cdf(spectrum, reference):
        spectrum_values, spectrum_unique_indices, spectrum_counts = np.unique(spectrum.ravel(),
                                                                              return_inverse=True,
                                                                              return_counts=True)
        reference_values, reference_counts = np.unique(reference.ravel(), return_counts=True)

        spectrum_quantiles = np.cumsum(spectrum_counts) / spectrum.size
        reference_quantiles = np.cumsum(reference_counts) / reference.size

        interp_a_values = np.interp(spectrum_quantiles, reference_quantiles, reference_values)
        return interp_a_values[spectrum_unique_indices].reshape(spectrum.shape)

    def fit(self, X):
        # X here is the reference spectrum, which we will store
        X = check_array(X)
        self.reference_spectrum_ = self.aggregate_function(X, axis=0)
        return self

    def transform(self, X):
        X = check_array(X)
        return np.vstack([self._match_cdf(spectrum, self.reference_spectrum_)
                          for spectrum in np.vsplit(X, X.shape[0])])
