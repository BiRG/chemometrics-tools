from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils.validation import check_array


class RangeProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, abscissa_min, abscissa_max, copy=False):
        self.abscissa_min = abscissa_min
        self.abscissa_max = abscissa_max
        self.range_indices_ = None
        self.copy = copy

    def fit(self, X, abscissa):
        """
        Learn the indices for the specified range.
        """
        self.range_indices_ = np.where((abscissa >= self.abscissa_min) & (abscissa <= self.abscissa_max))


class RangeZeroer(RangeProcessor):
    def transform(self, X, abscissa, copy=None):
        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy)
        abscissa = check_array(abscissa, copy=copy)
        X[:, self.range_indices_] = 0
        abscissa[:, self.range_indices_] = 0
        return X, abscissa


class RangeDeleter(RangeProcessor):
    def transform(self, X, abscissa, copy=None):
        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy)
        abscissa = check_array(abscissa, copy=copy)
        return np.delete(X, self.range_indices_, axis=1), np.delete(abscissa, self.range_indices_)
