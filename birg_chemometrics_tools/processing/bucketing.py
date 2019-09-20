from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class Bucketer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.abscissa_min_ = None
        self.abscissa_max_ = None
        self.abscissa_ = None
        self.bin_inds_ = None


class UniformBucketer(Bucketer):
    def __init__(self, bin_width):
        super().__init__()
        self.bin_width = bin_width

    def fit(self, X, abscissa):
        self.abscissa_min_ = np.arange(np.min(abscissa), np.max(abscissa), self.bin_width)
        self.abscissa_max_ = self.abscissa_min_ + self.bin_width
        self.abscissa_ = self.abscissa_min_ + 0.5 * self.bin_width
        self.bin_inds_ = [
            np.where((abscissa >= abscissa_min) & (abscissa < abscissa_max))
            for abscissa_min, abscissa_max in zip(self.abscissa_min_, self.abscissa_max_)
        ]

    def transform(self, X):
        return np.row_stack([
            np.array(np.sum(row[inds]) for inds in self.bin_inds_)
            for row in np.vsplit(X, X.shape[0])
        ])
