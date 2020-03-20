from sklearn.base import BaseEstimator, TransformerMixin
from pyyawt.denoising import wden
import numpy as np
from sklearn.utils.validation import check_is_fitted


class WaveletDenoiser(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_rule='sqtwolog', threshold_type='s', scale='one', level=1, wavelet_name='sym4'):
        self.threshold_rule = threshold_rule
        self.threshold_type = threshold_type
        self.scale = scale
        self.level = level
        self.wavelet_name = wavelet_name
        self.denoised_ = None
        self.coef_ = None
        self.length_ = None

    def fit(self, X):
        results = [
            wden(X=x,
                 TPTR=self.threshold_rule,
                 SORH=self.threshold_type,
                 SCAL=self.scale,
                 N=self.level,
                 wname=self.wavelet_name)
            for x in np.vsplit(X)
        ]
        self.denoised_ = np.row_stack([result[0] for result in results])
        self.coef_ = [result[1] for result in results]
        self.length_ = [result[2] for result in results]
        return self

    def transform(self, X):
        check_is_fitted(self, ['denoised_'])
        if self.denoised_.shape == X.shape:
            return self.denoised_
        else:
            raise ValueError('Passed X is different shape from that used in fit. This function can only be applied to '
                             'the same X the estimator was fitted with.')
