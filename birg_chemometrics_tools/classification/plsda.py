from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np


class PLSDiscriminator(BaseEstimator, ClassifierMixin):
    """A wrapper for sklearn.cross_decomposition PLSRegression to create a binary PLS-DA classifier.

    Parameters
    ----------
    estimator: _PLS, default PLSRegression()
        A PLS estimator to wrap. The default estimator is PLSRegression with the default parameters.

    Attributes
    ----------
    binarizer_: LabelBinarizer
        The LabelBinarizer used to binarize the binary variables this is fitted to

    """
    def __init__(self, estimator=None):
        self.estimator = estimator if estimator is not None else PLSRegression()
        self.binarizer_ = LabelBinarizer(-1, 1)

    @property
    def n_components(self):
        return self.estimator.n_components

    @n_components.setter
    def n_components(self, other):
        self.estimator.n_components = other

    def fit(self, X, Y, pos_label=None):
        """Fit model to data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples and must be binary. This can only be a
            single column. The one-v-one or one-v-all split is up to you.

        pos_label : any, default None
            Which value in the target represents the "positive" class. If none, the order of the labels will be
            determined by the binarizer.
        """
        self.binarizer_.fit(Y)
        if pos_label is not None:
            if self.binarizer_.classes_[1] != pos_label:
                self.binarizer_.classes_ = np.flip(self.binarizer_.classes_)
        self.estimator.fit(X, self.binarizer_.transform(Y))

    def predict(self, X):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        check_is_fitted(self.estimator, 'x_mean_')
        return self.binarizer_.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X):
        """Compute pseudo-probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Returns
        -------
        T : array-like, shape (n_samples, n_classes)
            Returns the probability of the sample for the positive class label

        Notes
        -----
        The "probability" is a measure of how "confident" the discriminator is in the prediction.
        A target value of 0 represents a probability of 0.5 (the point is equally likely to belong to either class).
        A target value of 1 or higher represents a probability of 1 for the positive class, 0 for the negative class.
        A target value of -1 or lower represents a probability of 0 for the positive class, 1 for the negative class.
        """
        check_is_fitted(self.estimator, 'x_mean_')
        return 0.5 * (np.clip(self.estimator.predict(X), -1, 1) + 1)

    def r2d_score(self, X, y, sample_weight=None):
        """Compute an adjusted R-squared score
        This score disregards prediction errors beyond the (-1, 1) class labels.
        E.g., responses values of -1.1 and 1.1 are treated as responses of 1 and -1, respectively.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like of shape = (n_samples, 1)
            Ground truth (correct) target values.

        sample_weight : array-like of shape = (n_samples), optional
            Sample weights.

        Returns
        -------
        z : float
            The R^2D (or Q^2D)score

        References
        ----------
        J. A. Westerhuis, E. J. J. van Velzen, H. C. J. Hoefsloot, A. K. Smilde. Discriminant Q^2 (DQ^2) for improved
        discrimination in PLSDA models. Metabolomics 4 (4) 2008.
        """
        check_is_fitted(self.estimator, 'x_mean_')
        y_pred = np.clip(self.estimator.predict(X), -1, 1)
        return r2_score(y, y_pred, sample_weight)
