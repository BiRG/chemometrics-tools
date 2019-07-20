import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target


class PLSValidator:
    def __init__(self, estimator=None):
        self.estimator = estimator if estimator is not None else PLSRegression()

    def determine_n_components(self, X, Y, min_n_components=2, cv=None, scoring=None,
                               n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        X = check_array(X)
        if cv is None:
            if type_of_target(Y).startswith('binary') or type_of_target(Y).startswith('multiclass'):
                cv = StratifiedKFold(5)
            else:
                cv = KFold(5)

        self.estimator.n_components = min_n_components
        prev_score = np.sum(cross_val_score(self.estimator, X, Y, scoring=scoring, cv=cv,
                                            n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch))

        for n_components in range(min_n_components + 1, X.shape[1]):
            self.estimator.n_components = n_components
            next_score = np.sum(cross_val_score(self.estimator, X, Y, scoring=q2d_score, cv=cv,
                                                n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch))
            if prev_score >= next_score:
                self.estimator.n_components -= 1
                break

