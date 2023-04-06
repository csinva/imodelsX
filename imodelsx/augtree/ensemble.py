from copy import deepcopy
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.utils import resample
from tqdm import tqdm

class BaggingEstimatorText:
    def __init__(self, estimator, n_estimators=10, max_samples=1.0,
    bootstrap=True, random_state=None):
        """Use this class because sklearn's class doesn't support passing X_text
        """
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y, X_text, feature_names=None):
        self.estimators_ = []
        for _ in tqdm(range(self.n_estimators)):
            estimator = deepcopy(self.estimator)
            if self.bootstrap:
                X_, y_, X_text_ = resample(X, y, X_text, replace=True, random_state=self.random_state)
            else:
                X_, y_, X_text_ = X, y, X_text
            if self.max_samples < 1.0:
                X_, y_, X_text_ = resample(X_, y_, X_text_, 
                n_samples=int(self.max_samples * X_.shape[0]), random_state=self.random_state)
            X_ = X_.toarray()
            estimator.fit(X=X_, y=y_, X_text=X_text_, feature_names=feature_names)
            self.estimators_.append(estimator)
        return self

    def predict(self, X_text):
        return (self.predict_proba(X_text)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X_text):
        if hasattr(X_text, 'shape'):
            n = X_text.shape[0]
        else:
            n = len(X_text)
        y_pred = np.zeros((n, 2))
        for estimator in self.estimators_:
            y_pred += estimator.predict_proba(X_text)
        y_pred /= len(self.estimators_)
        return y_pred

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "bootstrap_features": self.bootstrap_features,
            "random_state": self.random_state,
        }