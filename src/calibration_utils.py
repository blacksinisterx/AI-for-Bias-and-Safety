from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

try:
    # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover
    FrozenEstimator = None


class ProbabilityInputClassifier(BaseEstimator, ClassifierMixin):
    """Treats the single input feature as model probability for class 1."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        x = np.asarray(X).reshape(-1)
        x = np.clip(x, 0.0, 1.0)
        return np.vstack([1.0 - x, x]).T

    def predict(self, X):
        return (np.asarray(X).reshape(-1) >= 0.5).astype(int)


def fit_isotonic_calibrator(raw_prob: np.ndarray, y_true: np.ndarray) -> CalibratedClassifierCV:
    """
    Fit sklearn CalibratedClassifierCV(method='isotonic') on raw probabilities.
    """
    X = np.asarray(raw_prob).reshape(-1, 1)
    y = np.asarray(y_true).astype(int)

    base = ProbabilityInputClassifier().fit(X, y)

    # sklearn <=1.5 uses cv='prefit'. sklearn >=1.6 expects a FrozenEstimator.
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(base),
            method="isotonic",
            cv=None,
        )
    else:
        calibrator = CalibratedClassifierCV(base, method="isotonic", cv="prefit")

    calibrator.fit(X, y)
    return calibrator


def calibrate_probabilities(
    calibrator: CalibratedClassifierCV,
    raw_prob: np.ndarray,
) -> np.ndarray:
    X = np.asarray(raw_prob).reshape(-1, 1)
    return calibrator.predict_proba(X)[:, 1]
