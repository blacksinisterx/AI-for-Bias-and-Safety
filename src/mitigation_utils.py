from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score


HIGH_BLACK_CONDITION = "black >= 0.5"
REFERENCE_CONDITION = "black < 0.1 and white >= 0.5"


class ScoreEstimator(BaseEstimator, ClassifierMixin):
    """Estimator wrapper that treats incoming X values as probabilities/scores."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        x = np.asarray(X).reshape(-1)
        x = np.clip(x, 0.0, 1.0)
        return np.vstack([1.0 - x, x]).T

    def predict(self, X):
        return (np.asarray(X).reshape(-1) >= 0.5).astype(int)


def add_fairness_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["is_high_black"] = work.eval(HIGH_BLACK_CONDITION).astype(int)
    work["is_reference"] = work.eval(REFERENCE_CONDITION).astype(int)
    return work


def compute_reweighing_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Compute sample weights with AIF360 Reweighing for target cohorts."""
    work = add_fairness_group_columns(train_df)

    # Add a neutral feature so BinaryLabelDataset has at least one feature column.
    work["_dummy_feature"] = 0.0

    aif_df = work[["label", "is_high_black", "is_reference", "_dummy_feature"]].copy()

    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=aif_df,
        label_names=["label"],
        protected_attribute_names=["is_high_black", "is_reference"],
    )

    rw = Reweighing(
        unprivileged_groups=[{"is_high_black": 1, "is_reference": 0}],
        privileged_groups=[{"is_high_black": 0, "is_reference": 1}],
    )

    transformed = rw.fit_transform(dataset)
    return np.asarray(transformed.instance_weights)


def oversample_high_black(train_df: pd.DataFrame, duplicate_times: int = 3) -> pd.DataFrame:
    """Duplicate high-black rows duplicate_times in addition to original row."""
    work = train_df.copy()
    hb = work.query(HIGH_BLACK_CONDITION).copy()
    if hb.empty:
        return work

    dup_frames = [hb.copy() for _ in range(duplicate_times)]
    out = pd.concat([work] + dup_frames, axis=0, ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


@dataclass
class ThresholdSweepResult:
    tol: float
    f1_macro: float
    equal_opportunity_difference: float
    statistical_parity_difference: float


def threshold_optimize_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_features: Iterable,
    tol: float = 0.0,
) -> np.ndarray:
    """Apply Fairlearn ThresholdOptimizer with equalized_odds constraint."""
    X = np.asarray(y_prob).reshape(-1, 1)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(list(sensitive_features))

    estimator = ScoreEstimator().fit(X, y)
    optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints="equalized_odds",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
        tol=tol,
    )
    optimizer.fit(X, y, sensitive_features=s)
    y_pred = optimizer.predict(X, sensitive_features=s)
    return np.asarray(y_pred).astype(int)


def sweep_threshold_optimizer(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_features: Iterable,
    fairness_metric_fn,
    parity_metric_fn,
    tolerances: Iterable[float],
) -> pd.DataFrame:
    """Sweep tolerance and return Pareto-ready summary."""
    rows = []
    for tol in tolerances:
        y_pred = threshold_optimize_predictions(
            y_true=y_true,
            y_prob=y_prob,
            sensitive_features=sensitive_features,
            tol=float(tol),
        )
        rows.append(
            {
                "tol": float(tol),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "equal_opportunity_difference": float(fairness_metric_fn(y_true, y_pred, sensitive_features)),
                "statistical_parity_difference": float(parity_metric_fn(y_true, y_pred, sensitive_features)),
            }
        )

    return pd.DataFrame(rows)
