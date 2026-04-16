from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import confusion_matrix


HIGH_BLACK_CONDITION = "black >= 0.5"
REFERENCE_CONDITION = "black < 0.1 and white >= 0.5"


def build_cohorts(eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    high_black = eval_df.query(HIGH_BLACK_CONDITION).copy()
    reference = eval_df.query(REFERENCE_CONDITION).copy()
    return high_black, reference


def confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "TPR": tpr,
        "FPR": fpr,
        "FNR": fnr,
        "Precision": precision,
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
    }


def compute_cohort_metrics(
    cohort_df: pd.DataFrame,
    threshold: float,
    prob_col: str = "y_prob",
) -> Dict[str, float]:
    y_true = cohort_df["label"].to_numpy()
    y_pred = (cohort_df[prob_col].to_numpy() >= threshold).astype(int)
    return confusion_rates(y_true, y_pred)


def compute_bias_audit(
    eval_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    work = eval_df.copy()
    work["y_prob"] = y_prob

    high_black, reference = build_cohorts(work)

    hb_metrics = compute_cohort_metrics(high_black, threshold=threshold)
    ref_metrics = compute_cohort_metrics(reference, threshold=threshold)

    di_ratio = (
        hb_metrics["FPR"] / ref_metrics["FPR"]
        if ref_metrics["FPR"] > 0
        else float("nan")
    )

    summary = pd.DataFrame(
        [
            {"cohort": "high_black", **hb_metrics, "size": len(high_black)},
            {"cohort": "reference", **ref_metrics, "size": len(reference)},
        ]
    )

    work["pred_label"] = (work["y_prob"] >= threshold).astype(int)
    aif360_metrics = compute_aif360_metrics(work)

    return {
        "summary_table": summary,
        "disparate_impact_fpr_ratio": di_ratio,
        "aif360": aif360_metrics,
        "high_black": high_black,
        "reference": reference,
    }


def compute_aif360_metrics(df_with_preds: pd.DataFrame) -> Dict[str, float]:
    """
    Compute statistical parity difference and equal opportunity difference
    on two cohorts: high-black (unprivileged) and reference (privileged).
    """
    work = df_with_preds.copy()
    hb_mask = work.eval(HIGH_BLACK_CONDITION)
    ref_mask = work.eval(REFERENCE_CONDITION)

    # Keep only rows belonging to either fairness cohort.
    sub = work[hb_mask | ref_mask].copy()
    if sub.empty:
        return {
            "statistical_parity_difference": float("nan"),
            "equal_opportunity_difference": float("nan"),
        }

    sub["group"] = np.where(sub.eval(HIGH_BLACK_CONDITION), 1, 0)

    true_df = sub[["label", "group"]].copy()
    pred_df = sub[["pred_label", "group"]].copy().rename(columns={"pred_label": "label"})

    dataset_true = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=true_df,
        label_names=["label"],
        protected_attribute_names=["group"],
    )
    dataset_pred = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=pred_df,
        label_names=["label"],
        protected_attribute_names=["group"],
    )

    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=[{"group": 1}],
        privileged_groups=[{"group": 0}],
    )

    return {
        "statistical_parity_difference": float(metric.statistical_parity_difference()),
        "equal_opportunity_difference": float(metric.equal_opportunity_difference()),
    }


def plot_grouped_rates(summary_df: pd.DataFrame, title: str = "Cohort Rate Comparison") -> None:
    metrics = ["TPR", "FPR", "FNR"]
    x = np.arange(len(metrics))
    width = 0.35

    hb = summary_df.loc[summary_df["cohort"] == "high_black", metrics].values.flatten()
    ref = summary_df.loc[summary_df["cohort"] == "reference", metrics].values.flatten()

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, hb, width, label="high_black")
    plt.bar(x + width / 2, ref, width, label="reference")
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.show()


def cohort_confusion_matrix(
    cohort_df: pd.DataFrame,
    threshold: float,
    prob_col: str = "y_prob",
) -> np.ndarray:
    y_true = cohort_df["label"].to_numpy()
    y_pred = (cohort_df[prob_col].to_numpy() >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
