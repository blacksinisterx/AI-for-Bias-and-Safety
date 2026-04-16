import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def infer_toxicity_column(df: pd.DataFrame) -> str:
    """Infer which column stores toxicity score in [0,1]."""
    candidates = ["toxic", "target", "toxicity"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find toxicity score column in dataframe. Tried: {candidates}."
    )


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load Jigsaw dataset and create standardized columns used across notebooks."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    toxic_col = infer_toxicity_column(df)

    if "comment_text" not in df.columns:
        raise ValueError("Expected column 'comment_text' not found.")

    for identity_col in ["black", "white"]:
        if identity_col not in df.columns:
            raise ValueError(f"Expected identity column '{identity_col}' not found.")

    df = df.copy()
    df["toxic_score"] = df[toxic_col].astype(float)
    df["label"] = (df["toxic_score"] >= 0.5).astype(int)
    df["comment_text"] = df["comment_text"].fillna("")

    return df


def make_stratified_subsets(
    df: pd.DataFrame,
    train_size: int,
    eval_size: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Create non-overlapping eval and train subsets stratified by binarized label."""
    if train_size + eval_size > len(df):
        raise ValueError(
            f"Requested train_size + eval_size = {train_size + eval_size}, "
            f"but dataset has only {len(df)} rows."
        )

    labels = df["label"].to_numpy()
    indices = np.arange(len(df))

    # Step 1: sample evaluation subset from full data.
    eval_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=eval_size,
        random_state=random_state,
    )
    remaining_idx, eval_idx = next(eval_splitter.split(indices, labels))

    # Step 2: sample training subset from remaining data.
    remaining_labels = labels[remaining_idx]
    remaining_indices = indices[remaining_idx]
    train_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )
    train_take, _ = next(train_splitter.split(remaining_indices, remaining_labels))
    train_idx = remaining_indices[train_take]

    train_df = df.iloc[train_idx].copy().reset_index(drop=False)
    eval_df = df.iloc[eval_idx].copy().reset_index(drop=False)

    overlap = set(train_df["index"]).intersection(set(eval_df["index"]))
    if overlap:
        raise RuntimeError("Data leakage detected: train/eval subsets overlap.")

    metadata = {
        "full_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_positive": int(train_df["label"].sum()),
        "eval_positive": int(eval_df["label"].sum()),
        "seed": int(random_state),
    }

    return train_df, eval_df, metadata


def save_split_indices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_path: str | Path,
    metadata: Dict[str, int] | None = None,
) -> None:
    """Persist selected source row indices to keep splits consistent across all parts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "train_indices": [int(x) for x in train_df["index"].tolist()],
        "eval_indices": [int(x) for x in eval_df["index"].tolist()],
        "metadata": metadata or {},
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_indices(split_path: str | Path) -> Dict[str, list[int]]:
    """Load stored split indices."""
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found at {split_path}. Run split creation first (Part 1)."
        )

    data = json.loads(split_path.read_text(encoding="utf-8"))
    required_keys = ["train_indices", "eval_indices"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Split file missing required key: {key}")

    return data


def build_subsets_from_indices(
    df: pd.DataFrame,
    split_payload: Dict[str, list[int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/eval subsets from source-row indices."""
    train_idx = split_payload["train_indices"]
    eval_idx = split_payload["eval_indices"]

    train_df = df.iloc[train_idx].copy().reset_index(drop=False)
    eval_df = df.iloc[eval_idx].copy().reset_index(drop=False)

    overlap = set(train_df["index"]).intersection(set(eval_df["index"]))
    if overlap:
        raise RuntimeError("Loaded split has overlap between train and eval indices.")

    return train_df, eval_df


def class_balance(df: pd.DataFrame) -> Dict[str, float]:
    """Quick class balance report for binary labels."""
    pos_rate = float(df["label"].mean()) if len(df) else 0.0
    return {
        "rows": int(len(df)),
        "positive_count": int(df["label"].sum()),
        "positive_rate": pos_rate,
    }
