from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class WeightedTrainer(Trainer):
    """Trainer variant that supports per-example sample weights."""

    def compute_loss(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_example_loss = loss_fct(logits, labels)

        if sample_weight is not None:
            sample_weight = sample_weight.float()
            loss = (per_example_loss * sample_weight).mean()
        else:
            loss = per_example_loss.mean()

        return (loss, outputs) if return_outputs else loss


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 128,
    with_sample_weight: bool = False,
) -> Dataset:
    """Convert dataframe to HuggingFace dataset and tokenize text column."""
    keep_cols = ["comment_text", "label"]
    if with_sample_weight and "sample_weight" in df.columns:
        keep_cols.append("sample_weight")

    work_df = df[keep_cols].copy()
    work_df = work_df.rename(columns={"label": "labels"})

    dataset = Dataset.from_pandas(work_df, preserve_index=False)

    def _tokenize(batch):
        return tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(_tokenize, batched=True)
    remove_cols = ["comment_text"]
    dataset = dataset.remove_columns(remove_cols)
    dataset.set_format(type="torch")
    return dataset


def trainer_metrics(eval_pred) -> Dict[str, float]:
    """Metrics callback for HuggingFace Trainer."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }

    # roc_auc_score fails when only one class appears in labels.
    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = float("nan")

    return metrics


def build_training_args(
    output_dir: str | Path,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> TrainingArguments:
    output_dir = str(output_dir)
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=200,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )


def train_distilbert(
    model_name: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: str | Path,
    max_length: int = 128,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    seed: int = 42,
    use_sample_weights: bool = False,
):
    """Train DistilBERT with optional sample weighting."""
    set_global_seed(seed)

    tokenizer = build_tokenizer(model_name)
    model = build_model(model_name)

    train_ds = tokenize_dataframe(
        train_df,
        tokenizer,
        max_length=max_length,
        with_sample_weight=use_sample_weights,
    )
    eval_ds = tokenize_dataframe(eval_df, tokenizer, max_length=max_length)

    args = build_training_args(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_cls = WeightedTrainer if use_sample_weights else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=trainer_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return trainer, tokenizer


def predict_probabilities(
    trainer: Trainer,
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 128,
) -> np.ndarray:
    """Return positive-class probabilities for a dataframe."""
    ds = tokenize_dataframe(df, tokenizer, max_length=max_length)
    output = trainer.predict(ds)
    logits = output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return probs


def predict_probabilities_from_model(
    model,
    tokenizer,
    texts: Iterable[str],
    max_length: int = 128,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Inference helper used by attacks and pipeline demos."""
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = list(texts)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_text,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_probs)
