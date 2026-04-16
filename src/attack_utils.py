from __future__ import annotations

import random
import re
from typing import Iterable

import numpy as np
import pandas as pd

ZERO_WIDTH_SPACE = "\u200b"

TOXIC_HINT_WORDS = {
    "hate",
    "kill",
    "idiot",
    "stupid",
    "moron",
    "trash",
    "dumb",
    "racist",
    "terrorist",
    "die",
    "ugly",
}

HOMOGLYPH_MAP = {
    "a": "а",  # cyrillic a
    "e": "е",  # cyrillic e
    "o": "о",  # cyrillic o
    "p": "р",  # cyrillic er
    "c": "с",  # cyrillic es
    "x": "х",  # cyrillic ha
    "i": "і",  # cyrillic i
    "y": "у",  # cyrillic u
}

WORD_RE = re.compile(r"\w+")


def _insert_zero_width(word: str, rng: random.Random) -> str:
    """Insert zero-width spaces every 2-3 chars for selected words."""
    if len(word) < 4:
        return word
    out = []
    idx = 0
    while idx < len(word):
        step = rng.choice([2, 3])
        out.append(word[idx : idx + step])
        idx += step
    return ZERO_WIDTH_SPACE.join(out)


def _homoglyph_substitute(word: str, rng: random.Random, prob: float = 0.35) -> str:
    chars = []
    for ch in word:
        low = ch.lower()
        if low in HOMOGLYPH_MAP and rng.random() < prob:
            repl = HOMOGLYPH_MAP[low]
            chars.append(repl if ch.islower() else repl.upper())
        else:
            chars.append(ch)
    return "".join(chars)


def _duplicate_characters(word: str, rng: random.Random, prob: float = 0.20) -> str:
    chars = []
    for ch in word:
        chars.append(ch)
        if ch.isalpha() and rng.random() < prob:
            chars.append(ch)
    return "".join(chars)


def perturb(text: str, seed: int | None = None) -> str:
    """
    Apply all required transformations:
    1) zero-width insertion on toxic-looking words
    2) unicode homoglyph substitution
    3) random character duplication (20%)
    """
    rng = random.Random(seed)

    def _transform(match: re.Match[str]) -> str:
        word = match.group(0)
        base = word

        if base.lower() in TOXIC_HINT_WORDS or any(x in base.lower() for x in ["kill", "hate", "idiot"]):
            base = _insert_zero_width(base, rng)

        base = _homoglyph_substitute(base, rng)
        base = _duplicate_characters(base, rng, prob=0.20)
        return base

    return WORD_RE.sub(_transform, text)


def perturb_many(texts: Iterable[str], seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return [perturb(t, seed=rng.randint(0, 10_000_000)) for t in texts]


def compute_attack_success_rate(
    original_probs: np.ndarray,
    attacked_probs: np.ndarray,
    decision_threshold: float = 0.5,
) -> float:
    """ASR = fraction of originally toxic predictions flipped to non-toxic after attack."""
    orig_pred = (original_probs >= decision_threshold).astype(int)
    new_pred = (attacked_probs >= decision_threshold).astype(int)

    denominator_mask = orig_pred == 1
    denom = int(denominator_mask.sum())
    if denom == 0:
        return 0.0

    flipped = int(((orig_pred == 1) & (new_pred == 0)).sum())
    return flipped / denom


def poison_flip_labels(
    train_df: pd.DataFrame,
    flip_fraction: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Flip labels on a random subset for poisoning attack simulation."""
    if not (0.0 < flip_fraction < 1.0):
        raise ValueError("flip_fraction must be in (0,1).")

    poisoned = train_df.copy()
    n = len(poisoned)
    n_flip = int(round(n * flip_fraction))

    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(n, size=n_flip, replace=False)

    poisoned.loc[flip_idx, "label"] = 1 - poisoned.loc[flip_idx, "label"]
    poisoned["is_poisoned_row"] = 0
    poisoned.loc[flip_idx, "is_poisoned_row"] = 1

    return poisoned
