from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BASE_MODEL_NAME, MAX_LENGTH, MODELS_DIR


BLOCKLIST = {
    "direct_threat": [
        re.compile(r"\bi\s*(?:will|ll|am\s+going\s+to|gonna)\s+(kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re|\s+are)?\s+going\s+to\s+die\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|ll|gonna)\s+find\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bsomeone\s+should\s+(?:kill|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|ll|gonna)\s+beat\s+you\s+up\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        re.compile(r"\b(?:go\s+)?kill\s+yourself\b", re.IGNORECASE),
        re.compile(r"\byou\s+should\s+kill\s+yourself\b", re.IGNORECASE),
        re.compile(r"\bnobody\s+would\s+miss\s+you\s+if\s+you\s+died\b", re.IGNORECASE),
        re.compile(r"\bdo\s+everyone\s+a\s+favo(?:u)?r\s+and\s+disappear\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi\s+know\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|ll|gonna)\s+post\s+your\s+address\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:found|have\s+found)\s+your\s+real\s+name\b", re.IGNORECASE),
        re.compile(r"\beveryone\s+will\s+know\s+who\s+you\s+really\s+are\b", re.IGNORECASE),
    ],
    "dehumanization": [
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+are\s+not\s+(?:human|people|person)\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+are\s+animals\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+should\s+be\s+exterminated\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+are\s+a\s+disease\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        re.compile(r"\beveryone\s+report\s+\S+\b", re.IGNORECASE),
        re.compile(r"\blet'?s\s+all\s+go\s+after\b", re.IGNORECASE),
        re.compile(r"\bmass\s+report\b(?=\s+\S+)", re.IGNORECASE),
    ],
}


class ModerationPipeline:
    """Three-layer moderation guardrail pipeline."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        calibrator_path: str | Path | None = None,
        allow_threshold: float = 0.4,
        block_threshold: float = 0.6,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if allow_threshold >= block_threshold:
            raise ValueError("allow_threshold must be lower than block_threshold.")

        self.allow_threshold = float(allow_threshold)
        self.block_threshold = float(block_threshold)
        self.max_length = int(max_length)

        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR / "part4_best_model"
        if self.model_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        else:
            # Fallback to base model for development bootstrapping.
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.calibrator = None
        if calibrator_path:
            path = Path(calibrator_path)
            if path.exists():
                self.calibrator = joblib.load(path)

    @staticmethod
    def input_filter(text: str) -> Optional[Dict[str, object]]:
        """Returns a block decision dict if matched, else None."""
        for category, patterns in BLOCKLIST.items():
            for pattern in patterns:
                if pattern.search(text):
                    return {
                        "decision": "block",
                        "layer": "input_filter",
                        "category": category,
                        "confidence": 1.0,
                    }
        return None

    def _raw_probability(self, text: str) -> float:
        encoded = self.tokenizer(
            [text],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        return float(prob)

    def _calibrate(self, raw_prob: float) -> float:
        if self.calibrator is None:
            return raw_prob
        calibrated = self.calibrator.predict_proba(np.array([[raw_prob]]))[:, 1][0]
        return float(calibrated)

    def predict(self, text: str) -> Dict[str, object]:
        """Run layer 1->2->3 and return a structured moderation decision."""
        l1 = self.input_filter(text)
        if l1 is not None:
            return l1

        raw_prob = self._raw_probability(text)
        calibrated_prob = self._calibrate(raw_prob)

        if calibrated_prob >= self.block_threshold:
            return {
                "decision": "block",
                "layer": "model",
                "confidence": round(calibrated_prob, 6),
                "raw_confidence": round(raw_prob, 6),
            }

        if calibrated_prob <= self.allow_threshold:
            return {
                "decision": "allow",
                "layer": "model",
                "confidence": round(calibrated_prob, 6),
                "raw_confidence": round(raw_prob, 6),
            }

        return {
            "decision": "review",
            "layer": "model",
            "confidence": round(calibrated_prob, 6),
            "raw_confidence": round(raw_prob, 6),
        }
