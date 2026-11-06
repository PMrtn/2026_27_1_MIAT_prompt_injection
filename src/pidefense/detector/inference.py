"""Sliding-window inference of the fine-tuned detector and its OutputDefense wrapper (the ML layer)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pidefense.defenses.base import OutputDefense
from pidefense.detector.common import ARTIFACT_DIR, MAX_TOKENS, STRIDE
from pidefense.schemas import DefenseDecision, ToolContext

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = ARTIFACT_DIR / "best"


class InjectionDetector:
    """Scores texts with P(injection); long texts are split into overlapping windows and the max is taken."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        device: str | None = None,
        max_tokens: int = MAX_TOKENS,
        stride: int = STRIDE,
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()
        self.max_tokens = max_tokens
        self.stride = stride
        self.batch_size = batch_size
        threshold_file = model_dir / "threshold.json"
        self.threshold = json.loads(threshold_file.read_text())["threshold"] if threshold_file.exists() else 0.5
        logger.info("Detector loaded from %s on %s (threshold %.4f)", model_dir, self.device, self.threshold)

    @torch.inference_mode()
    def score(self, texts: Sequence[str]) -> list[float]:
        """P(injection) per text = max over its windows."""
        if not texts:
            return []
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_tokens,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        mapping = encoded.pop("overflow_to_sample_mapping").tolist()
        probabilities: list[float] = []
        for start in range(0, len(mapping), self.batch_size):
            batch = {key: value[start : start + self.batch_size].to(self.device) for key, value in encoded.items()}
            logits = self.model(**batch).logits.float()
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
        best = [0.0] * len(texts)
        for window, sample_index in enumerate(mapping):
            best[sample_index] = max(best[sample_index], probabilities[window])
        return best

    def windows(self, text: str) -> int:
        """Number of windows a text is split into (for latency analysis)."""
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_tokens, stride=self.stride, return_overflowing_tokens=True
        )
        return len(encoded["input_ids"])


class MLDetectorDefense(OutputDefense):
    """Blocks a tool output when the detector score reaches the validation-calibrated threshold."""

    name = "ml_detector"

    def __init__(self, detector: InjectionDetector, threshold: float | None = None) -> None:
        self.detector = detector
        self.threshold = detector.threshold if threshold is None else threshold

    def inspect(self, text: str, ctx: ToolContext) -> DefenseDecision:
        score = self.detector.score([text])[0]
        blocked = score >= self.threshold
        return DefenseDecision(
            layer=self.name,
            allowed=not blocked,
            score=score,
            reason=f"injection probability {score:.3f} >= {self.threshold:.3f}" if blocked else "",
        )
