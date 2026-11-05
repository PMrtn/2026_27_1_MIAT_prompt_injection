"""Shared helpers: dataset loading, metrics and threshold selection."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pidefense.config import REPO_ROOT
from pidefense.schemas import Sample

logger = logging.getLogger(__name__)

DATASET_DIR = REPO_ROOT / "data" / "dataset"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "detector"
RESULTS_DIR = REPO_ROOT / "results" / "detector"
MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
MAX_TOKENS = 256
STRIDE = 128
TARGET_FPR = 0.01


def load_split(split: str, dataset_dir: Path = DATASET_DIR) -> list[Sample]:
    """Read one split written by ``pidefense.data.build``."""
    with (dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [Sample.model_validate_json(line) for line in handle if line.strip()]


def threshold_for_fpr(benign_scores: Sequence[float], target_fpr: float = TARGET_FPR) -> float:
    """Smallest threshold whose false positive rate on the given benign scores is at most ``target_fpr``.

    A sample is flagged when ``score >= threshold``. Kept for the threshold sweep; model thresholds use
    ``select_threshold``.
    """
    scores = np.sort(np.asarray(benign_scores, dtype=float))
    if scores.size == 0:
        return 0.5
    allowed = int(np.floor(target_fpr * scores.size))
    # Flag strictly above the (n - allowed)-th smallest benign score.
    pivot = scores[scores.size - allowed - 1]
    return float(np.nextafter(pivot, np.inf))


def select_threshold(labels: Sequence[int], scores: Sequence[float], target_fpr: float = TARGET_FPR) -> float:
    """Validation threshold: maximise F1 subject to FPR <= ``target_fpr``; ties go to the threshold closest to 0.5.

    On separable validation data every threshold in the gap between the classes is optimal, and the tie-break keeps
    the natural 0.5 instead of a hair-trigger just above the highest benign score.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate([s, np.nextafter(s, np.inf), [0.5]]))
    best: tuple[float, float, float] | None = None  # (f1, -distance_to_0.5, threshold)
    for threshold in candidates:
        pred = s >= threshold
        fp = np.sum(pred & (y == 0))
        negatives = max(1, int(np.sum(y == 0)))
        if fp / negatives > target_fpr:
            continue
        tp = np.sum(pred & (y == 1))
        fn = np.sum(~pred & (y == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        key = (float(f1), -abs(float(threshold) - 0.5), float(threshold))
        if best is None or key > best:
            best = key
    return best[2] if best else 0.5


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, float]:
    """Precision, recall, F1, FPR at a threshold, plus threshold-free ROC-AUC / PR-AUC when both classes exist."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result: dict[str, float] = {
        "n": int(y.size),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }
    if 0 < np.sum(y == 1) < y.size:
        result["roc_auc"] = float(roc_auc_score(y, s))
        result["pr_auc"] = float(average_precision_score(y, s))
    return result


def write_json(path: Path, payload: object) -> None:
    """Pretty JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
