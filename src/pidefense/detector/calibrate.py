"""Re-calibrate the saved detector's decision threshold on validation data only (no test data involved).

Usage:
    python -m pidefense.detector.calibrate
"""

from __future__ import annotations

import json
import logging

from pidefense.detector.common import RESULTS_DIR, TARGET_FPR, binary_metrics, load_split, select_threshold, write_json
from pidefense.detector.inference import DEFAULT_MODEL_DIR, InjectionDetector
from pidefense.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Score validation, select the threshold, update threshold.json and training.json."""
    setup_logging()
    detector = InjectionDetector(DEFAULT_MODEL_DIR)
    val = load_split("val")
    scores = detector.score([s.text for s in val])
    labels = [s.label for s in val]
    threshold = select_threshold(labels, scores, TARGET_FPR)
    rule = "max F1 s.t. val FPR <= target, ties closest to 0.5"
    write_json(
        DEFAULT_MODEL_DIR / "threshold.json",
        {"threshold": threshold, "target_fpr": TARGET_FPR, "calibrated_on": "val", "rule": rule},
    )
    metrics = binary_metrics(labels, scores, threshold)
    training_path = RESULTS_DIR / "training.json"
    if training_path.exists():
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training.update({"threshold": threshold, "threshold_rule": rule, "val_metrics_at_threshold": metrics})
        write_json(training_path, training)
    logger.info("Threshold %.6f; val metrics %s", threshold, metrics)


if __name__ == "__main__":
    main()
