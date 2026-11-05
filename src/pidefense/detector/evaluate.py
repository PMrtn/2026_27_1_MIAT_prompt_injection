"""Classifier-level evaluation of DistilBERT vs. TF-IDF+LR vs. the regex guardrail.

Thresholds are fixed on validation only (FPR <= 1% for score-based models). Slices of the test split:

* ``test_seen``   - benign test + injected seen families (S1-S5, new templates and carriers)
* ``test_unseen`` - benign test + injected held-out families (U1-U3), plus one slice per family
* ``lang_en`` / ``lang_hu`` - all test samples of one language
* ``hard_negatives`` - benign instruction-like texts only (FPR)
* ``external`` - deepset/prompt-injections (different distribution; sanity check only)

Usage:
    python -m pidefense.detector.evaluate
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections.abc import Callable, Sequence

from pidefense.config import REPO_ROOT
from pidefense.data.external import load_external
from pidefense.detector.baselines import RegexBaseline, TfidfBaseline
from pidefense.detector.common import RESULTS_DIR, TARGET_FPR, binary_metrics, load_split, select_threshold, write_json
from pidefense.detector.inference import InjectionDetector
from pidefense.logging_utils import setup_logging
from pidefense.schemas import SEEN_FAMILIES, UNSEEN_FAMILIES, Sample

logger = logging.getLogger(__name__)

Scorer = Callable[[Sequence[str]], list[float]]


def _slices(test: list[Sample]) -> dict[str, list[int]]:
    benign = [i for i, s in enumerate(test) if s.label == 0]
    slices: dict[str, list[int]] = {
        "test_all": list(range(len(test))),
        "test_seen": benign + [i for i, s in enumerate(test) if s.family in SEEN_FAMILIES],
        "test_unseen": benign + [i for i, s in enumerate(test) if s.family in UNSEEN_FAMILIES],
        "lang_en": [i for i, s in enumerate(test) if s.lang == "en"],
        "lang_hu": [i for i, s in enumerate(test) if s.lang == "hu"],
        "hard_negatives": [i for i, s in enumerate(test) if s.label == 0 and s.hard_negative],
    }
    for family in SEEN_FAMILIES + UNSEEN_FAMILIES:
        slices[f"family_{family}"] = benign + [i for i, s in enumerate(test) if s.family == family]
    return slices


def _timed(scorer: Scorer, texts: Sequence[str]) -> tuple[list[float], float]:
    start = time.perf_counter()
    scores = scorer(texts)
    return scores, (time.perf_counter() - start) * 1000 / max(1, len(texts))


def _load_ood() -> list[Sample]:
    """The OpsPilot out-of-domain set (pidefense.data.ops_ood), if built."""
    path = REPO_ROOT / "data" / "ood_ops" / "ood_ops.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [Sample.model_validate_json(line) for line in handle if line.strip()]


def _ood_slices(ood: list[Sample]) -> dict[str, list[int]]:
    benign = [i for i, s in enumerate(ood) if s.label == 0]
    slices = {
        "ood_ops_all": list(range(len(ood))),
        "ood_ops_hard_negatives": [i for i, s in enumerate(ood) if s.label == 0 and s.hard_negative],
        "ood_ops_en": [i for i, s in enumerate(ood) if s.lang == "en"],
        "ood_ops_hu": [i for i, s in enumerate(ood) if s.lang == "hu"],
    }
    for family in sorted({s.family for s in ood if s.label == 1}):
        slices[f"ood_family_{family}"] = benign + [i for i, s in enumerate(ood) if s.family == family]
    return slices


def _models(train: list[Sample], val: list[Sample], baselines_only: bool) -> dict[str, tuple[Scorer, float]]:
    """Scorer and decision threshold per classifier; TF-IDF is refit and its threshold chosen on validation."""
    tfidf = TfidfBaseline().fit(train)
    tfidf.threshold = select_threshold([s.label for s in val], tfidf.score([s.text for s in val]), TARGET_FPR)
    tfidf.save()
    models: dict[str, tuple[Scorer, float]] = {
        "regex_guardrail": (RegexBaseline().score, 0.5),
        "tfidf_lr": (tfidf.score, tfidf.threshold),
    }
    if not baselines_only:
        detector = InjectionDetector()
        models["distilbert"] = (detector.score, detector.threshold)
    return models


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument(
        "--baselines-only", action="store_true", help="evaluate only the regex and TF-IDF baselines (no checkpoint)"
    )
    args = parser.parse_args()
    setup_logging()

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    external = [] if args.skip_external else load_external()

    models = _models(train, val, args.baselines_only)

    ood = _load_ood()
    ood_column: dict[str, list[float]] = {}
    slices = _slices(test)
    labels = [s.label for s in test]
    results: dict[str, dict[str, object]] = {}
    score_columns: dict[str, list[float]] = {}
    for name, (scorer, threshold) in models.items():
        scores, ms_per_sample = _timed(scorer, [s.text for s in test])
        score_columns[name] = scores
        model_result: dict[str, object] = {"threshold": threshold, "ms_per_sample_batch": ms_per_sample}
        for slice_name, indices in slices.items():
            model_result[slice_name] = binary_metrics(
                [labels[i] for i in indices], [scores[i] for i in indices], threshold
            )
        val_scores = scorer([s.text for s in val])
        model_result["val"] = binary_metrics([s.label for s in val], val_scores, threshold)
        if external:
            ext_scores = scorer([str(row["text"]) for row in external])
            model_result["external"] = binary_metrics([int(row["label"]) for row in external], ext_scores, threshold)
        if ood:
            ood_scores = scorer([s.text for s in ood])
            ood_column[name] = ood_scores
            for slice_name, indices in _ood_slices(ood).items():
                model_result[slice_name] = binary_metrics(
                    [ood[i].label for i in indices], [ood_scores[i] for i in indices], threshold
                )
        results[name] = model_result
        logger.info(
            "%s: seen F1 %.3f | unseen recall %.3f | hard-neg FPR %.3f",
            name,
            model_result["test_seen"]["f1"],
            model_result["test_unseen"]["recall"],
            model_result["hard_negatives"]["fpr"],
        )

    write_json(RESULTS_DIR / "metrics.json", results)
    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "slice", "n", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"])
        for name, model_result in results.items():
            for slice_name, metrics in model_result.items():
                if not isinstance(metrics, dict):
                    continue
                writer.writerow(
                    [name, slice_name, metrics["n"]]
                    + [
                        f"{metrics[key]:.4f}" if key in metrics else ""
                        for key in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")
                    ]
                )
    with (RESULTS_DIR / "test_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "label", "lang", "family", "hard_negative", "carrier_type", *score_columns])
        for index, sample in enumerate(test):
            writer.writerow(
                [sample.id, sample.label, sample.lang, sample.family, int(sample.hard_negative), sample.carrier_type]
                + [f"{score_columns[name][index]:.6f}" for name in score_columns]
            )
    if ood:
        with (RESULTS_DIR / "ood_ops_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id", "label", "lang", "family", "goal", "hard_negative", "carrier_type", *ood_column])
            for index, sample in enumerate(ood):
                writer.writerow(
                    [sample.id, sample.label, sample.lang, sample.family, sample.goal, int(sample.hard_negative)]
                    + [sample.carrier_type]
                    + [f"{ood_column[name][index]:.6f}" for name in ood_column]
                )
    logger.info("Wrote %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
