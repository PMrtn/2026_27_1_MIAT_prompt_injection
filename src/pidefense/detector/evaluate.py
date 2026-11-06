"""Classifier-level evaluation of DistilBERT vs. TF-IDF+LR vs. the regex guardrail.

Thresholds are fixed on validation only (FPR <= 1% for score-based models). Slices of the test split:

* ``test_seen``   - benign test + injected seen families (S1-S5, new templates and carriers)
* ``test_unseen`` - benign test + injected held-out families (U1-U3), plus one slice per family
* ``lang_en`` / ``lang_hu`` - all test samples of one language
* ``hard_negatives`` - benign instruction-like texts only (FPR)
* ``external`` - deepset/prompt-injections (different distribution; sanity check only)
* ``ood_*``      - the OpsPilot-domain set (``pidefense.data.ops_ood``)
* ``focus_*``    - the test split of the focus component (``pidefense.data.focus``): delegated documents with injected
  steps (D1) and escalation follow-ups (D2), against legitimate documents, inserted steps and follow-ups

Usage:
    python -m pidefense.detector.evaluate [--extra-dataset data/focus] [--model-dir DIR] [--tag base]

``--extra-dataset`` adds that component's train/val splits to the TF-IDF fit (match it to the DistilBERT run).
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pidefense.config import REPO_ROOT
from pidefense.data.external import load_external
from pidefense.detector.baselines import RegexBaseline, TfidfBaseline
from pidefense.detector.common import (
    DATASET_DIR,
    RESULTS_DIR,
    TARGET_FPR,
    binary_metrics,
    load_split,
    select_threshold,
    write_json,
)
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


def _focus_slices(focus: list[Sample]) -> dict[str, list[int]]:
    benign = [i for i, s in enumerate(focus) if s.label == 0]
    return {
        "focus_all": list(range(len(focus))),
        "focus_D1": benign + [i for i, s in enumerate(focus) if s.family == "D1"],
        "focus_D2": benign + [i for i, s in enumerate(focus) if s.family == "D2"],
        "focus_hard_negatives": benign,
        "focus_benign_inserts": [i for i in benign if focus[i].position in ("slot", "appended")],
        "focus_en": [i for i, s in enumerate(focus) if s.lang == "en"],
        "focus_hu": [i for i, s in enumerate(focus) if s.lang == "hu"],
    }


def _score_slices(
    scorer: Scorer, threshold: float, samples: list[Sample], slices: dict[str, list[int]]
) -> tuple[list[float], dict[str, dict[str, object]]]:
    """Scores of every sample and the metrics of every slice."""
    scores = scorer([s.text for s in samples])
    metrics = {
        name: binary_metrics([samples[i].label for i in indices], [scores[i] for i in indices], threshold)
        for name, indices in slices.items()
    }
    return scores, metrics


def _write_summary(path: Path, results: dict[str, dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
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


def _write_scores(path: Path, samples: list[Sample], columns: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "label", "lang", "family", "goal", "hard_negative", "carrier_type", *columns])
        for index, sample in enumerate(samples):
            writer.writerow(
                [sample.id, sample.label, sample.lang, sample.family, sample.goal, int(sample.hard_negative)]
                + [sample.carrier_type]
                + [f"{column[index]:.6f}" for column in columns.values()]
            )


def _models(
    train: list[Sample], val: list[Sample], baselines_only: bool, model_dir: Path | None = None
) -> dict[str, tuple[Scorer, float]]:
    """Scorer and decision threshold per classifier; TF-IDF is refit and its threshold chosen on validation."""
    tfidf = TfidfBaseline().fit(train)
    tfidf.threshold = select_threshold([s.label for s in val], tfidf.score([s.text for s in val]), TARGET_FPR)
    tfidf.save()
    models: dict[str, tuple[Scorer, float]] = {
        "regex_guardrail": (RegexBaseline().score, 0.5),
        "tfidf_lr": (tfidf.score, tfidf.threshold),
    }
    if not baselines_only:
        detector = InjectionDetector(model_dir) if model_dir else InjectionDetector()
        models["distilbert"] = (detector.score, detector.threshold)
    return models


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument(
        "--baselines-only", action="store_true", help="evaluate only the regex and TF-IDF baselines (no checkpoint)"
    )
    parser.add_argument("--extra-dataset", type=Path, action="append", default=[])
    parser.add_argument("--model-dir", type=Path, default=None, help="DistilBERT checkpoint (default: artifacts/...)")
    parser.add_argument("--tag", default="", help="suffix of the output files, e.g. base")
    args = parser.parse_args()
    setup_logging()
    suffix = f"_{args.tag}" if args.tag else ""

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    for directory in args.extra_dataset:
        train, val = train + load_split("train", directory), val + load_split("val", directory)
    external = [] if args.skip_external else load_external()

    models = _models(train, val, args.baselines_only, args.model_dir)

    ood = _load_ood()
    focus_dir = DATASET_DIR.parent / "focus"
    focus = load_split("test", focus_dir) if (focus_dir / "test.jsonl").exists() else []
    ood_column: dict[str, list[float]] = {}
    focus_column: dict[str, list[float]] = {}
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
            ood_column[name], ood_metrics = _score_slices(scorer, threshold, ood, _ood_slices(ood))
            model_result.update(ood_metrics)
        if focus:
            focus_column[name], focus_metrics = _score_slices(scorer, threshold, focus, _focus_slices(focus))
            model_result.update(focus_metrics)
        results[name] = model_result
        logger.info(
            "%s: seen F1 %.3f | unseen recall %.3f | hard-neg FPR %.3f",
            name,
            model_result["test_seen"]["f1"],
            model_result["test_unseen"]["recall"],
            model_result["hard_negatives"]["fpr"],
        )

    write_json(RESULTS_DIR / f"metrics{suffix}.json", results)
    _write_summary(RESULTS_DIR / f"summary{suffix}.csv", results)
    _write_scores(RESULTS_DIR / f"test_scores{suffix}.csv", test, score_columns)
    if ood:
        _write_scores(RESULTS_DIR / f"ood_ops_scores{suffix}.csv", ood, ood_column)
    if focus:
        _write_scores(RESULTS_DIR / f"focus_scores{suffix}.csv", focus, focus_column)
    logger.info("Wrote %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
