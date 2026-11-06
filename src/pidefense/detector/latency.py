"""Latency micro-benchmark of each defense layer on one tool output at a time (as in the agent loop).

Texts of increasing token length are built from test carriers. Each (layer, length) is timed ``--reps`` times after
warm-up; p50/p95 are written to ``results/detector/latency.csv``.

Usage:
    python -m pidefense.detector.latency [--reps 30]
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import time
from collections.abc import Callable

import torch

from pidefense.detector.baselines import RegexBaseline, TfidfBaseline
from pidefense.detector.common import RESULTS_DIR, load_split
from pidefense.detector.inference import InjectionDetector
from pidefense.logging_utils import setup_logging

logger = logging.getLogger(__name__)

LENGTHS = (64, 128, 256, 512, 1024, 2048)


def _texts_of_length(detector: InjectionDetector, corpus: list[str], target: int) -> str:
    """Concatenate corpus texts and cut to ``target`` model tokens."""
    joined = "\n\n".join(corpus)
    ids = detector.tokenizer(joined, add_special_tokens=False)["input_ids"][: target - 2]
    return detector.tokenizer.decode(ids)


def _time(fn: Callable[[], object], reps: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    setup_logging()

    corpus = [s.text for s in load_split("test") if s.label == 0][:200]
    gpu = InjectionDetector(device="cuda") if torch.cuda.is_available() else None
    cpu = InjectionDetector(device="cpu")
    regex = RegexBaseline()
    tfidf = TfidfBaseline.load()
    layers: dict[str, Callable[[str], object]] = {
        "regex_guardrail": lambda text: regex.score([text]),
        "tfidf_lr": lambda text: tfidf.score([text]),
        "distilbert_cpu": lambda text: cpu.score([text]),
    }
    if gpu is not None:
        layers["distilbert_gpu"] = lambda text: gpu.score([text])

    rows = []
    for length in LENGTHS:
        text = _texts_of_length(cpu, corpus, length)
        windows = cpu.windows(text)
        for name, fn in layers.items():
            reps = args.reps if not (name == "distilbert_cpu" and length >= 1024) else max(5, args.reps // 3)
            timings = _time(lambda fn=fn, text=text: fn(text), reps)
            row = {
                "layer": name,
                "tokens": length,
                "windows": windows,
                "p50_ms": round(statistics.median(timings), 3),
                "p95_ms": round(sorted(timings)[max(0, int(0.95 * len(timings)) - 1)], 3),
                "mean_ms": round(statistics.fmean(timings), 3),
                "reps": len(timings),
            }
            rows.append(row)
            logger.info("%s", row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", RESULTS_DIR / "latency.csv")


if __name__ == "__main__":
    main()
