"""Base vs focus detector on the OpsPilot tool outputs that matter (offline, no LLM).

Inputs are the layer evaluations of both checkpoints:

* ``python -m pidefense.experiments.layer_eval --with-ml --model-dir artifacts/detector_base/best --tag base``
* ``python -m pidefense.experiments.layer_eval --with-ml`` (focus model, the one used by configurations 3 and 4)

and the undefended OpsPilot runs, which tell which attacks succeeded. Output: ``results/layers/model_comparison.csv``.
For each slice and level it gives the share of payload-carrying outputs withheld by R2, by the base detector and by
the focus detector. For benign outputs it gives the share wrongly withheld.

Usage:
    python -m pidefense.experiments.detector_compare
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pidefense.experiments.layer_eval import OUT_DIR as LAYERS_DIR
from pidefense.experiments.ops_runner import OPS_DIR, successful_attacks
from pidefense.logging_utils import setup_logging

logger = logging.getLogger(__name__)

RUNBOOK_TASKS = frozenset({"P03", "P04"})
COLUMNS = {"r2": "r2_block", "ml_base": "blocked_ml_base", "ml_focus": "blocked_ml"}


def _merge(focus: pd.DataFrame, base: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Focus-model rows with the base model's ML verdict as ``blocked_ml_base``."""
    base = base[keys + ["blocked_ml", "ml_score"]].rename(
        columns={"blocked_ml": "blocked_ml_base", "ml_score": "ml_score_base"}
    )
    merged = focus.merge(base, on=keys, how="left", validate="one_to_one")
    if merged["blocked_ml_base"].isna().any():
        raise ValueError("base and focus layer evaluations do not cover the same outputs; re-run layer_eval")
    return merged


def attack_slices(attacks: pd.DataFrame, successes: set[str]) -> dict[str, pd.Series]:
    """Boolean masks of the attack slices."""
    ops = attacks["domain"] == "ops"
    runbook = attacks["task_id"].isin(RUNBOOK_TASKS)
    return {
        "office attacks": attacks["domain"] == "office",
        "OpsPilot attacks": ops,
        "OpsPilot runbook attacks (P03/P04)": ops & runbook,
        "OpsPilot other attacks": ops & ~runbook,
        "OpsPilot undefended successes": ops & attacks["attack_id"].isin(successes),
    }


def compare(attacks: pd.DataFrame, benign: pd.DataFrame, successes: set[str]) -> pd.DataFrame:
    """Withheld share per slice, level and layer."""
    rows = []
    for name, mask in attack_slices(attacks, successes).items():
        for level, sub in attacks[mask].groupby("level"):
            row = {"slice": name, "level": level, "n": len(sub)}
            row.update({layer: float(sub[column].astype(bool).mean()) for layer, column in COLUMNS.items()})
            rows.append(row)
    benign = benign.assign(task=benign["episode_id"].str.split("|").str[2])
    for name, mask in (
        ("benign office outputs (FPR)", benign["domain"] == "office"),
        ("benign OpsPilot outputs (FPR)", benign["domain"] == "ops"),
        ("benign OpsPilot runbooks (FPR)", (benign["domain"] == "ops") & (benign["tool"] == "read_runbook")),
    ):
        sub = benign[mask]
        row = {"slice": name, "level": "-", "n": len(sub)}
        row.update({layer: float(sub[column].astype(bool).mean()) for layer, column in COLUMNS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ops-episodes",
        type=Path,
        nargs="+",
        default=[OPS_DIR / "sample_episodes.jsonl", OPS_DIR / "rest_episodes.jsonl"],
    )
    args = parser.parse_args()
    setup_logging()
    attacks = _merge(
        pd.read_csv(LAYERS_DIR / "attack_outputs.csv"),
        pd.read_csv(LAYERS_DIR / "attack_outputs_base.csv"),
        ["domain", "attack_id", "level"],
    )
    benign_focus, benign_base = pd.read_csv(LAYERS_DIR / "benign_outputs.csv"), pd.read_csv(
        LAYERS_DIR / "benign_outputs_base.csv"
    )
    benign_focus["row"], benign_base["row"] = range(len(benign_focus)), range(len(benign_base))
    benign = _merge(benign_focus, benign_base, ["row", "episode_id", "tool"])
    successes = set().union(*(successful_attacks(path) for path in args.ops_episodes if path.exists()))
    table = compare(attacks, benign, successes)
    table.to_csv(LAYERS_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    per_success = attacks[attacks["attack_id"].isin(successes)][
        [
            "attack_id",
            "family",
            "goal",
            "level",
            "r2_block",
            "ml_score_base",
            "ml_score",
            "blocked_ml_base",
            "blocked_ml",
        ]
    ]
    per_success.to_csv(LAYERS_DIR / "successes_by_layer.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200):
        logger.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
