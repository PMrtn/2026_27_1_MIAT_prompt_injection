"""Estimated attack success per defended configuration, from the undefended LLM runs and the offline layer decisions.

The LLM runs of configuration 'none' tell which attacks the undefended agent complied with. The offline layer
evaluation (``pidefense.experiments.layer_eval``) tells, per attack, whether a defended configuration would withhold the
payload-carrying tool output (OpsPilot: at L0, and at L1 where the same output also carries the escalation follow-up)
and whether the action policy would veto the attacker's target action. Per attack and configuration:

* target action vetoed (guardrails, combined)  -> fails: the action is never executed
* carrier withheld at every level              -> fails: no attacker text ever reaches the model
* carrier withheld at no level, no veto        -> same outcome as undefended (identical prompt; R1 only normalises it)
* carrier withheld at one level only           -> unknown: a failure in the lower bound, the undefended outcome in
                                                  the upper bound

The estimate is checked against real defended runs on a small verification sample (``ops_runner --verify``).

Usage:
    python -m pidefense.experiments.counterfactual
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pidefense.config import REPO_ROOT
from pidefense.experiments.analysis import episode_frame, load_episodes
from pidefense.experiments.attacks import load_grid
from pidefense.experiments.layer_eval import OUT_DIR as LAYERS_DIR
from pidefense.experiments.ops_analysis import frame as ops_frame
from pidefense.experiments.ops_analysis import load as ops_load
from pidefense.experiments.ops_attacks import load_ops_attacks
from pidefense.experiments.ops_runner import ATTACKS_PATH, OPS_DIR
from pidefense.experiments.runner import E2E_DIR, GRID_PATH
from pidefense.logging_utils import setup_logging
from pidefense.schemas import SEEN_FAMILIES

logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "results" / "counterfactual"
DEFENDED = ("guardrails", "ml", "combined")
WITH_ACTION_POLICY = frozenset({"guardrails", "combined"})


def undefended(office_path: Path, ops_paths: list[Path]) -> pd.DataFrame:
    """One row per undefended attack episode: domain, prompt variant, attack id, family group, goal reached."""
    frames = []
    if office_path.exists():
        payloads = {attack.attack_id: attack.payload for attack in load_grid(GRID_PATH)}
        office = episode_frame(load_episodes(office_path), payloads)
        office = office[office["is_attack"] & office["error"].isna() & (office["config"] == "none")]
        frames.append(
            pd.DataFrame(
                {
                    "domain": "office",
                    "variant": "-",
                    "attack_id": office["attack_id"],
                    "family_group": office["family"].map(lambda f: "seen" if f in SEEN_FAMILIES else "unseen"),
                    "success_none": office["attack_success"].astype(bool),
                }
            )
        )
    existing = [path for path in ops_paths if path.exists()]
    if existing:
        payloads = {attack.attack_id: attack.payload for attack in load_ops_attacks(ATTACKS_PATH)}
        ops = ops_frame(ops_load(existing), payloads)
        ops = ops[ops["is_attack"] & ops["error"].isna() & (ops["config"] == "none")]
        frames.append(
            pd.DataFrame(
                {
                    "domain": "ops",
                    "variant": ops["variant"],
                    "attack_id": ops["attack_id"],
                    "family_group": ops["family_group"],
                    "success_none": ops["success_L1"].astype(bool),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def layer_decisions(path: Path) -> pd.DataFrame:
    """Per (domain, attack id): withheld at L0 / at every level, and action veto, for each available configuration."""
    layers = pd.read_csv(path)
    configs = [c for c in DEFENDED if f"blocked_{c}" in layers]
    grouped = layers.groupby(["domain", "attack_id"])
    out = pd.DataFrame({"r3_veto": grouped["r3_blocks_action"].first().astype(bool)})
    for config in configs:
        column = layers[f"blocked_{config}"].astype(bool)
        out[f"{config}_any"] = column.groupby([layers["domain"], layers["attack_id"]]).any()
        out[f"{config}_all"] = column.groupby([layers["domain"], layers["attack_id"]]).all()
    return out.reset_index()


def estimate(outcomes: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Per attack episode and configuration: lower / upper estimate of attack success."""
    merged = outcomes.merge(decisions, on=["domain", "attack_id"], how="left", validate="many_to_one")
    missing = int(merged["r3_veto"].isna().sum())
    if missing:
        raise ValueError(f"{missing} attack episodes have no layer decision; re-run layer_eval")
    rows: list[dict[str, Any]] = []
    configs = [c for c in DEFENDED if f"{c}_any" in merged]
    for record in merged.to_dict("records"):
        for config in configs:
            vetoed = config in WITH_ACTION_POLICY and bool(record["r3_veto"])
            withheld_all, withheld_any = bool(record[f"{config}_all"]), bool(record[f"{config}_any"])
            if vetoed or withheld_all:
                lower = upper = False
            elif not withheld_any:
                lower = upper = bool(record["success_none"])
            else:
                lower, upper = False, bool(record["success_none"])
            rows.append(
                {
                    "domain": record["domain"],
                    "variant": record["variant"],
                    "attack_id": record["attack_id"],
                    "family_group": record["family_group"],
                    "config": config,
                    "success_none": bool(record["success_none"]),
                    "vetoed": vetoed,
                    "withheld_all": withheld_all,
                    "withheld_any": withheld_any,
                    "success_lower": lower,
                    "success_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def summarize(outcomes: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """Measured undefended ASR and estimated [lower, upper] ASR per (domain, variant, configuration)."""
    rows = []
    for (domain, variant), sub in outcomes.groupby(["domain", "variant"]):
        rows.append(
            {
                "domain": domain,
                "variant": variant,
                "config": "none",
                "n": len(sub),
                "k_lower": int(sub["success_none"].sum()),
                "k_upper": int(sub["success_none"].sum()),
                "source": "measured",
            }
        )
        est = estimates[(estimates["domain"] == domain) & (estimates["variant"] == variant)]
        for config, group in est.groupby("config", sort=False):
            rows.append(
                {
                    "domain": domain,
                    "variant": variant,
                    "config": config,
                    "n": len(group),
                    "k_lower": int(group["success_lower"].sum()),
                    "k_upper": int(group["success_upper"].sum()),
                    "source": "estimated",
                }
            )
    table = pd.DataFrame(rows)
    table["asr_lower"] = table["k_lower"] / table["n"]
    table["asr_upper"] = table["k_upper"] / table["n"]
    return table


def main() -> None:
    """CLI entry point: writes results/counterfactual/{summary,per_attack}.csv."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--office-episodes", type=Path, default=E2E_DIR / "episodes.jsonl")
    parser.add_argument(
        "--ops-episodes",
        type=Path,
        nargs="+",
        default=[OPS_DIR / "sample_episodes.jsonl", OPS_DIR / "rest_episodes.jsonl"],
    )
    parser.add_argument("--layers", type=Path, default=LAYERS_DIR / "attack_outputs.csv")
    parser.add_argument("--tag", default="", help="suffix of the output files, e.g. base (with --layers ..._base.csv)")
    args = parser.parse_args()
    setup_logging()
    suffix = f"_{args.tag}" if args.tag else ""
    outcomes = undefended(args.office_episodes, args.ops_episodes)
    estimates = estimate(outcomes, layer_decisions(args.layers))
    summary = summarize(outcomes, estimates)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(OUT_DIR / f"per_attack{suffix}.csv", index=False)
    summary.to_csv(OUT_DIR / f"summary{suffix}.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200):
        logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
