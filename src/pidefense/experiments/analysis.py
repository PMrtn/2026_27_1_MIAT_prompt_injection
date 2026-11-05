"""Metrics, confidence intervals and paired tests for the end-to-end episodes.

Definitions (attack episodes unless stated):

* ASR            - share of attack episodes whose goal was achieved (executed action / phishing link in answer)
* hijack rate    - share where the LLM *attempted* the attacker action (even if R3 then blocked it)
* delivery rate  - share where the payload-carrying tool output reached the LLM without being withheld
* utility        - share of episodes whose user task was completed (benign episodes / under attack)
* tool FPR       - benign episodes: share of read-tool outputs withheld by a defense
* episode FPR    - benign episodes: share with at least one withheld output or blocked action

Usage:
    python -m pidefense.experiments.analysis [--episodes results/e2e/episodes.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

from pidefense.experiments.attacks import load_grid
from pidefense.experiments.runner import E2E_DIR, GRID_PATH
from pidefense.logging_utils import setup_logging
from pidefense.schemas import SEEN_FAMILIES, EpisodeResult

logger = logging.getLogger(__name__)

CONFIG_ORDER = ["none", "guardrails", "ml", "combined"]
_STRIP = re.compile(r"[\s\\\"']")


def load_episodes(path: Path) -> list[EpisodeResult]:
    """Read episode JSONL; the last record per episode_id wins (re-runs after errors)."""
    episodes: dict[str, EpisodeResult] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result = EpisodeResult.model_validate_json(line)
                episodes[result.episode_id] = result
    return list(episodes.values())


def _payload_core(payload: str) -> str:
    return _STRIP.sub("", payload)[:40]


def _contains_payload(output: str | None, core: str) -> bool:
    return bool(output) and core in _STRIP.sub("", output or "")


def episode_frame(episodes: Iterable[EpisodeResult], payloads: dict[str, str]) -> pd.DataFrame:
    """One row per episode with derived outcome, defense and latency columns."""
    rows = []
    for ep in episodes:
        is_attack = ep.attack_id is not None
        core = _payload_core(payloads.get(ep.attack_id or "", "")) if is_attack else ""
        reads = [tc for tc in ep.tool_calls if tc.kind == "read"]
        carrier_reads = [tc for tc in reads if is_attack and _contains_payload(tc.raw_output, core)]
        withheld_reads = [tc for tc in reads if tc.blocked_by]
        blocked_actions = [tc for tc in ep.tool_calls if tc.kind == "action" and tc.blocked_by]
        rows.append(
            {
                "episode_id": ep.episode_id,
                "config": ep.config,
                "task_id": ep.task_id,
                "attack_id": ep.attack_id,
                "is_attack": is_attack,
                "family": ep.family,
                "seen": (ep.family in SEEN_FAMILIES) if is_attack else None,
                "lang": ep.lang,
                "goal": ep.goal,
                "repetition": ep.repetition,
                "utility": bool(ep.utility_success),
                "attack_success": bool(ep.attack_success) if is_attack else None,
                "hijack": bool(ep.hijack_attempted) if is_attack else None,
                "payload_read": bool(carrier_reads) if is_attack else None,
                "delivered": any(not tc.blocked_by for tc in carrier_reads) if is_attack else None,
                "n_reads": len(reads),
                "n_withheld": len(withheld_reads),
                "n_action_blocks": len(blocked_actions),
                "withheld_by": ",".join(sorted({str(tc.blocked_by) for tc in withheld_reads})),
                "provider_block": ep.blocked_by_provider,
                "error": ep.error,
                "steps": ep.steps,
                "total_latency_ms": ep.total_latency_ms,
                "defense_latency_ms": ep.defense_latency_ms,
                "llm_latency_ms": sum(call.latency_ms for call in ep.llm_calls),
                "llm_calls": len(ep.llm_calls),
                "cost_usd": sum(call.cost_usd for call in ep.llm_calls),
                "read_defense_ms": [tc.defense_latency_ms for tc in reads],
            }
        )
    return pd.DataFrame(rows)


def rate(values: pd.Series) -> dict[str, float]:
    """Share of True with a Wilson 95% CI."""
    values = values.dropna().astype(bool)
    n, k = int(values.size), int(values.sum())
    if n == 0:
        return {"k": 0, "n": 0, "rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
    return {"k": k, "n": n, "rate": k / n, "ci_low": float(low), "ci_high": float(high)}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Headline metrics per configuration."""
    valid = frame[frame["error"].isna()]
    rows = []
    for config in CONFIG_ORDER:
        sub = valid[valid["config"] == config]
        if sub.empty:
            continue
        attack, benign = sub[sub["is_attack"]], sub[~sub["is_attack"]]
        benign_reads = int(benign["n_reads"].sum())
        read_ms = np.concatenate([np.asarray(v, dtype=float) for v in sub["read_defense_ms"] if len(v)] or [[0.0]])
        row: dict[str, object] = {
            "config": config,
            "episodes": len(sub),
            "errors": int(frame[frame["config"] == config]["error"].notna().sum()),
        }
        for name, series in (
            ("asr", attack["attack_success"]),
            ("hijack", attack["hijack"]),
            ("delivery", attack["delivered"]),
            ("utility_benign", benign["utility"]),
            ("utility_attack", attack["utility"]),
            ("episode_fpr", (benign["n_withheld"] + benign["n_action_blocks"]) > 0),
            (
                "asr_seen",
                attack[attack["seen"] == True]["attack_success"],
            ),  # noqa: E712 pylint: disable=singleton-comparison
            (
                "asr_unseen",
                attack[attack["seen"] == False]["attack_success"],
            ),  # noqa: E712 pylint: disable=singleton-comparison
            ("asr_en", attack[attack["lang"] == "en"]["attack_success"]),
            ("asr_hu", attack[attack["lang"] == "hu"]["attack_success"]),
        ):
            stats = rate(series)
            row[name] = stats["rate"]
            row[f"{name}_ci"] = f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]"
            row[f"{name}_n"] = stats["n"]
        row["tool_fpr"] = int(benign["n_withheld"].sum()) / benign_reads if benign_reads else float("nan")
        row["tool_fpr_n"] = benign_reads
        row["defense_ms_per_read_p50"] = float(np.percentile(read_ms, 50))
        row["defense_ms_per_read_p95"] = float(np.percentile(read_ms, 95))
        row["episode_ms_p50"] = float(sub["total_latency_ms"].median())
        row["llm_ms_per_episode_p50"] = float(sub["llm_latency_ms"].median())
        row["defense_share_of_episode"] = float(
            sub["defense_latency_ms"].sum() / max(1.0, sub["total_latency_ms"].sum())
        )
        row["provider_blocks"] = int(sub["provider_block"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def breakdown(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """ASR per configuration and a grouping column (family, lang, goal, task_id)."""
    attack = frame[frame["is_attack"] & frame["error"].isna()]
    table = attack.pivot_table(index=by, columns="config", values="attack_success", aggfunc="mean")
    counts = attack[attack["config"] == "none"].groupby(by).size().rename("n")
    return table.reindex(columns=[c for c in CONFIG_ORDER if c in table.columns]).join(counts)


def mcnemar(frame: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    """Exact McNemar test on attack success, paired by attack_id."""
    attack = frame[frame["is_attack"] & frame["error"].isna()]
    pivot = attack.pivot_table(index="attack_id", columns="config", values="attack_success", aggfunc="first").dropna()
    if a not in pivot or b not in pivot:
        return {"a": a, "b": b, "n_pairs": 0}
    only_a = int(((pivot[a] == True) & (pivot[b] == False)).sum())  # noqa: E712 pylint: disable=singleton-comparison
    only_b = int(((pivot[a] == False) & (pivot[b] == True)).sum())  # noqa: E712 pylint: disable=singleton-comparison
    discordant = only_a + only_b
    p_value = binomtest(min(only_a, only_b), discordant, 0.5).pvalue if discordant else 1.0
    return {"a": a, "b": b, "n_pairs": int(len(pivot)), "a_only": only_a, "b_only": only_b, "p_value": float(p_value)}


def main() -> None:
    """CLI entry point: writes summary tables to results/e2e/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, default=E2E_DIR / "episodes.jsonl")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    setup_logging()

    payloads = {attack.attack_id: attack.payload for attack in load_grid(GRID_PATH)}
    frame = episode_frame(load_episodes(args.episodes), payloads)
    suffix = f"_{args.tag}" if args.tag else ""
    summary = summarize(frame)
    summary.to_csv(E2E_DIR / f"summary{suffix}.csv", index=False, float_format="%.4f")
    for by in ("family", "lang", "goal", "task_id"):
        breakdown(frame, by).to_csv(E2E_DIR / f"asr_by_{by}{suffix}.csv", float_format="%.4f")
    tests = [
        mcnemar(frame, a, b)
        for a, b in (
            ("none", "guardrails"),
            ("none", "ml"),
            ("guardrails", "combined"),
            ("ml", "combined"),
            ("guardrails", "ml"),
        )
    ]
    (E2E_DIR / f"mcnemar{suffix}.json").write_text(json.dumps(tests, indent=2) + "\n", encoding="utf-8")
    frame.drop(columns=["read_defense_ms"]).to_csv(E2E_DIR / f"episodes_flat{suffix}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        logger.info(
            "\n%s",
            summary[
                [
                    "config",
                    "episodes",
                    "errors",
                    "asr",
                    "asr_ci",
                    "hijack",
                    "delivery",
                    "utility_benign",
                    "utility_attack",
                    "tool_fpr",
                    "episode_fpr",
                    "defense_ms_per_read_p50",
                    "episode_ms_p50",
                ]
            ].to_string(index=False),
        )
        logger.info("\n%s", breakdown(frame, "family").to_string())
    logger.info("McNemar: %s", tests)


if __name__ == "__main__":
    main()
