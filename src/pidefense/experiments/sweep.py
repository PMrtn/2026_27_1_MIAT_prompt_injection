"""Security/usability trade-off of the ML layer over its decision threshold (counterfactual sweep).

The detector is scored offline on the tool outputs logged in the episodes of configuration 1 ('none') and
configuration 2 ('guardrails'). For a threshold t:

* ML only (config 3 at t): an attack succeeds iff it succeeded in config 1 AND the payload-carrying output scored
  below t (a withheld output cannot deliver the payload; an output that passes yields the same LLM request as
  config 1 and, under the exact-request cache, the same response).
* guardrails + ML (config 4 at t): same, starting from config 2 outcomes and scoring the R1-sanitised text.
* FPR(t): share of benign read-tool outputs (benign episodes) flagged, by the detector alone or by R2 OR the
  detector.

The estimate at the deployed threshold can be compared with the measured configs 3/4 as a consistency check.

Usage:
    python -m pidefense.experiments.sweep
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pidefense.defenses.guardrails import Sanitizer, SignatureFilter
from pidefense.detector.inference import InjectionDetector
from pidefense.experiments.analysis import _contains_payload, _payload_core, load_episodes
from pidefense.experiments.attacks import load_grid
from pidefense.experiments.runner import E2E_DIR, GRID_PATH
from pidefense.logging_utils import setup_logging
from pidefense.schemas import EpisodeResult, ToolContext

logger = logging.getLogger(__name__)

THRESHOLDS = np.concatenate([np.linspace(0.0, 0.9, 19), np.linspace(0.91, 0.999, 30), [0.9995, 0.9999, 1.0001]])


def _payload_outputs(episode: EpisodeResult, core: str) -> list[str]:
    return [
        tc.raw_output or "" for tc in episode.tool_calls if tc.kind == "read" and _contains_payload(tc.raw_output, core)
    ]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, default=E2E_DIR / "episodes.jsonl")
    args = parser.parse_args()
    setup_logging()

    payloads = {a.attack_id: a.payload for a in load_grid(GRID_PATH)}
    episodes = [e for e in load_episodes(args.episodes) if e.error is None and e.config in ("none", "guardrails")]
    detector = InjectionDetector()
    sanitizer, signatures = Sanitizer(), SignatureFilter()
    ctx = ToolContext(tool_name="sweep")

    def sanitized(text: str) -> str:
        return sanitizer.inspect(text, ctx).text or text

    attack_rows = []
    for episode in (e for e in episodes if e.attack_id):
        outputs = _payload_outputs(episode, _payload_core(payloads[episode.attack_id]))
        text_for_ml = [sanitized(o) for o in outputs] if episode.config == "guardrails" else outputs
        score = max(detector.score(text_for_ml), default=0.0)
        attack_rows.append({"config": episode.config, "success": bool(episode.attack_success), "score": score})
    benign_outputs = [
        tc.raw_output or ""
        for e in episodes
        if e.attack_id is None and e.config == "none"
        for tc in e.tool_calls
        if tc.kind == "read"
    ]
    benign_sanitized = [sanitized(o) for o in benign_outputs]
    benign_scores = np.asarray(detector.score(benign_outputs))
    benign_scores_san = np.asarray(detector.score(benign_sanitized))
    benign_r2 = np.asarray([not signatures.inspect(o, ctx).allowed for o in benign_sanitized])
    attacks = pd.DataFrame(attack_rows)

    rows = []
    for threshold in THRESHOLDS:
        for config, base, fpr in (
            ("ml", "none", float(np.mean(benign_scores >= threshold)) if benign_scores.size else np.nan),
            (
                "combined",
                "guardrails",
                float(np.mean(benign_r2 | (benign_scores_san >= threshold))) if benign_scores.size else np.nan,
            ),
        ):
            sub = attacks[attacks["config"] == base]
            asr = float(np.mean(sub["success"] & (sub["score"] < threshold))) if len(sub) else np.nan
            rows.append({"threshold": float(threshold), "config": config, "asr_est": asr, "tool_fpr_est": fpr})
    out = E2E_DIR / "threshold_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.5f")
    logger.info(
        "Wrote %s (%d attack episodes, %d benign outputs; deployed threshold %.4f)",
        out,
        len(attacks),
        len(benign_outputs),
        detector.threshold,
    )


if __name__ == "__main__":
    main()
