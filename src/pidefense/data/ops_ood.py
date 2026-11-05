"""Out-of-domain classifier test set from the OpsPilot domain (evaluation only, never training).

Benign carriers are ops-style tool outputs authored by Claude (``pidefense.data.carriers_claude``, ops kinds):
merchant support tickets, runbooks, log excerpts, monitoring alerts, CRM notes, vendor status pages, config files,
change requests and incident chat. Runbooks, configs, change requests and incident chat read like instructions by
nature, so they are hard negatives. Every carrier yields one benign sample and two injected samples drawn from all
16 attack families (S1-S5, U1-U3, P1-P8) x 8 ops goals x EN/HU. No LLM is called.

Usage:
    python -m pidefense.data.ops_ood
"""

from __future__ import annotations

import collections
import json
import logging
import random
from collections.abc import Sequence

from pidefense.config import REPO_ROOT
from pidefense.data.build import make_payload
from pidefense.data.carriers_claude import AuthoredCarrier, load_authored
from pidefense.data.embed import choose_position_and_wrapper, embed
from pidefense.data.templates import all_templates, template_pools
from pidefense.experiments.ops_attacks import ALL_FAMILIES, OPS_FAMILIES, OPS_GOALS, _ops_payload
from pidefense.logging_utils import setup_logging
from pidefense.schemas import UNSEEN_FAMILIES, Sample

logger = logging.getLogger(__name__)

OOD_DIR = REPO_ROOT / "data" / "ood_ops"


def build_ood(carriers: Sequence[AuthoredCarrier], seed: int = 99, injected_per_carrier: int = 2) -> list[Sample]:
    """One benign + ``injected_per_carrier`` injected samples per carrier, families cycled for balance."""
    rng = random.Random(seed)
    templates = all_templates()
    pools = template_pools(templates)
    dataset_candidates = collections.defaultdict(list)
    for template in templates:
        if template.family in UNSEEN_FAMILIES or pools[template.id] == "test":
            dataset_candidates[(template.family, template.lang)].append(template)
    samples: list[Sample] = []
    counter = 0
    for carrier in sorted(carriers, key=lambda c: c.carrier_id):
        samples.append(
            Sample(
                id=f"{carrier.carrier_id}-benign",
                text=carrier.text,
                label=0,
                lang=carrier.lang,
                carrier_type=carrier.carrier_type,
                carrier_id=carrier.carrier_id,
                split="ood_ops",
                hard_negative=carrier.hard_negative,
                generator=carrier.generator,
            )
        )
        for k in range(injected_per_carrier):
            family = ALL_FAMILIES[counter % len(ALL_FAMILIES)]
            counter += 1
            lang = carrier.lang if rng.random() > 0.1 else ("hu" if carrier.lang == "en" else "en")
            goal = rng.choice(sorted(OPS_GOALS))
            goal_text = rng.choice(OPS_GOALS[goal][lang])
            if family in OPS_FAMILIES:
                payload, template_id = _ops_payload(family, lang, goal_text, rng)
            else:
                template = rng.choice(dataset_candidates[(family, lang)])
                payload, template_id = make_payload(template, goal_text, "Gábor Kiss", rng).strip("\n"), template.id
            position, wrapper = choose_position_and_wrapper(rng, family)
            samples.append(
                Sample(
                    id=f"{carrier.carrier_id}-inj{k}",
                    text=embed(carrier.text, payload, position, wrapper, rng),
                    label=1,
                    lang=lang,
                    family=family,
                    goal=goal,
                    carrier_type=carrier.carrier_type,
                    position=position,
                    carrier_id=carrier.carrier_id,
                    template_id=template_id,
                    split="ood_ops",
                    hard_negative=carrier.hard_negative,
                    generator=f"template+{carrier.generator}",
                )
            )
    return samples


def main() -> None:
    """CLI entry point."""
    setup_logging()
    samples = build_ood(load_authored(ops=True))
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    with (OOD_DIR / "ood_ops.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")
    stats = {
        "total": len(samples),
        "by_label": dict(collections.Counter(s.label for s in samples)),
        "by_lang": dict(collections.Counter(s.lang for s in samples)),
        "by_family": dict(sorted(collections.Counter(s.family for s in samples).items())),
        "by_carrier_type": dict(sorted(collections.Counter(s.carrier_type for s in samples).items())),
        "hard_negative_benign": sum(1 for s in samples if s.label == 0 and s.hard_negative),
    }
    (OOD_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("OOD ops set: %s", stats)


if __name__ == "__main__":
    main()
