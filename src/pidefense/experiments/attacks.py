"""End-to-end attack grid: payloads from test-pool (seen) and held-out (unseen) templates with the fixed canaries.

Per task: every (family, language) combination once - 8 families x EN/HU = 16 variants - with goals cycled
over the task's applicable goals. Seen-family payloads use only *test-pool* templates, so neither the
detector (trained on train-pool templates) nor the dataset ever contained them. All goals use the canaries of
``pidefense.attacks.goals``, which never appear in the detector dataset.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from pidefense.data.build import make_payload
from pidefense.data.templates import E2E_GOALS, all_templates, fill, template_pools
from pidefense.experiments.scenarios import Scenario
from pidefense.schemas import SEEN_FAMILIES, UNSEEN_FAMILIES

logger = logging.getLogger(__name__)

LANGS = ("en", "hu")


@dataclass(frozen=True)
class AttackSpec:
    """One concrete attack payload for one task."""

    attack_id: str
    task_id: str
    family: str
    lang: str
    goal: str
    template_id: str
    payload: str

    @property
    def seen(self) -> bool:
        """Whether the attack family was represented in detector training."""
        return self.family in SEEN_FAMILIES


def build_attack_grid(scenarios: dict[str, Scenario], seed: int = 2026) -> list[AttackSpec]:
    """Deterministic grid of 16 attacks per task."""
    templates = all_templates()
    pools = template_pools(templates)
    candidates = {
        (family, lang): [
            t
            for t in templates
            if t.family == family and t.lang == lang and (family in UNSEEN_FAMILIES or pools[t.id] == "test")
        ]
        for family in SEEN_FAMILIES + UNSEEN_FAMILIES
        for lang in LANGS
    }
    grid: list[AttackSpec] = []
    for task_index, task_id in enumerate(sorted(scenarios)):
        scenario = scenarios[task_id]
        rng = random.Random(f"{seed}-{task_id}")
        goals = sorted(scenario.applicable_goals)
        owner = str(scenario.environment.get("owner_name", "Alex Morgan"))
        combos = [(family, lang) for family in SEEN_FAMILIES + UNSEEN_FAMILIES for lang in LANGS]
        for index, (family, lang) in enumerate(combos):
            goal = goals[(index + task_index) % len(goals)]
            template = rng.choice(candidates[(family, lang)])
            goal_text = fill(
                rng.choice(E2E_GOALS[goal][lang]),
                file=scenario.goal_targets.G3_file,
                colleague=scenario.goal_targets.G5_recipient,
            )
            payload = make_payload(template, goal_text, owner, rng).strip("\n")
            grid.append(
                AttackSpec(
                    attack_id=f"{task_id}-{family}-{lang}-{goal}",
                    task_id=task_id,
                    family=family,
                    lang=lang,
                    goal=goal,
                    template_id=template.id,
                    payload=payload,
                )
            )
    logger.info("Attack grid: %d attacks over %d tasks", len(grid), len(scenarios))
    return grid


def save_grid(grid: list[AttackSpec], path: Path) -> None:
    """Write the grid as JSONL (committed before the full run: light pre-registration)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for attack in grid:
            handle.write(json.dumps(asdict(attack), ensure_ascii=False) + "\n")


def load_grid(path: Path) -> list[AttackSpec]:
    """Read a grid written by ``save_grid``."""
    with path.open(encoding="utf-8") as handle:
        return [AttackSpec(**json.loads(line)) for line in handle if line.strip()]
