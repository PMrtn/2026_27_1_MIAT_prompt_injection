"""OpsPilot attack set and out-of-domain set: coverage, diversity and leakage rules (offline)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from pidefense.data.carriers_claude import AuthoredCarrier
from pidefense.data.ops_ood import build_ood
from pidefense.data.templates import template_pools
from pidefense.experiments.ops_attacks import ALL_FAMILIES, FOLLOWUPS, OPS_FAMILIES, OPS_GOALS, build_ops_attacks
from pidefense.schemas import SEEN_FAMILIES


@dataclass
class _Task:
    applicable_goals: list[str] = field(default_factory=lambda: ["O1", "O2", "O4", "O5", "O8"])


def _tasks(n: int = 8) -> dict[str, _Task]:
    return {f"P{i:02d}": _Task() for i in range(1, n + 1)}


def test_every_task_gets_every_family_and_unique_ids() -> None:
    attacks = build_ops_attacks(_tasks(), per_task=24)
    assert len(attacks) == 8 * 24
    assert len({a.attack_id for a in attacks}) == len(attacks)
    for task_id in _tasks():
        families = {a.family for a in attacks if a.task_id == task_id}
        assert families == set(ALL_FAMILIES)


def test_goals_languages_and_followups_are_diverse() -> None:
    attacks = build_ops_attacks(_tasks(), per_task=24)
    assert {a.lang for a in attacks} == {"en", "hu"}
    assert {a.goal for a in attacks} == {"O1", "O2", "O4", "O5", "O8"}
    assert {a.followup_id for a in attacks} == set(FOLLOWUPS)
    assert len({a.payload for a in attacks}) > 0.95 * len(attacks)
    assert all("{" not in a.followup or '"' in a.followup for a in attacks)


def test_seen_families_use_test_pool_templates_only() -> None:
    pools = template_pools()
    for attack in build_ops_attacks(_tasks(), per_task=24):
        if attack.family in SEEN_FAMILIES:
            assert pools[attack.technique_id] == "test"


def test_every_goal_has_phrasings_in_both_languages() -> None:
    for goal, phrasings in OPS_GOALS.items():
        assert phrasings["en"] and phrasings["hu"], goal


def test_ood_set_is_balanced_over_families() -> None:
    carriers = [
        AuthoredCarrier(f"c{i}", "runbook", "en" if i % 2 else "hu", "t", i % 3 == 0, "Step one. Step two. Done.")
        for i in range(80)
    ]
    samples = build_ood(carriers, injected_per_carrier=2)
    counts = collections.Counter(s.family for s in samples if s.label == 1)
    assert set(counts) == set(ALL_FAMILIES)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert sum(1 for s in samples if s.label == 0) == 80
    assert {s.family for s in samples if s.label == 1} >= set(OPS_FAMILIES)
