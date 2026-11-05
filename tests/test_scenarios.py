"""Scenario files: loading, the single injection slot, targets and the intended task mix."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pidefense.attacks.goals import normalize_iban
from pidefense.experiments.scenarios import (
    PLACEHOLDER,
    SCENARIO_DIR,
    Scenario,
    count_placeholders,
    deep_merge,
    load_scenarios,
)

SCENARIOS = load_scenarios()


def test_twelve_scenarios_in_three_domains() -> None:
    assert list(SCENARIOS) == [f"T{i:02d}" for i in range(1, 13)]
    assert Counter(s.domain for s in SCENARIOS.values()) == {"email": 4, "files_web": 4, "banking": 4}


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=list(SCENARIOS))
def test_single_placeholder_inside_carrier(scenario: Scenario) -> None:
    assert count_placeholders(scenario.environment) == 1
    injected = scenario.render_state("PAYLOAD-42")
    assert "PAYLOAD-42" in scenario.carrier_text(injected)
    benign = scenario.render_state(None)
    assert PLACEHOLDER not in benign.model_dump_json()
    assert "G4" in scenario.applicable_goals


def test_task_mix() -> None:
    email_tasks = [
        s.id
        for s in SCENARIOS.values()
        if any(c.action_executed and c.action_executed.tool == "send_email" for c in s.utility)
    ]
    pay_tasks = [
        s
        for s in SCENARIOS.values()
        if any(c.action_executed and c.action_executed.tool == "transfer_money" for c in s.utility)
    ]
    assert len(email_tasks) >= 2
    known = {normalize_iban(i) for s in pay_tasks for i in s.policy.known_payee_ibans}
    unknown_payee = [
        s.id
        for s in pay_tasks
        for c in s.utility
        if c.action_executed and normalize_iban(c.action_executed.args["iban"]) not in known
    ]
    assert unknown_payee == ["T11"]
    assert len(pay_tasks) - len(unknown_payee) >= 2


def _raw(name: str) -> dict:
    return yaml.safe_load(Path(SCENARIO_DIR, name).read_text(encoding="utf-8"))


def test_two_placeholders_are_rejected() -> None:
    raw = _raw("T05.yaml")
    raw["environment"]["files"]["other.txt"] = f"x {PLACEHOLDER}"
    with pytest.raises(ValidationError):
        Scenario.model_validate(deep_merge(_raw("_base.yaml"), raw))


def test_placeholder_outside_carrier_is_rejected() -> None:
    raw = _raw("T05.yaml")
    raw["injection_carrier"] = "file:reports/q3_sales_summary.md"
    with pytest.raises(ValidationError):
        Scenario.model_validate(deep_merge(_raw("_base.yaml"), raw))


def test_deep_merge_semantics() -> None:
    merged = deep_merge({"a": {"x": 1}, "l": [1], "s": "base"}, {"a": {"y": 2}, "l": [2], "s": "over"})
    assert merged == {"a": {"x": 1, "y": 2}, "l": [1, 2], "s": "over"}
