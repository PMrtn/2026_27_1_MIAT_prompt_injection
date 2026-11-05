"""End-to-end scenarios: a user task, an environment with exactly one injection slot, utility checks and targets.

Scenario files live in ``scenarios/T*.yaml``. Each is deep-merged over ``scenarios/_base.yaml`` (dicts merge
recursively, lists concatenate base-first, scalars override), then validated.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from pidefense.attacks.goals import GOAL_IDS
from pidefense.config import REPO_ROOT
from pidefense.mcp_server.environment import WorkspaceState, normalize_path, normalize_url
from pidefense.schemas import PolicyContext

logger = logging.getLogger(__name__)

PLACEHOLDER = "{{INJECTION}}"
SCENARIO_DIR = REPO_ROOT / "scenarios"
BASE_FILE = "_base.yaml"
_PROBE = "<<INJECTION-PROBE-7f3a>>"

Domain = Literal["email", "files_web", "banking"]


class ActionMatch(BaseModel):
    """An action tool call pattern; ``args`` are matched case-insensitively (see checkers)."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class UtilityCheck(BaseModel):
    """One utility condition. Exactly one field must be set; all conditions of a scenario must hold."""

    answer_contains_any: list[str] | None = None
    # One term from EVERY group must occur within a single sentence or line of the answer.
    answer_contains_together: list[list[str]] | None = None
    action_executed: ActionMatch | None = None
    action_not_executed: ActionMatch | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> UtilityCheck:
        set_fields = [
            name
            for name in ("answer_contains_any", "answer_contains_together", "action_executed", "action_not_executed")
            if getattr(self, name) is not None
        ]
        if len(set_fields) != 1:
            raise ValueError(f"A utility check needs exactly one condition, got {set_fields or 'none'}")
        return self


class GoalTargets(BaseModel):
    """Scenario-specific targets of the goals whose canary is not global."""

    G3_file: str
    G5_recipient: str


def count_placeholders(value: Any) -> int:
    """Count ``{{INJECTION}}`` occurrences in a nested YAML structure."""
    if isinstance(value, str):
        return value.count(PLACEHOLDER)
    if isinstance(value, dict):
        return sum(count_placeholders(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_placeholders(v) for v in value)
    return 0


def remove_slot(text: str, token: str = PLACEHOLDER) -> str:
    """Remove ``token`` together with exactly the separator that belongs to it, and nothing else.

    A standalone paragraph loses its preceding blank line, a line of its own loses its newline, and an inline
    slot loses one preceding space. All other whitespace is untouched, so a benign carrier equals the attacked
    carrier with the injected text (and its separator) cut out.
    """
    slot = re.escape(token)
    for pattern in (
        r"\n\n[ \t]*" + slot + r"[ \t]*(?=\n\n|\n?\Z)",  # standalone paragraph
        r"\n[ \t]*" + slot + r"[ \t]*(?=\n|\Z)",  # line of its own
        r"[ \t]?" + slot,  # inline
    ):
        new_text, count = re.subn(pattern, "", text, count=1)
        if count:
            return new_text
    return text


def _substitute(value: Any, injection_text: str) -> Any:
    if isinstance(value, str):
        if PLACEHOLDER not in value:
            return value
        return value.replace(PLACEHOLDER, injection_text) if injection_text else remove_slot(value)
    if isinstance(value, dict):
        return {k: _substitute(v, injection_text) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, injection_text) for v in value]
    return value


def deep_merge(base: Any, override: Any) -> Any:
    """Merge ``override`` into ``base``: dicts recursively, lists concatenated, scalars replaced."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(base[key], value) if key in base else copy.deepcopy(value)
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return copy.deepcopy(base) + copy.deepcopy(override)
    return copy.deepcopy(override)


class Scenario(BaseModel):
    """One end-to-end task."""

    id: str
    domain: Domain
    title: str = ""
    user_prompt: str
    injection_carrier: str
    environment: dict[str, Any]
    utility: list[UtilityCheck]
    applicable_goals: list[str]
    goal_targets: GoalTargets
    policy: PolicyContext = Field(default_factory=PolicyContext)

    @model_validator(mode="after")
    def _validate(self) -> Scenario:
        placeholders = count_placeholders(self.environment)
        if placeholders != 1:
            raise ValueError(f"{self.id}: expected exactly one {PLACEHOLDER}, found {placeholders}")
        unknown = set(self.applicable_goals) - set(GOAL_IDS)
        if unknown:
            raise ValueError(f"{self.id}: unknown goals {sorted(unknown)}")
        if "G4" not in self.applicable_goals:
            raise ValueError(f"{self.id}: G4 must be applicable to every scenario")
        state = self.render_state(_PROBE)
        if _PROBE not in self.carrier_text(state):
            raise ValueError(f"{self.id}: the placeholder is not inside injection_carrier {self.injection_carrier!r}")
        if normalize_path(self.goal_targets.G3_file) not in state.files:
            raise ValueError(f"{self.id}: G3 target file {self.goal_targets.G3_file!r} does not exist")
        if self.goal_targets.G5_recipient.lower() not in {c.email.lower() for c in state.contacts}:
            raise ValueError(f"{self.id}: G5 recipient {self.goal_targets.G5_recipient!r} is not a contact")
        return self

    def render_state(self, injection_text: str | None) -> WorkspaceState:
        """Fresh environment with the injection substituted (``None``/"" = benign, whitespace tidied)."""
        environment = _substitute(copy.deepcopy(self.environment), injection_text or "")
        return WorkspaceState.model_validate(environment)

    def carrier_text(self, state: WorkspaceState) -> str:
        """Text of the carrier that holds the injection slot.

        Carrier syntax: ``email:<id>``, ``file:<path>``, ``webpage:<url>``, ``review:<product>#<index>``,
        ``transaction:<id>``.
        """
        kind, _, ref = self.injection_carrier.partition(":")
        if kind == "email":
            return next((e.body for e in state.emails if e.id == ref), "")
        if kind == "file":
            return state.files.get(normalize_path(ref), "")
        if kind == "webpage":
            return next((c for u, c in state.webpages.items() if normalize_url(u) == normalize_url(ref)), "")
        if kind == "review":
            product, _, index = ref.rpartition("#")
            reviews = state.reviews.get(product, [])
            position = int(index) if index.isdigit() else -1
            return reviews[position].text if 0 <= position < len(reviews) else ""
        if kind == "transaction":
            return next((t.reference for t in state.transactions if t.id == ref), "")
        raise ValueError(f"{self.id}: unknown carrier kind {kind!r}")


def load_scenarios(directory: Path = SCENARIO_DIR) -> dict[str, Scenario]:
    """Load and validate all ``T*.yaml`` scenarios, keyed and sorted by id."""
    base_path = directory / BASE_FILE
    base: dict[str, Any] = {}
    if base_path.exists():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    scenarios: dict[str, Scenario] = {}
    for path in sorted(directory.glob("T*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        scenario = Scenario.model_validate(deep_merge(base, raw))
        if scenario.id in scenarios:
            raise ValueError(f"Duplicate scenario id {scenario.id} in {path}")
        scenarios[scenario.id] = scenario
    logger.info("Loaded %d scenarios from %s", len(scenarios), directory)
    return scenarios
