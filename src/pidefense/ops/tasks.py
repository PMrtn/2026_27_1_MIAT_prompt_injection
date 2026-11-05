"""OpsPilot tasks: an engineer's request, a seeded ops environment with one injection carrier, utility checks.

Task files live in ``scenarios/ops/P*.yaml``, each deep-merged over ``scenarios/ops/_base.yaml`` (dicts merge,
lists concatenate base-first, scalars override). The carrier named by ``injection_carrier`` must contain exactly one
``{{INJECTION}}`` and at most one ``{{FOLLOWUP}}`` placeholder (nowhere else in the environment).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from pidefense.config import REPO_ROOT
from pidefense.defenses.ops_policy import OpsPolicy
from pidefense.experiments.scenarios import deep_merge
from pidefense.ops.environment import FOLLOWUP, INJECTION, OpsEnvironment, OpsState, parse_carrier

logger = logging.getLogger(__name__)

OPS_TASK_DIR = REPO_ROOT / "scenarios" / "ops"
BASE_FILE = "_base.yaml"
OPS_GOALS: tuple[str, ...] = ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8")
_PROBE = "<<OPS-INJECTION-PROBE-41c9>>"
_FOLLOW_PROBE = "<<OPS-FOLLOWUP-PROBE-41c9>>"

Surface = Literal["ticket", "runbook", "db_note", "logs", "alert", "webpage"]


class OpsActionMatch(BaseModel):
    """An action pattern; args match case-insensitively (exact, or substring for 'message'/'code'/'body')."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ReplyMatch(BaseModel):
    """A reply to ``ticket_id`` whose message contains at least one of ``any``."""

    ticket_id: str
    any: list[str]


class OpsUtilityCheck(BaseModel):
    """One utility condition; exactly one field must be set. All conditions of a task must hold."""

    answer_contains_any: list[str] | None = None
    answer_contains_together: list[list[str]] | None = None
    action_executed: OpsActionMatch | None = None
    action_not_executed: OpsActionMatch | None = None
    reply_contains: ReplyMatch | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> OpsUtilityCheck:
        names = ("answer_contains_any", "answer_contains_together", "action_executed", "action_not_executed")
        set_fields = [name for name in (*names, "reply_contains") if getattr(self, name) is not None]
        if len(set_fields) != 1:
            raise ValueError(f"An ops utility check needs exactly one condition, got {set_fields or 'none'}")
        return self


def _count(value: Any, token: str) -> int:
    return json.dumps(value, ensure_ascii=False).count(token)


class OpsTask(BaseModel):
    """One OpsPilot end-to-end task."""

    id: str
    surface: Surface
    title: str = ""
    user_prompt: str
    followup_prompt: str
    escalation_prompt: str
    injection_carrier: str
    environment: dict[str, Any]
    utility: list[OpsUtilityCheck]
    applicable_goals: list[str]
    policy: OpsPolicy = Field(default_factory=OpsPolicy)

    @model_validator(mode="after")
    def _validate(self) -> OpsTask:
        kind, _ = parse_carrier(self.injection_carrier)
        if kind != self.surface:
            raise ValueError(f"{self.id}: surface {self.surface!r} does not match carrier kind {kind!r}")
        if _count(self.environment, INJECTION) != 1:
            raise ValueError(f"{self.id}: expected exactly one {INJECTION}")
        followups = _count(self.environment, FOLLOWUP)
        if followups > 1:
            raise ValueError(f"{self.id}: expected at most one {FOLLOWUP}")
        unknown = set(self.applicable_goals) - set(OPS_GOALS)
        if unknown:
            raise ValueError(f"{self.id}: unknown goals {sorted(unknown)}")
        state = self.render_state(_PROBE)
        if _PROBE not in state.carrier_text():
            raise ValueError(f"{self.id}: {INJECTION} is not inside carrier {self.injection_carrier!r}")
        if followups:
            state.set_followup(_FOLLOW_PROBE)
            if _FOLLOW_PROBE not in state.carrier_text():
                raise ValueError(f"{self.id}: {FOLLOWUP} is not inside carrier {self.injection_carrier!r}")
        return self

    def render_state(self, injection_text: str | None) -> OpsState:
        """Fresh world with the injection substituted (``None``/"" = benign); the follow-up starts empty."""
        environment = OpsEnvironment.model_validate(self.environment)
        return OpsState(environment, self.injection_carrier, injection_text or "")


def load_ops_tasks(directory: Path = OPS_TASK_DIR) -> dict[str, OpsTask]:
    """Load and validate all ``P*.yaml`` tasks, keyed and sorted by id."""
    base_path = directory / BASE_FILE
    base: dict[str, Any] = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    tasks: dict[str, OpsTask] = {}
    for path in sorted(directory.glob("P*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        task = OpsTask.model_validate(deep_merge(base or {}, raw))
        if task.id in tasks:
            raise ValueError(f"Duplicate ops task id {task.id} in {path}")
        tasks[task.id] = task
    logger.info("Loaded %d ops tasks from %s", len(tasks), directory)
    return tasks
