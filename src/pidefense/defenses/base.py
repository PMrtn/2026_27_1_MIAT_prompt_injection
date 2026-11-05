"""Defense interfaces, the four experimental configurations and the middleware pipeline.

The pipeline sits between MCP tool execution and the LLM:

* ``check_action`` runs before an *action* tool executes (R3 policy). A block returns an error to the LLM.
* ``filter_output`` runs on every *read* tool output, layer by layer (R1 sanitize -> R2 signatures -> ML).
  The first blocking layer replaces the whole output with ``WITHHELD_TEMPLATE``; later layers do not run.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from pidefense.schemas import DefenseDecision, ToolContext

logger = logging.getLogger(__name__)

WITHHELD_TEMPLATE = (
    "[Content withheld by security filter: possible prompt injection detected in the output of '{tool}'.]"
)
ACTION_BLOCKED_TEMPLATE = "Error: the action '{tool}' was blocked by the security policy: {reason}"


class OutputDefense(ABC):
    """A layer that inspects (and may sanitise or block) a tool output before the LLM sees it."""

    name: str = "output-defense"

    @abstractmethod
    def inspect(self, text: str, ctx: ToolContext) -> DefenseDecision:
        """Return a decision; set ``allowed=False`` to block or ``text`` to rewrite the output."""


class ActionPolicy(ABC):
    """A deterministic rule set that may veto an action tool call before it executes."""

    name: str = "action-policy"

    @abstractmethod
    def check(self, ctx: ToolContext) -> DefenseDecision:
        """Return a decision; ``allowed=False`` vetoes the call."""


@dataclass(frozen=True)
class DefenseConfig:
    """One of the four experimental configurations."""

    guardrails: bool
    ml: bool

    @property
    def name(self) -> str:
        """Short label used in logs, tables and file names."""
        if self.guardrails and self.ml:
            return "combined"
        if self.guardrails:
            return "guardrails"
        if self.ml:
            return "ml"
        return "none"

    @property
    def number(self) -> int:
        """Configuration number as used in the research plan (1-4)."""
        return {"none": 1, "guardrails": 2, "ml": 3, "combined": 4}[self.name]

    @classmethod
    def from_name(cls, name: str) -> DefenseConfig:
        """Parse 'none' | 'guardrails' | 'ml' | 'combined'."""
        for config in ALL_CONFIGS:
            if config.name == name:
                return config
        raise ValueError(f"Unknown defense config {name!r}")


ALL_CONFIGS: tuple[DefenseConfig, ...] = (
    DefenseConfig(guardrails=False, ml=False),
    DefenseConfig(guardrails=True, ml=False),
    DefenseConfig(guardrails=False, ml=True),
    DefenseConfig(guardrails=True, ml=True),
)


@dataclass
class OutputFilterResult:
    """What the LLM receives for one tool output, and why."""

    text: str
    blocked: bool
    blocked_by: str | None = None
    decisions: list[DefenseDecision] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class ActionCheckResult:
    """Whether an action may execute, and why."""

    allowed: bool
    blocked_by: str | None = None
    reason: str = ""
    decisions: list[DefenseDecision] = field(default_factory=list)
    latency_ms: float = 0.0


class DefensePipeline:
    """Ordered output layers and action policies. An empty pipeline is configuration 1 (no defense)."""

    def __init__(
        self, output_defenses: Sequence[OutputDefense] = (), action_policies: Sequence[ActionPolicy] = ()
    ) -> None:
        self.output_defenses = list(output_defenses)
        self.action_policies = list(action_policies)

    @property
    def layer_names(self) -> list[str]:
        """Names of all layers in execution order."""
        return [layer.name for layer in self.output_defenses] + [policy.name for policy in self.action_policies]

    def filter_output(self, text: str, ctx: ToolContext) -> OutputFilterResult:
        """Run the output layers in order; the first block withholds the whole output."""
        current = text
        decisions: list[DefenseDecision] = []
        total_ms = 0.0
        for layer in self.output_defenses:
            start = time.perf_counter()
            decision = layer.inspect(current, ctx)
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_ms += elapsed_ms
            decision = decision.model_copy(update={"layer": layer.name, "latency_ms": elapsed_ms})
            decisions.append(decision)
            if not decision.allowed:
                logger.debug("Output of %s blocked by %s: %s", ctx.tool_name, layer.name, decision.reason)
                return OutputFilterResult(
                    text=WITHHELD_TEMPLATE.format(tool=ctx.tool_name),
                    blocked=True,
                    blocked_by=layer.name,
                    decisions=decisions,
                    latency_ms=total_ms,
                )
            if decision.text is not None:
                current = decision.text
        return OutputFilterResult(text=current, blocked=False, decisions=decisions, latency_ms=total_ms)

    def check_action(self, ctx: ToolContext) -> ActionCheckResult:
        """Run the action policies in order; the first veto blocks the call."""
        decisions: list[DefenseDecision] = []
        total_ms = 0.0
        for policy in self.action_policies:
            start = time.perf_counter()
            decision = policy.check(ctx)
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_ms += elapsed_ms
            decision = decision.model_copy(update={"layer": policy.name, "latency_ms": elapsed_ms})
            decisions.append(decision)
            if not decision.allowed:
                logger.debug("Action %s blocked by %s: %s", ctx.tool_name, policy.name, decision.reason)
                return ActionCheckResult(
                    allowed=False,
                    blocked_by=policy.name,
                    reason=decision.reason,
                    decisions=decisions,
                    latency_ms=total_ms,
                )
        return ActionCheckResult(allowed=True, decisions=decisions, latency_ms=total_ms)
