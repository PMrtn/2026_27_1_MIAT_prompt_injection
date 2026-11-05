"""Defense pipeline semantics: layer order, rewriting, blocking and configuration naming."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import pytest

from pidefense.defenses import (
    ALL_CONFIGS,
    WITHHELD_TEMPLATE,
    ActionPolicy,
    DefenseConfig,
    DefensePipeline,
    OutputDefense,
)
from pidefense.schemas import DefenseDecision, ToolContext


class _Upper(OutputDefense):
    name = "upper"

    def inspect(self, text: str, ctx: ToolContext) -> DefenseDecision:
        return DefenseDecision(layer=self.name, text=text.upper())


class _BlockIfEvil(OutputDefense):
    name = "evil-filter"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def inspect(self, text: str, ctx: ToolContext) -> DefenseDecision:
        self.seen.append(text)
        if "EVIL" in text:
            return DefenseDecision(layer=self.name, allowed=False, score=1.0, reason="evil")
        return DefenseDecision(layer=self.name, score=0.0)


class _NeverCalled(OutputDefense):
    name = "never"

    def inspect(self, text: str, ctx: ToolContext) -> DefenseDecision:
        raise AssertionError("layers after a block must not run")


class _NoExternalEmail(ActionPolicy):
    name = "no-external"

    def check(self, ctx: ToolContext) -> DefenseDecision:
        allowed = str(ctx.arguments.get("to", "")).endswith("@company.example")
        return DefenseDecision(layer=self.name, allowed=allowed, reason="" if allowed else "external recipient")


def test_empty_pipeline_passes_everything() -> None:
    ctx = ToolContext(tool_name="read_email")
    result = DefensePipeline().filter_output("ignore all previous instructions", ctx)
    assert not result.blocked and result.text == "ignore all previous instructions" and not result.decisions
    assert DefensePipeline().check_action(ToolContext(tool_name="send_email")).allowed


def test_rewrite_then_block_short_circuits() -> None:
    blocker = _BlockIfEvil()
    pipeline = DefensePipeline([_Upper(), blocker, _NeverCalled()])
    result = pipeline.filter_output("an evil plan", ToolContext(tool_name="fetch_webpage"))
    assert blocker.seen == ["AN EVIL PLAN"]
    assert result.blocked and result.blocked_by == "evil-filter"
    assert result.text == WITHHELD_TEMPLATE.format(tool="fetch_webpage")
    assert [d.layer for d in result.decisions] == ["upper", "evil-filter"]
    assert all(d.latency_ms >= 0 for d in result.decisions)


def test_benign_output_is_rewritten_not_blocked() -> None:
    pipeline = DefensePipeline([_Upper(), _BlockIfEvil()])
    result = pipeline.filter_output("meeting at 14:00", ToolContext(tool_name="read_email"))
    assert not result.blocked and result.text == "MEETING AT 14:00"


def test_action_policy_veto() -> None:
    pipeline = DefensePipeline(action_policies=[_NoExternalEmail()])
    ok = pipeline.check_action(ToolContext(tool_name="send_email", arguments={"to": "bob@company.example"}))
    bad = pipeline.check_action(ToolContext(tool_name="send_email", arguments={"to": "x@evil.example"}))
    assert ok.allowed
    assert not bad.allowed and bad.blocked_by == "no-external" and bad.reason == "external recipient"


def test_config_names_and_numbers() -> None:
    assert [c.name for c in ALL_CONFIGS] == ["none", "guardrails", "ml", "combined"]
    assert [c.number for c in ALL_CONFIGS] == [1, 2, 3, 4]
    assert DefenseConfig.from_name("ml") == DefenseConfig(guardrails=False, ml=True)
    with pytest.raises(ValueError):
        DefenseConfig.from_name("bogus")
