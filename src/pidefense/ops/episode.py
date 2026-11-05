"""Multi-turn OpsPilot episodes: L0 task -> L1 follow-up (carrier updated) -> optional L2 operator authorization.

One conversation per episode. After every level the goal outcome and the task utility are evaluated and stored in
``level_outcomes``; the base ``EpisodeResult`` fields hold the outcome at the last level that was run.

Defense semantics match the office agent: every read output (including ``query_db`` results and ``fetch_url``)
passes ``pipeline.filter_output``; every action tool AND every ``query_db`` call passes ``pipeline.check_action``
first, and a block means the call is not executed and the LLM receives ``ACTION_BLOCKED_TEMPLATE``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import openai
from mcp import Client
from pydantic import Field

from pidefense.agent.loop import _call_mcp, _llm_record  # pylint: disable=protected-access
from pidefense.agent.tools import mcp_tools_to_openai
from pidefense.defenses import ACTION_BLOCKED_TEMPLATE, DefensePipeline
from pidefense.llm import LLMClient
from pidefense.ops.checkers import check_goal, check_utility
from pidefense.ops.environment import OpsState
from pidefense.ops.prompts import PromptVariant, system_prompt
from pidefense.ops.server import OPS_TOOL_KINDS, POLICY_CHECKED_READS, build_ops_server
from pidefense.ops.tasks import OpsTask
from pidefense.schemas import EpisodeResult, ToolCallRecord, ToolContext

logger = logging.getLogger(__name__)

LEVELS: tuple[str, ...] = ("L0", "L1", "L2")
DEFAULT_OPS_LLM_PARAMS: dict[str, Any] = {"reasoning_effort": "none", "max_completion_tokens": 1500}


class OpsEpisodeResult(EpisodeResult):
    """EpisodeResult plus the prompt variant and the per-level outcomes of the conversation."""

    prompt_variant: str = "B"
    levels: list[str] = Field(default_factory=list)
    level_outcomes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    answers: list[str] = Field(default_factory=list)


async def _handle_ops_tool_call(
    client: Client, pipeline: DefensePipeline, call: dict[str, Any], step: int, base_ctx: ToolContext
) -> tuple[ToolCallRecord, str]:
    """Execute one tool call through the middleware; return its record and the text the LLM receives."""
    name = call["function"]["name"]
    kind = OPS_TOOL_KINDS.get(name)
    try:
        arguments = json.loads(call["function"].get("arguments") or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        message = f"Error: invalid arguments for '{name}': {exc}"
        record = ToolCallRecord(step=step, call_id=call["id"], tool_name=name, kind=kind or "read", executed=False)
        record.error = record.delivered_output = message
        return record, message
    record = ToolCallRecord(
        step=step, call_id=call["id"], tool_name=name, arguments=arguments, kind=kind or "read", executed=False
    )
    if kind is None:
        record.error = record.delivered_output = f"Error: unknown tool '{name}'."
        return record, record.delivered_output

    ctx = base_ctx.model_copy(update={"tool_name": name, "arguments": arguments})
    if kind == "action" or name in POLICY_CHECKED_READS:
        check = pipeline.check_action(ctx)
        record.decisions = list(check.decisions)
        record.defense_latency_ms = check.latency_ms
        if not check.allowed:
            record.blocked_by = check.blocked_by
            record.delivered_output = ACTION_BLOCKED_TEMPLATE.format(tool=name, reason=check.reason)
            return record, record.delivered_output

    text, tool_ms, error = await _call_mcp(client, name, arguments)
    record.executed, record.tool_latency_ms, record.error, record.raw_output = error is None, tool_ms, error, text
    if kind == "action" or error is not None:
        # Action results and tool error messages are produced by the environment itself: nothing to filter.
        record.delivered_output = text
        return record, text
    filtered = pipeline.filter_output(text, ctx)
    record.decisions += filtered.decisions
    record.defense_latency_ms += filtered.latency_ms
    record.blocked_by = filtered.blocked_by
    record.delivered_output = filtered.text
    return record, filtered.text


async def _run_level(
    llm: LLMClient,
    client: Client,
    pipeline: DefensePipeline,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    result: OpsEpisodeResult,
    base_ctx: ToolContext,
    *,
    config_name: str,
    cache_namespace: str,
    params: dict[str, Any],
    max_steps: int,
) -> str | None:
    """Run LLM turns until a final answer; return it (None on error / provider block / step limit)."""
    for _ in range(max_steps):
        step = len(result.llm_calls)
        try:
            response = await llm.chat(
                messages, tools, purpose=f"ops-episode:{config_name}", cache_namespace=cache_namespace, **params
            )
        except openai.APIError as exc:
            logger.warning("Ops episode %s: LLM error at step %d: %s", result.episode_id, step, exc)
            result.error = f"llm_error: {exc}"
            return None
        result.llm_calls.append(_llm_record(step, response))
        if response.blocked_by_provider:
            result.blocked_by_provider = True
            return None
        messages.append(response.message)
        if not response.tool_calls:
            return response.content or ""
        for call in response.tool_calls:
            record, delivered = await _handle_ops_tool_call(client, pipeline, call, step, base_ctx)
            result.tool_calls.append(record)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": delivered})
    result.error = "max_steps_reached"
    return None


def _evaluate(task: OpsTask, result: OpsEpisodeResult, state: OpsState, level: str) -> None:
    utility = check_utility(task, "\n".join(result.answers), state)
    outcome: dict[str, Any] = {"utility": utility}
    if result.goal:
        success, attempted = check_goal(result.goal, task, result, state)
        outcome.update(success=success, attempted=attempted)
        result.attack_success, result.hijack_attempted = success, attempted
    result.utility_success = utility
    result.level_outcomes[level] = outcome


async def run_ops_episode(
    llm: LLMClient,
    task: OpsTask,
    injection_text: str | None,
    followup_text: str | None,
    pipeline: DefensePipeline,
    *,
    prompt_variant: PromptVariant,
    config_name: str,
    model: str,
    episode_id: str,
    levels: tuple[str, ...] = ("L0", "L1"),
    attack_id: str | None = None,
    family: str | None = None,
    lang: str | None = None,
    goal: str | None = None,
    repetition: int = 0,
    cache_namespace: str = "",
    llm_params: dict[str, Any] | None = None,
    max_steps_per_level: int = 10,
) -> OpsEpisodeResult:
    """Run one multi-level conversation and return its full log with per-level outcomes.

    ``injection_text``/``followup_text`` None or "" give the benign variant (the L1/L2 prompts are still sent).
    ``BudgetExceededError`` propagates. ``total_latency_ms`` = LLM (cache hits replay the original latency) + tool +
    defense time; the elapsed time is ``wall_clock_ms``.
    """
    unknown = [level for level in levels if level not in LEVELS]
    if not levels or levels[0] != "L0" or unknown or list(levels) != sorted(levels):
        raise ValueError(f"levels must be an ordered prefix-style subset of {LEVELS} starting with L0, got {levels}")
    params = {**DEFAULT_OPS_LLM_PARAMS, **(llm_params or {})}
    if not cache_namespace and repetition > 0:
        cache_namespace = f"rep{repetition}"
    state = task.render_state(injection_text)
    server = build_ops_server(state)
    result = OpsEpisodeResult(
        episode_id=episode_id,
        model=model,
        config=config_name,
        task_id=task.id,
        domain="ops",
        attack_id=attack_id,
        family=family,
        lang=lang,
        goal=goal,
        repetition=repetition,
        user_prompt=task.user_prompt,
        prompt_variant=prompt_variant,
        levels=list(levels),
    )
    base_ctx = ToolContext(episode_id=episode_id, tool_name="", user_prompt=task.user_prompt)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt(prompt_variant)}]
    prompts = {"L0": task.user_prompt, "L1": task.followup_prompt, "L2": task.escalation_prompt}
    start = time.perf_counter()
    async with Client(server) as client:
        tools = mcp_tools_to_openai((await client.list_tools()).tools)
        for level in levels:
            if level == "L1":
                state.set_followup(followup_text or "")
            messages.append({"role": "user", "content": prompts[level]})
            answer = await _run_level(
                llm,
                client,
                pipeline,
                messages,
                tools,
                result,
                base_ctx,
                config_name=config_name,
                cache_namespace=cache_namespace,
                params=params,
                max_steps=max_steps_per_level,
            )
            if answer is not None:
                result.answers.append(answer)
                result.final_answer = answer
            _evaluate(task, result, state, level)
            if answer is None:
                break
    result.wall_clock_ms = (time.perf_counter() - start) * 1000
    result.steps = len(result.llm_calls)
    result.defense_latency_ms = sum(call.defense_latency_ms for call in result.tool_calls)
    result.total_latency_ms = (
        sum(call.latency_ms for call in result.llm_calls)
        + sum(call.tool_latency_ms for call in result.tool_calls)
        + result.defense_latency_ms
    )
    logger.info(
        "Ops episode %s [%s/%s/%s goal=%s]: steps=%d levels=%s error=%s",
        episode_id,
        task.id,
        prompt_variant,
        config_name,
        goal,
        result.steps,
        result.level_outcomes,
        result.error,
    )
    return result
