"""Cost accounting, budget enforcement and the exact-request cache (no network)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest

from pidefense.config import GPT_56_LUNA_PRICING, LLMSettings
from pidefense.llm import AzureChatClient, BudgetExceededError, CostLedger, ResponseCache, TokenUsage


def test_usage_parsing_and_cost() -> None:
    usage = TokenUsage.from_usage_dict(
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "prompt_tokens_details": {"cached_tokens": 400_000, "cache_write_tokens": 100_000},
            "completion_tokens_details": {"reasoning_tokens": 250_000},
        }
    )
    assert usage.reasoning_tokens == 250_000
    # 500k uncached * 0.20 + 400k cached * 0.02 + 100k write * 0.25 + 1M out * 1.20 (per 1M tokens)
    expected = 0.5 * 0.20 + 0.4 * 0.02 + 0.1 * 0.25 + 1.0 * 1.20
    assert usage.cost_usd(GPT_56_LUNA_PRICING) == pytest.approx(expected)


def test_empty_usage_is_free() -> None:
    assert TokenUsage.from_usage_dict(None).cost_usd(GPT_56_LUNA_PRICING) == 0.0


def test_ledger_persists_and_enforces_hard_budget(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    ledger = CostLedger(path, warn_usd=0.5, hard_usd=1.0)
    usage = TokenUsage(prompt_tokens=0, completion_tokens=500_000)  # $0.60
    ledger.ensure_budget()
    assert ledger.record("test", "dep", usage, GPT_56_LUNA_PRICING) == pytest.approx(0.6)
    ledger.ensure_budget()  # $0.60 < $1.00
    ledger.record("test", "dep", usage, GPT_56_LUNA_PRICING)

    reloaded = CostLedger(path, warn_usd=0.5, hard_usd=1.0)
    assert reloaded.total_usd == pytest.approx(1.2)
    with pytest.raises(BudgetExceededError):
        reloaded.ensure_budget()


def test_cache_roundtrip_and_key_stability(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    key_a = ResponseCache.make_key({"b": 1, "a": [1, 2]})
    key_b = ResponseCache.make_key({"a": [1, 2], "b": 1})
    assert key_a == key_b
    cache = ResponseCache(path)
    cache.put(key_a, {"x": "é"})
    assert ResponseCache(path).get(key_a) == {"x": "é"}
    assert ResponseCache.make_key({"namespace": "rep1"}) != ResponseCache.make_key({"namespace": "rep2"})


def _settings(tmp_path: Path, hard_usd: float = 9.5) -> LLMSettings:
    return LLMSettings(ledger_path=tmp_path / "ledger.csv", budget_hard_usd=hard_usd)


async def test_client_serves_cache_hits_without_spending(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cache = ResponseCache(tmp_path / "cache.jsonl")
    messages = [{"role": "user", "content": "hi"}]
    key = ResponseCache.make_key({"namespace": "", "model": settings.deployment, "messages": messages})
    raw = {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    cache.put(key, {"raw": raw, "latency_ms": 123.0})
    client = AzureChatClient(settings=settings, cache=cache)
    response = await client.chat(messages, purpose="test")
    assert response.content == "hello"
    assert response.from_cache and response.cost_usd == 0.0 and response.latency_ms == 123.0
    assert client.ledger.total_usd == 0.0


async def test_client_refuses_calls_over_budget(tmp_path: Path) -> None:
    settings = _settings(tmp_path, hard_usd=0.0)
    client = AzureChatClient(settings=settings)
    with pytest.raises(BudgetExceededError):
        await client.chat([{"role": "user", "content": "hi"}], purpose="test")
