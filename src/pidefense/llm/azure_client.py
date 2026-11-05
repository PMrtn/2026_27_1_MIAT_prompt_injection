"""Async Azure OpenAI chat client: Entra ID auth, cost ledger, hard budget and exact-request cache.

Smoke test (one tiny billed call): ``python -m pidefense.llm "Reply with exactly: OK"``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai
from openai import AsyncOpenAI

from pidefense.config import LLMSettings
from pidefense.llm.cache import ResponseCache
from pidefense.llm.cost import CostLedger, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalised assistant turn plus accounting data."""

    message: dict[str, Any]
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    server_latency: dict[str, Any] | None = None
    from_cache: bool = False
    blocked_by_provider: bool = False
    finish_reason: str | None = None

    @property
    def content(self) -> str | None:
        """Assistant text, if any."""
        return self.message.get("content")

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Tool calls in OpenAI wire format: {"id", "type": "function", "function": {"name", "arguments"}}."""
        return list(self.message.get("tool_calls") or [])


class LLMClient(Protocol):
    """Interface shared by the Azure client and test doubles."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        purpose: str,
        cache_namespace: str = "",
        **params: Any,
    ) -> LLMResponse:
        """Return the next assistant turn."""


def _normalise_message(raw_message: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": raw_message.get("content")}
    tool_calls = raw_message.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
            }
            for call in tool_calls
        ]
    return message


def _is_content_filter_error(exc: openai.APIStatusError) -> bool:
    body = exc.body if isinstance(exc.body, dict) else {}
    code = str(body.get("code") or "") + str((body.get("error") or {}).get("code") if body.get("error") else "")
    return "content_filter" in code or "content_filter" in str(exc).lower()


class AzureChatClient:
    """Chat-completions client for the project's Azure deployment."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        ledger: CostLedger | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.settings = settings or LLMSettings.from_env()
        self.ledger = ledger or CostLedger(
            self.settings.ledger_path, self.settings.budget_warn_usd, self.settings.budget_hard_usd
        )
        self.cache = cache
        self._client: AsyncOpenAI | None = None
        self._credential: Any = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            # Imported lazily so tests and offline tools never need Azure credentials.
            from azure.identity.aio import (  # pylint: disable=import-outside-toplevel
                AzureCliCredential,
                get_bearer_token_provider,
            )

            self._credential = AzureCliCredential()
            token_provider = get_bearer_token_provider(self._credential, self.settings.token_scope)
            self._client = AsyncOpenAI(
                base_url=self.settings.endpoint,
                api_key=token_provider,
                max_retries=self.settings.max_retries,
                timeout=self.settings.timeout_s,
            )
        return self._client

    async def aclose(self) -> None:
        """Release HTTP and credential resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._credential is not None:
            await self._credential.close()
            self._credential = None

    def _from_raw(self, raw: dict[str, Any], latency_ms: float, cost: float, from_cache: bool) -> LLMResponse:
        choice = raw["choices"][0]
        finish_reason = choice.get("finish_reason")
        usage_raw = raw.get("usage") or {}
        return LLMResponse(
            message=_normalise_message(choice.get("message") or {}),
            usage=TokenUsage.from_usage_dict(usage_raw),
            cost_usd=cost,
            latency_ms=latency_ms,
            server_latency=usage_raw.get("latency_checkpoint"),
            from_cache=from_cache,
            blocked_by_provider=finish_reason == "content_filter",
            finish_reason=finish_reason,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        purpose: str,
        cache_namespace: str = "",
        **params: Any,
    ) -> LLMResponse:
        """Run one chat completion. Extra keyword params (e.g. reasoning_effort) are passed through."""
        request: dict[str, Any] = {"model": self.settings.deployment, "messages": messages}
        if tools:
            request["tools"] = tools
        request.update({key: value for key, value in params.items() if value is not None})

        key = ResponseCache.make_key({"namespace": cache_namespace, **request}) if self.cache is not None else None
        if self.cache is not None and key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                logger.debug("Cache hit (%s)", purpose)
                if hit.get("blocked_by_provider"):
                    return LLMResponse(
                        message={"role": "assistant", "content": None}, from_cache=True, blocked_by_provider=True
                    )
                return self._from_raw(hit["raw"], float(hit.get("latency_ms", 0.0)), 0.0, from_cache=True)

        self.ledger.ensure_budget()
        start = time.perf_counter()
        try:
            completion = await self._get_client().chat.completions.create(**request)
        except openai.BadRequestError as exc:
            if not _is_content_filter_error(exc):
                raise
            logger.warning("Request blocked by the provider content filter (%s)", purpose)
            if self.cache is not None and key is not None:
                self.cache.put(key, {"blocked_by_provider": True})
            return LLMResponse(
                message={"role": "assistant", "content": None},
                latency_ms=(time.perf_counter() - start) * 1000,
                blocked_by_provider=True,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        raw = completion.model_dump()
        usage = TokenUsage.from_usage_dict(raw.get("usage"))
        cost = self.ledger.record(purpose, self.settings.deployment, usage, self.settings.pricing)
        if self.cache is not None and key is not None:
            self.cache.put(key, {"raw": raw, "latency_ms": latency_ms})
        return self._from_raw(raw, latency_ms, cost, from_cache=False)
