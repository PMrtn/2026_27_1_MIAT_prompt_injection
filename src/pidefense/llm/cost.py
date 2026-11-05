"""Token accounting, a persistent cost ledger and the hard project budget."""

from __future__ import annotations

import csv
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pidefense.config import ModelPricing

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised before an API call once the cumulative spend has reached the hard budget."""


@dataclass(frozen=True)
class TokenUsage:
    """Billable token counts of one chat completion."""

    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_usage_dict(cls, usage: Mapping[str, Any] | None) -> TokenUsage:
        """Parse the `usage` object of a chat completion (Azure adds cache_write_tokens)."""
        if not usage:
            return cls()
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_tokens=int(prompt_details.get("cached_tokens") or 0),
            cache_write_tokens=int(prompt_details.get("cache_write_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        )

    def cost_usd(self, pricing: ModelPricing) -> float:
        """Price the call. Cached and cache-write tokens are subsets of prompt tokens; reasoning is in completion."""
        uncached = max(self.prompt_tokens - self.cached_tokens - self.cache_write_tokens, 0)
        micro_usd = (
            uncached * pricing.input_per_m
            + self.cached_tokens * pricing.cached_input_per_m
            + self.cache_write_tokens * pricing.cache_write_per_m
            + self.completion_tokens * pricing.output_per_m
        )
        return micro_usd / 1_000_000


class CostLedger:
    """Append-only CSV of every billed call. The running total survives restarts and enforces the budget.

    Concurrent in-flight requests can overshoot the hard limit by at most (concurrency x cost of one call),
    which is a fraction of a cent for this project.
    """

    FIELDS = (
        "timestamp",
        "purpose",
        "deployment",
        "prompt_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cost_usd",
    )

    def __init__(self, path: Path, warn_usd: float, hard_usd: float) -> None:
        self.path = path
        self.warn_usd = warn_usd
        self.hard_usd = hard_usd
        self._lock = threading.Lock()
        self._total = self._load_total()
        self._warned = self._total >= warn_usd
        logger.info(
            "Cost ledger %s: spent so far $%.4f (warn $%.2f, hard stop $%.2f)", path, self._total, warn_usd, hard_usd
        )

    def _load_total(self) -> float:
        if not self.path.exists():
            return 0.0
        with self.path.open(newline="", encoding="utf-8") as handle:
            return sum(float(row["cost_usd"]) for row in csv.DictReader(handle))

    @property
    def total_usd(self) -> float:
        """Cumulative spend recorded so far."""
        return self._total

    def ensure_budget(self) -> None:
        """Refuse to start another call once the hard budget is reached."""
        if self._total >= self.hard_usd:
            raise BudgetExceededError(
                f"LLM budget exhausted: ${self._total:.4f} spent, hard limit ${self.hard_usd:.2f}"
            )

    def record(self, purpose: str, deployment: str, usage: TokenUsage, pricing: ModelPricing) -> float:
        """Append one billed call and return its cost in USD."""
        cost = usage.cost_usd(pricing)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": purpose,
            "deployment": deployment,
            "prompt_tokens": usage.prompt_tokens,
            "cached_tokens": usage.cached_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cost_usd": f"{cost:.8f}",
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS, lineterminator="\n")
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
            self._total += cost
            if not self._warned and self._total >= self.warn_usd:
                self._warned = True
                logger.warning("LLM spend passed the warning level: $%.4f of $%.2f", self._total, self.hard_usd)
        return cost
