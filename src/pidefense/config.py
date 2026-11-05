"""Central configuration: Azure endpoint, model pricing and the hard LLM budget."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelPricing:
    """Token prices in USD per 1M tokens."""

    input_per_m: float
    cached_input_per_m: float
    cache_write_per_m: float
    output_per_m: float


# Azure Retail Prices API (prices.azure.com), swedencentral, Global Standard, short context, queried 2026-11-05.
GPT_56_LUNA_PRICING = ModelPricing(input_per_m=0.20, cached_input_per_m=0.02, cache_write_per_m=0.25, output_per_m=1.20)
# gpt-4.1 has no separate cache-write meter; cache-write tokens (if ever reported) are billed as normal input.
GPT_41_PRICING = ModelPricing(input_per_m=2.00, cached_input_per_m=0.50, cache_write_per_m=2.00, output_per_m=8.00)
PRICING_BY_DEPLOYMENT: dict[str, ModelPricing] = {"gpt-5.6-luna": GPT_56_LUNA_PRICING, "gpt-4.1": GPT_41_PRICING}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r, using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class LLMSettings:
    """Settings for every Azure OpenAI call made by this project (data generation and agent runs)."""

    endpoint: str = "https://00dept-openai.openai.azure.com/openai/v1/"
    deployment: str = "gpt-5.6-luna"
    token_scope: str = "https://cognitiveservices.azure.com/.default"
    pricing: ModelPricing = GPT_56_LUNA_PRICING
    budget_warn_usd: float = 5.0
    budget_hard_usd: float = 9.5
    ledger_path: Path = REPO_ROOT / "results" / "llm_cost_ledger.csv"
    max_retries: int = 6
    timeout_s: float = 180.0

    @classmethod
    def from_env(cls) -> LLMSettings:
        """Build settings, allowing overrides through PIDEFENSE_* environment variables."""
        defaults = cls()
        deployment = os.environ.get("PIDEFENSE_DEPLOYMENT", defaults.deployment)
        if deployment not in PRICING_BY_DEPLOYMENT:
            raise ValueError(f"No pricing configured for deployment {deployment!r}; add it to PRICING_BY_DEPLOYMENT")
        return cls(
            endpoint=os.environ.get("PIDEFENSE_AZURE_ENDPOINT", defaults.endpoint),
            deployment=deployment,
            pricing=PRICING_BY_DEPLOYMENT[deployment],
            budget_warn_usd=_env_float("PIDEFENSE_BUDGET_WARN_USD", defaults.budget_warn_usd),
            budget_hard_usd=_env_float("PIDEFENSE_BUDGET_HARD_USD", defaults.budget_hard_usd),
            ledger_path=Path(os.environ.get("PIDEFENSE_LEDGER_PATH", str(defaults.ledger_path))),
        )
