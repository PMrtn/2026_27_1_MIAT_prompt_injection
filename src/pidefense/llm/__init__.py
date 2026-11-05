"""LLM access: Azure OpenAI client with cost ledger, hard budget and exact-request cache."""

from pidefense.llm.azure_client import AzureChatClient, LLMClient, LLMResponse
from pidefense.llm.cache import ResponseCache
from pidefense.llm.cost import BudgetExceededError, CostLedger, TokenUsage

__all__ = [
    "AzureChatClient",
    "BudgetExceededError",
    "CostLedger",
    "LLMClient",
    "LLMResponse",
    "ResponseCache",
    "TokenUsage",
]
