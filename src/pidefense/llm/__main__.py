"""Smoke test of the Azure deployment: one tiny billed call, recorded in the cost ledger.

Usage: python -m pidefense.llm ["prompt"]
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from pidefense.llm.azure_client import AzureChatClient
from pidefense.logging_utils import setup_logging

logger = logging.getLogger(__name__)


async def _smoke(prompt: str) -> None:
    client = AzureChatClient()
    try:
        response = await client.chat([{"role": "user", "content": prompt}], purpose="smoke", max_completion_tokens=200)
    finally:
        await client.aclose()
    logger.info("Reply: %r", response.content)
    logger.info("Usage: %s | cost $%.6f | latency %.0f ms", response.usage, response.cost_usd, response.latency_ms)
    logger.info("Ledger total: $%.4f", client.ledger.total_usd)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", default="Reply with exactly: OK")
    args = parser.parse_args()
    setup_logging()
    asyncio.run(_smoke(args.prompt))


if __name__ == "__main__":
    main()
