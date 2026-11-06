"""Token-rate limiter (offline, uses a short window)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import time

from pidefense.llm.ratelimit import TokenRateLimiter, estimate_tokens


async def test_waits_when_window_is_full() -> None:
    limiter = TokenRateLimiter(tokens_per_minute=100, window_s=0.3)
    first = await limiter.acquire(80)
    start = time.monotonic()
    await limiter.acquire(50)
    assert time.monotonic() - start >= 0.2
    assert first.tokens == 80


async def test_settle_frees_capacity() -> None:
    limiter = TokenRateLimiter(tokens_per_minute=100, window_s=5.0)
    reservation = await limiter.acquire(90)
    TokenRateLimiter.settle(reservation, 10)
    start = time.monotonic()
    await asyncio.wait_for(limiter.acquire(80), timeout=1.0)
    assert time.monotonic() - start < 0.5


async def test_oversized_request_passes_when_window_empty() -> None:
    limiter = TokenRateLimiter(tokens_per_minute=10, window_s=5.0)
    await asyncio.wait_for(limiter.acquire(500), timeout=1.0)


def test_estimate_is_conservative() -> None:
    assert estimate_tokens(3000, 1000) == 1250
    assert estimate_tokens(0, None) == 250
