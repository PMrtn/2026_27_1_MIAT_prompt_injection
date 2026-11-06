"""Client-side token-rate limiter: keeps the project's usage under a tokens-per-minute cap.

Azure enforces a per-deployment TPM quota (HTTP 429 when exceeded). Long multi-turn episodes with many tools can
exceed it quickly under concurrency, and the deployment may be shared with other users. Each request reserves an
estimated token count in a sliding 60-second window before it is sent; the reservation is corrected to the actual
usage afterwards.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class Reservation:
    """Tokens booked in the window for one request."""

    __slots__ = ("timestamp", "tokens")

    def __init__(self, timestamp: float, tokens: int) -> None:
        self.timestamp = timestamp
        self.tokens = tokens


class TokenRateLimiter:
    """Sliding-window limiter. ``acquire`` waits until the estimate fits under ``tokens_per_minute``."""

    def __init__(self, tokens_per_minute: int, window_s: float = 60.0) -> None:
        self.tokens_per_minute = tokens_per_minute
        self.window_s = window_s
        self._events: list[Reservation] = []
        self._lock = asyncio.Lock()
        self.waited_s = 0.0

    def _used(self, now: float) -> int:
        self._events = [event for event in self._events if now - event.timestamp < self.window_s]
        return sum(event.tokens for event in self._events)

    async def acquire(self, estimate: int) -> Reservation:
        """Reserve ``estimate`` tokens, sleeping until the window has room (a single oversized request may pass)."""
        async with self._lock:
            while True:
                now = time.monotonic()
                used = self._used(now)
                if used + estimate <= self.tokens_per_minute or not self._events:
                    reservation = Reservation(now, estimate)
                    self._events.append(reservation)
                    return reservation
                wait = max(0.05, self._events[0].timestamp + self.window_s - now)
                logger.debug(
                    "TPM window full (%d + %d > %d); waiting %.1fs", used, estimate, self.tokens_per_minute, wait
                )
                self.waited_s += wait
                await asyncio.sleep(wait)

    @staticmethod
    def settle(reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate by the actual token usage reported by the API."""
        if actual_tokens > 0:
            reservation.tokens = actual_tokens


def estimate_tokens(request_json_chars: int, max_completion_tokens: int | None) -> int:
    """Conservative token estimate: ~3 characters per token for the request plus a quarter of the output cap."""
    return int(request_json_chars / 3) + int(0.25 * (max_completion_tokens or 1000))
