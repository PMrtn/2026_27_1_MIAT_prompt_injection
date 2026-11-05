"""Exact-request response cache (JSONL), used for reproducibility and to avoid paying twice for one request.

A request that is byte-for-byte identical to an earlier one (same model, messages, tools, parameters and
namespace) returns the stored response. In the end-to-end experiment this implements common random numbers:
when a defense passes a tool output unchanged, configs 2-4 reuse the config-1 response for the same prefix.
Use a distinct ``namespace`` (e.g. the repetition index) whenever fresh samples are wanted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResponseCache:
    """Append-only JSONL map from request hash to stored response."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping corrupt cache line %d in %s", line_no, path)
                        continue
                    self._entries[record["key"]] = record["value"]
        logger.info("Response cache %s: %d entries", path, len(self._entries))

    @staticmethod
    def make_key(payload: Mapping[str, Any]) -> str:
        """Stable SHA-256 of a JSON-serialisable request description."""
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the stored value or None."""
        return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store a value and append it to the JSONL file."""
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
