"""Logging setup shared by all command-line entry points."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once; the level defaults to $PIDEFENSE_LOG_LEVEL or INFO."""
    resolved = (level or os.environ.get("PIDEFENSE_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=resolved, format=_FORMAT)
    # Third-party clients are chatty at INFO (one line per HTTP request).
    for noisy in ("httpx", "httpx2", "httpcore", "azure", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.debug("Logging configured at %s", resolved)
