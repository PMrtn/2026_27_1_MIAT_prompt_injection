"""Small text helpers shared by the simulated environment, the outcome checkers and the guardrails (R3)."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Pragmatic address pattern: finds addresses inside display names ("Name <a@b.example>"), "mailto:" links and
# free text joined by commas, semicolons, "and" or whitespace.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def extract_emails(value: Any) -> list[str]:
    """Return every email address in a string or (nested) list/tuple/set: lower-cased, de-duplicated, in order."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        found: list[str] = []
        for item in value:
            found.extend(extract_emails(item))
    else:
        found = [match.lower() for match in EMAIL_RE.findall(str(value))]
    return list(dict.fromkeys(found))
