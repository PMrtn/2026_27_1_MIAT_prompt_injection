"""Benign carrier texts authored by Claude (the project's coding agent), never by the evaluated model.

The texts live in ``data/generated/claude/*.txt`` as delimited items::

    <<<ITEM kind=email lang=en hard_negative=false topic=business travel>>>
    From: ...
    <<<END>>>

Detector carriers (office-style kinds) train and test the detector; ops kinds form the out-of-domain test set only.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pidefense.attacks.goals import ATTACKER_DOMAIN, ATTACKER_EMAIL, ATTACKER_IBAN, MISINFO_CODE, normalize_iban
from pidefense.config import REPO_ROOT

logger = logging.getLogger(__name__)

AUTHORED_DIR = REPO_ROOT / "data" / "generated" / "claude"
OPS_KINDS = frozenset(
    {
        "ops_support_ticket",
        "runbook",
        "log_excerpt",
        "monitoring_alert",
        "crm_note",
        "vendor_status",
        "config_file",
        "change_request",
        "incident_chat",
    }
)
MIN_WORDS, MAX_WORDS = 15, 400

_ITEM = re.compile(r"<<<ITEM\s+(?P<attrs>.*?)>>>[ \t]*\r?\n(?P<text>.*?)\r?\n?[ \t]*<<<END>>>", re.S)
_ATTR = re.compile(r"(kind|lang|hard_negative|topic)=(.*?)(?=\s+(?:kind|lang|hard_negative|topic)=|$)")


@dataclass(frozen=True)
class AuthoredCarrier:
    """One authored benign text."""

    carrier_id: str
    carrier_type: str
    lang: str
    topic: str
    hard_negative: bool
    text: str
    generator: str = "claude"
    source_file: str = ""


def _has_canary(text: str) -> bool:
    lowered = text.lower()
    return (
        ATTACKER_DOMAIN in lowered
        or ATTACKER_EMAIL in lowered
        or MISINFO_CODE.lower() in lowered
        or normalize_iban(ATTACKER_IBAN) in normalize_iban(text)
    )


def parse_file(path: Path) -> tuple[list[AuthoredCarrier], collections.Counter[str]]:
    """Parse one file; return the valid items and a counter of rejection reasons."""
    rejected: collections.Counter[str] = collections.Counter()
    items: list[AuthoredCarrier] = []
    for match in _ITEM.finditer(path.read_text(encoding="utf-8")):
        attrs = {key: value.strip() for key, value in _ATTR.findall(match.group("attrs"))}
        text = match.group("text").strip()
        kind, lang = attrs.get("kind", ""), attrs.get("lang", "")
        words = len(text.split())
        if not kind or lang not in ("en", "hu"):
            rejected["bad_header"] += 1
            continue
        if words < MIN_WORDS or words > MAX_WORDS:
            rejected["length"] += 1
            continue
        if _has_canary(text):
            rejected["canary"] += 1
            continue
        digest = hashlib.sha1(f"{kind}|{lang}|{text}".encode("utf-8")).hexdigest()[:12]
        items.append(
            AuthoredCarrier(
                carrier_id=f"cl-{digest}",
                carrier_type=kind,
                lang=lang,
                topic=attrs.get("topic", ""),
                hard_negative=attrs.get("hard_negative", "false").lower() == "true",
                text=text,
                source_file=path.name,
            )
        )
    return items, rejected


def load_authored(ops: bool | None = None, directory: Path = AUTHORED_DIR) -> list[AuthoredCarrier]:
    """All authored carriers (deduplicated). ``ops=True`` -> ops kinds only, ``False`` -> detector kinds only."""
    seen: set[str] = set()
    carriers: list[AuthoredCarrier] = []
    rejected: collections.Counter[str] = collections.Counter()
    for path in sorted(directory.glob("*.txt")):
        items, file_rejected = parse_file(path)
        rejected.update(file_rejected)
        for item in items:
            key = re.sub(r"\s+", " ", item.text.lower())
            if key in seen:
                rejected["duplicate"] += 1
                continue
            if ops is not None and (item.carrier_type in OPS_KINDS) != ops:
                continue
            seen.add(key)
            carriers.append(item)
    logger.info("Loaded %d authored carriers (ops=%s); rejected %s", len(carriers), ops, dict(rejected))
    return carriers
