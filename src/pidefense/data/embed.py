"""Insert an attack payload into a benign carrier text (indirect injection), at a position and with a wrapper."""

from __future__ import annotations

import logging
import random
import re

logger = logging.getLogger(__name__)

WRAPPERS: tuple[str, ...] = ("plain", "html_comment", "quote", "signature", "hidden_span")
WRAPPER_WEIGHTS: tuple[float, ...] = (0.5, 0.125, 0.125, 0.125, 0.125)
POSITIONS: tuple[str, ...] = ("start", "middle", "end")
POSITION_WEIGHTS: tuple[float, ...] = (0.2, 0.4, 0.4)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def wrap(payload: str, wrapper: str) -> str:
    """Format the payload the way an attacker might hide or style it inside content."""
    payload = payload.strip("\n")
    if wrapper == "plain":
        return payload
    if wrapper == "html_comment":
        return f"<!-- {payload} -->"
    if wrapper == "quote":
        return "\n".join(f"> {line}" for line in payload.splitlines())
    if wrapper == "signature":
        return f"--\n{payload}"
    if wrapper == "hidden_span":
        return f'<span style="display:none">{payload}</span>'
    raise ValueError(f"Unknown wrapper {wrapper!r}")


def _segments(text: str) -> tuple[list[str], str]:
    """Split into paragraphs if possible, otherwise into sentences; return the pieces and the joiner."""
    paragraphs = [part for part in text.split("\n\n") if part.strip()]
    if len(paragraphs) >= 2:
        return paragraphs, "\n\n"
    sentences = [part for part in _SENTENCE_END.split(text) if part.strip()]
    return sentences, " "


def insert(carrier: str, block: str, position: str, rng: random.Random) -> str:
    """Place ``block`` at the start, a middle boundary, or the end of ``carrier``."""
    if position == "start":
        return f"{block}\n\n{carrier}"
    if position == "end":
        return f"{carrier}\n\n{block}"
    if position == "middle":
        pieces, joiner = _segments(carrier)
        if len(pieces) < 2:
            return f"{carrier}\n\n{block}"
        cut = rng.randint(1, len(pieces) - 1)
        return f"{joiner.join(pieces[:cut])}\n\n{block}\n\n{joiner.join(pieces[cut:])}"
    raise ValueError(f"Unknown position {position!r}")


def embed(carrier: str, payload: str, position: str, wrapper: str, rng: random.Random) -> str:
    """Wrap and insert a payload into a carrier text."""
    return insert(carrier, wrap(payload, wrapper), position, rng)


def choose_position_and_wrapper(rng: random.Random, family: str) -> tuple[str, str]:
    """Random placement; boundary-spoofing payloads (U1) only make sense unwrapped at the middle or end."""
    if family == "U1":
        return rng.choice(("middle", "end", "end")), "plain"
    position = rng.choices(POSITIONS, POSITION_WEIGHTS)[0]
    wrapper = rng.choices(WRAPPERS, WRAPPER_WEIGHTS)[0]
    return position, wrapper
