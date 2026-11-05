"""Conversion of MCP tool descriptions into OpenAI function-calling schemas."""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

from mcp import types

logger = logging.getLogger(__name__)


def _strip_titles(schema: Any) -> Any:
    """Drop pydantic's auto-generated 'title' keys: they add tokens and carry no information."""
    if isinstance(schema, dict):
        return {key: _strip_titles(value) for key, value in schema.items() if key != "title"}
    if isinstance(schema, list):
        return [_strip_titles(item) for item in schema]
    return schema


def mcp_tools_to_openai(tools: Iterable[types.Tool]) -> list[dict[str, Any]]:
    """Map MCP tools to the ``tools`` parameter of chat completions, preserving server order."""
    converted = []
    for tool in tools:
        parameters = _strip_titles(copy.deepcopy(tool.input_schema)) or {"type": "object", "properties": {}}
        converted.append(
            {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description or "", "parameters": parameters},
            }
        )
    logger.debug("Converted %d MCP tools", len(converted))
    return converted
