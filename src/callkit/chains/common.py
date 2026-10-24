"""Shared utilities for both interfaces: tool results, getting request bodies, and finding the first difference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from callkit.errors import CallkitError
from callkit.record import Record


@dataclass(frozen=True)
class ToolResult:
    # A string, or a list of content blocks such as text / image
    content: str | list[dict[str, Any]]
    is_error: bool = (
        # Anthropic writes it as tool_result.is_error; Responses has no such field and uses content only
        False
    )


def body_of(rec: Record) -> dict[str, Any]:
    """Parse a fresh request body each time; callers can modify it freely without affecting the record, and key order stays unchanged."""
    if not rec.body:
        raise CallkitError(
            f"record {rec.id} has no request body (load it from disk with store.load)"
        )
    return json.loads(rec.body)


def first_difference(a: list[Any], b: list[Any]) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))
