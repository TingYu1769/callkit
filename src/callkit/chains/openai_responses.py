"""Concatenation, parent inference, and stale reasoning counting for OpenAI Responses.

Concatenation is stateless replay: the previous ``input`` plus the previous response's ``output`` items are
copied verbatim (including encrypted reasoning, ``call_id``, ``phase``, id, status) + tool results + optional
user text. When copying, only the server-echoed ``metadata`` and ``internal_chat_message_metadata_passthrough``
are removed (Codex replay also removes them).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from callkit.chains.common import ToolResult, body_of
from callkit.errors import CallkitError, PairingError
from callkit.record import Record

# calls for the caller to execute and backfill
CLIENT_CALLS = ("function_call", "custom_tool_call", "tool_search_call")
OUTPUT_TYPES = {
    "function_call": "function_call_output",
    "custom_tool_call": "custom_tool_call_output",
}
# server-echoed turn information, removed on replay
ECHOED = ("metadata", "internal_chat_message_metadata_passthrough")
# fields removed during prefix comparison (recursive)
VOLATILE = {"id", "status", "annotations", "logprobs", *ECHOED}


def pending_calls(rec: Record) -> list[str]:
    output = (rec.message or {}).get("output") or []
    return [item["call_id"] for item in output if _is_client_call(item)]


def extend(
    rec: Record,
    *,
    tool_results: Mapping[str, str | ToolResult] | None = None,
    text: str | None = None,
    set_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The previous input + the response's output items + tool results (in call order) + user text. Other fields are copied as-is."""
    before = body_of(rec)
    if rec.message is None:
        raise CallkitError(f"record {rec.id} has no assembled response; cannot extend it")
    if before.get("previous_response_id"):
        raise CallkitError(
            "the request uses previous_response_id (server-side state), which stateless extend cannot continue; build the body yourself with fork"
        )
    output: list[dict[str, Any]] = rec.message.get("output") or []
    calls = [item for item in output if _is_client_call(item)]
    _check_pairing(calls, dict(tool_results or {}), text)
    results = dict(tool_results or {})
    replay = [{k: v for k, v in copy.deepcopy(item).items() if k not in ECHOED} for item in output]
    outputs = [_tool_output(call, results[call["call_id"]]) for call in calls]
    user = [_user_message(text)] if text else []
    after = {**before, "input": [*_as_items(before.get("input")), *replay, *outputs, *user]}
    after.update(set_params or {})
    return after


def infer_parent(body: dict[str, Any], prior: Sequence[Record]) -> tuple[str | None, list[str]]:
    """Candidates have the same instructions and tools, and their input + their output items, after removing
    volatile fields, form a prefix of this input; choose the longest. On ties, return (None, candidate id list)."""
    target = normalize_items(_as_items(body.get("input")))
    key = _context_key(body)
    best: list[str] = []
    best_len = -1
    for p in prior:
        if not p.body or p.message is None:
            continue
        pb = json.loads(p.body)
        if _context_key(pb) != key:
            continue
        base = normalize_items([*_as_items(pb.get("input")), *(p.message.get("output") or [])])
        if len(base) > len(target) or target[: len(base)] != base:
            continue
        if len(base) > best_len:
            best, best_len = [p.id], len(base)
        elif len(base) == best_len:
            best.append(p.id)
    if len(best) == 1:
        return best[0], []
    return None, best


def stale_reasoning(before: dict[str, Any], after: dict[str, Any]) -> int:
    """When the model changes, encrypted reasoning can only be reused within the same model family, so count all reasoning items; other changes do not count."""
    if before.get("model") == after.get("model"):
        return 0
    return sum(1 for item in _as_items(after.get("input")) if item.get("type") == "reasoning")


def normalize_items(items: Sequence[dict[str, Any]]) -> list[str]:
    """Compare items as a whole after removing fields known to vary (id, status, annotations, logprobs, server-echoed metadata)."""
    return [_project(item) for item in items]


def _check_pairing(calls: list[dict[str, Any]], results: dict[str, Any], text: str | None) -> None:
    unsupported = [c["call_id"] for c in calls if c.get("type") not in OUTPUT_TYPES]
    if unsupported:
        raise CallkitError(
            f"{unsupported} are client-executed tool_search_call items that need a tool list back, which extend does not support yet; build the body yourself with fork"
        )
    ids = [c["call_id"] for c in calls]
    missing = [i for i in ids if i not in results]
    extra = [k for k in results if k not in ids]
    if missing or extra:
        raise PairingError(
            f"tool results do not match the tool calls: missing {missing}, extra {extra}"
        )
    if not ids and not text:
        raise PairingError(
            "no tool results and no text: the request would end with the model output; pass tool results or a user text"
        )


def _is_client_call(item: dict[str, Any]) -> bool:
    kind = item.get("type")
    return kind in CLIENT_CALLS and (
        kind != "tool_search_call" or item.get("execution") == "client"
    )


def _tool_output(call: dict[str, Any], result: str | ToolResult) -> dict[str, Any]:
    r = result if isinstance(result, ToolResult) else ToolResult(result)
    return {"type": OUTPUT_TYPES[call["type"]], "call_id": call["call_id"], "output": r.content}


def _user_message(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _as_items(value: Any) -> list[dict[str, Any]]:
    """A string input is equivalent to a single user message."""
    if isinstance(value, str):
        return [_user_message(value)]
    return list(value or [])


def _project(item: dict[str, Any]) -> str:
    """Compare as a whole after removing fields known to vary (comparing only key fields would treat different
    images, namespaces, and search results as equal). Messages without a type are normalized to message, and
    string content is treated as a single text segment."""
    item = dict(item)
    if "type" not in item and "role" in item:
        item["type"] = "message"
    if item.get("type") == "message" and isinstance(item.get("content"), str):
        item["content"] = [{"type": "input_text", "text": item["content"]}]
    return _dump(_strip(item))


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _context_key(body: dict[str, Any]) -> str:
    return _dump([body.get("instructions"), body.get("tools")])


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
