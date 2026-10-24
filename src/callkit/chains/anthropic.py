"""Anthropic Messages concatenation, parent inference, and stale thinking counting."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from callkit.chains.common import ToolResult, body_of, first_difference
from callkit.errors import CallkitError, PairingError
from callkit.record import Record

THINKING_BLOCKS = ("thinking", "redacted_thinking")
# changing any of these invalidates all previous thinking
PREFIX_FIELDS = ("model", "system", "tools")
DEFAULT_CALLER = {"type": "direct"}


def extend(
    rec: Record,
    *,
    tool_results: Mapping[str, str | ToolResult] | None = None,
    text: str | None = None,
    set_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Previous messages + assistant (response content copied as is) + user (tool_result first, text after)."""
    before = body_of(rec)
    if rec.message is None:
        raise CallkitError(f"record {rec.id} has no assembled response; cannot extend it")
    content: list[dict[str, Any]] = rec.message.get("content") or []
    results = dict(tool_results or {})
    _check_pairing(rec, content, results, text, set_params)
    messages = [*before["messages"], {"role": "assistant", "content": copy.deepcopy(content)}]
    if rec.message.get("stop_reason") != "pause_turn":
        user = [
            _tool_result(b["id"], results[b["id"]]) for b in content if b.get("type") == "tool_use"
        ]
        if text:
            user.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": user})
    after = {**before, "messages": messages}
    after.update(set_params or {})
    return after


def pending_calls(rec: Record) -> list[str]:
    return [
        b["id"] for b in (rec.message or {}).get("content") or [] if b.get("type") == "tool_use"
    ]


def normalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalization before prefix comparison: drop cache_control and the default caller; treat string content as a single text block."""
    return [_normalize_message(m) for m in messages]


def infer_parent(body: dict[str, Any], prior: Sequence[Record]) -> tuple[str | None, list[str]]:
    """Find the parent in ``prior``: system and tools must match, and its request messages + response content,
    normalized, must be a prefix of this request's messages. Take the longest. If several tie, return (None, candidate ids)."""
    target = normalize_messages(body.get("messages") or [])
    key = _context_key(body)
    best: list[str] = []
    best_len = -1
    for p in prior:
        if not p.body or p.message is None:
            continue
        pb = json.loads(p.body)
        if _context_key(pb) != key:
            continue
        reply = {"role": "assistant", "content": p.message.get("content") or []}
        base = normalize_messages([*(pb.get("messages") or []), reply])
        if len(base) > len(target) or target[: len(base)] != base:
            continue
        if len(base) > best_len:
            best, best_len = [p.id], len(base)
        elif len(base) == best_len:
            best.append(p.id)
    if len(best) == 1:
        return best[0], []
    return None, best


def stale_thinking(
    before: dict[str, Any], after: dict[str, Any], reply: list[dict[str, Any]] | None = None
) -> int:
    """How many thinking blocks in ``after`` were generated with a prefix different from the current one and may now be stale.

    ``before`` is the original request body, and ``reply`` is its response content (treated as part of before when concatenating).
    If any of model, system, or tools changes, every block counts; otherwise count from the first differing message. Only counts; the request is not modified.
    """
    base = [*(before.get("messages") or [])]
    if reply is not None:
        base.append({"role": "assistant", "content": reply})
    msgs = after.get("messages") or []
    if any(_norm(before.get(k)) != _norm(after.get(k)) for k in PREFIX_FIELDS):
        start = 0
    else:
        start = first_difference(normalize_messages(base), normalize_messages(msgs))
    return sum(_count_thinking(m) for m in msgs[start:])


def _check_pairing(
    rec: Record,
    content: list[dict[str, Any]],
    results: dict[str, str | ToolResult],
    text: str | None,
    set_params: Mapping[str, Any] | None,
) -> None:
    stop = rec.message.get("stop_reason") if rec.message else None
    if stop == "pause_turn":
        if results or text:
            raise PairingError(
                "a pause_turn reply can only be sent back as is; do not add tool results or text"
            )
        return
    ids = [b["id"] for b in content if b.get("type") == "tool_use"]
    missing = [i for i in ids if i not in results]
    extra = [k for k in results if k not in ids]
    if missing or extra:
        raise PairingError(
            f"tool results do not match the tool calls: missing {missing}, extra {extra}"
        )
    if not ids and not text:
        raise PairingError(
            "no tool results and no text: the request would end with an assistant message (prefill), which newer models reject with 400"
        )
    if _pending_server_tools(content) and (text or "tools" in (set_params or {})):
        raise PairingError(
            "this turn has server tool calls without results: the next user message may only carry tool_result blocks, and tools must not change"
        )


def _pending_server_tools(content: list[dict[str, Any]]) -> list[str]:
    calls = [b["id"] for b in content if b.get("type") == "server_tool_use"]
    done = {
        b.get("tool_use_id") for b in content if str(b.get("type", "")).endswith("_tool_result")
    }
    return [c for c in calls if c not in done]


def _tool_result(tool_use_id: str, result: str | ToolResult) -> dict[str, Any]:
    r = result if isinstance(result, ToolResult) else ToolResult(result)
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": r.content,
    }
    if r.is_error:
        block["is_error"] = True
    return block


def _normalize_message(m: dict[str, Any]) -> dict[str, Any]:
    content = m.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return {"role": m.get("role"), "content": _norm(content)}


def _norm(value: Any) -> Any:
    """Recursively drop cache_control; treat a string system as a single text block."""
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return _strip(value)


def _strip(value: Any) -> Any:
    """Remove cache_control and caller values equal to the default (clients often omit it on replay)."""
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if not _ignored(k, v)}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _ignored(key: str, value: Any) -> bool:
    # Endpoints return caller: {"type": "direct"} on tool_use; Claude Code drops it on replay.
    # The default and absence mean the same, so ignore it during prefix comparison. Non-default caller values (such as code execution calls) still participate.
    return key == "cache_control" or (key == "caller" and value == DEFAULT_CALLER)


def _context_key(body: dict[str, Any]) -> str:
    return json.dumps([_norm(body.get("system")), _norm(body.get("tools"))], sort_keys=True)


def _count_thinking(message: dict[str, Any]) -> int:
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") in THINKING_BLOCKS)
