"""Message assembly and parent inference for OpenAI Chat Completions.

``extend``: previous messages + this assistant message (``choices[0]``; when multiple responses exist, only index 0 is used) + one
tool message per tool call + optional user text. The assistant message is copied as-is except for three cleanups (``replay_message``): drop null fields,
drop ``annotations`` (citations in the response, not a request parameter), and keep only ``{"id": ...}`` for ``audio``. Inference fields from compatible services (such as DeepSeek's
``reasoning_content`` that must be passed back unchanged in tool turns) and the deprecated but legal ``function_call`` are preserved.

Before comparing for parent inference, messages on both sides go through the same cleanup, and string content is treated as one text segment.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from callkit.chains.common import ToolResult, body_of
from callkit.errors import CallkitError, PairingError
from callkit.record import Record


def pending_calls(rec: Record) -> list[str]:
    return [tc["id"] for tc in _first_message(rec).get("tool_calls") or [] if tc.get("id")]


def replay_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Assistant message for replay: drop null, drop annotations, keep only the audio id, and leave everything else unchanged."""
    out: dict[str, Any] = {}
    for k, v in message.items():
        if v is None or k == "annotations":
            continue
        if k == "audio" and isinstance(v, dict):
            if v.get("id"):
                out[k] = {"id": v["id"]}
            continue
        out[k] = copy.deepcopy(v)
    return out


def extend(
    rec: Record,
    *,
    tool_results: Mapping[str, str | ToolResult] | None = None,
    text: str | None = None,
    set_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    before = body_of(rec)
    if rec.message is None:
        raise CallkitError(f"record {rec.id} has no assembled response; cannot extend it")
    message = _first_message(rec)
    ids = pending_calls(rec)
    results = dict(tool_results or {})
    missing = [i for i in ids if i not in results]
    extra = [k for k in results if k not in ids]
    if missing or extra:
        raise PairingError(
            f"tool results do not match the tool calls: missing {missing}, extra {extra}"
        )
    if not ids and not text:
        raise PairingError(
            "no tool results and no text: the request would end with an assistant message; pass tool results or a user text"
        )
    tools = [_tool_message(i, results[i]) for i in ids]
    user = [{"role": "user", "content": text}] if text else []
    after = {
        **before,
        "messages": [*(before.get("messages") or []), replay_message(message), *tools, *user],
    }
    after.update(set_params or {})
    return after


def infer_parent(body: dict[str, Any], prior: Sequence[Record]) -> tuple[str | None, list[str]]:
    """Candidates must have the same tools, and after cleanup their messages plus their assistant message must be a prefix of the current messages. Pick the longest."""
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
        base = normalize_messages([*(pb.get("messages") or []), _first_message(p)])
        if len(base) > len(target) or target[: len(base)] != base:
            continue
        if len(base) > best_len:
            best, best_len = [p.id], len(base)
        elif len(base) == best_len:
            best.append(p.id)
    if len(best) == 1:
        return best[0], []
    return None, best


def normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    return [_project(m) for m in messages]


def _project(message: Mapping[str, Any]) -> str:
    m = replay_message(message)
    if isinstance(m.get("content"), str):
        m["content"] = [{"type": "text", "text": m["content"]}]
    return json.dumps(m, sort_keys=True, ensure_ascii=False)


def _tool_message(call_id: str, result: str | ToolResult) -> dict[str, Any]:
    r = result if isinstance(result, ToolResult) else ToolResult(result)
    return {"role": "tool", "tool_call_id": call_id, "content": r.content}


def _first_message(rec: Record) -> dict[str, Any]:
    choices = (rec.message or {}).get("choices") or [{}]
    return choices[0].get("message") or {}


def _context_key(body: Mapping[str, Any]) -> str:
    return json.dumps(
        [body.get("tools"), body.get("functions")], sort_keys=True, ensure_ascii=False
    )
