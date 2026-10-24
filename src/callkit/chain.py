"""Compose multi-turn and forked requests. Pure offline functions that return new request bodies.

Public entry points are here, dispatching by record API to ``callkit.chains.anthropic`` and ``callkit.chains.openai_responses``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from callkit.adapters import get_adapter
from callkit.chains import anthropic, openai_chat, openai_responses
from callkit.chains.common import ToolResult, body_of
from callkit.record import Record

RESPONSES = "openai.responses"
ANTHROPIC = "anthropic.messages"
CHAT = "openai.chat_completions"
_EXTEND = {
    RESPONSES: openai_responses.extend,
    CHAT: openai_chat.extend,
    ANTHROPIC: anthropic.extend,
}
_PENDING = {
    RESPONSES: openai_responses.pending_calls,
    CHAT: openai_chat.pending_calls,
    ANTHROPIC: anthropic.pending_calls,
}
_PARENT = {
    RESPONSES: openai_responses.infer_parent,
    CHAT: openai_chat.infer_parent,
    ANTHROPIC: anthropic.infer_parent,
}

__all__ = [
    "ToolResult",
    "extend",
    "fork",
    "infer_parent",
    "normalize_messages",
    "pending_calls",
    "stale_thinking",
]

# Anthropic normalization before prefix comparison; kept as a legacy entry point.
normalize_messages = anthropic.normalize_messages


def extend(
    rec: Record,
    *,
    tool_results: Mapping[str, str | ToolResult] | None = None,
    text: str | None = None,
    set_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the next request body from the previous record: previous conversation + this response (copied as-is) + tool results + optional user text.
    Other fields are copied; ``set_params`` explicitly overrides top-level fields.

    If tool results are missing or extra, or if there is neither tool results nor text, raise PairingError.
    Anthropic: tool_result goes first in the user message; if the turn ended with pause_turn, send no content to return it unchanged.
    Responses: stateless replay; tool results are written as function_call_output or custom_tool_call_output according to the call type.
    Chat: tool results become tool messages; assistant messages drop null, annotations, and keep only the id in audio.
    If the original request has stream: false, switch to streaming.
    """
    after = _EXTEND.get(rec.api, anthropic.extend)(
        rec, tool_results=tool_results, text=text, set_params=set_params
    )
    return _streamify(rec.api, after)


def pending_calls(rec: Record) -> list[str]:
    """Tool call IDs in this response that the caller must execute and fill results for (call_id for Responses, id of tool_use for Anthropic).
    Server-executed calls (such as web_search_call, server_tool_use) are not included."""
    return _PENDING.get(rec.api, anthropic.pending_calls)(rec)


def fork(
    rec: Record,
    *,
    model: str | None = None,
    edit: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a modified copy of ``rec``'s request body; send it with ``client.call(body, fork_of=rec)``. Works across all three APIs.

    Parsing and re-serializing must preserve key order (the cache is sensitive to key order); stale reasoning is kept as-is and annotated by call.
    If the original request has stream: false, switch to streaming.
    ``edit`` receives a parsed copy and returns the edited copy (it may mutate and return the same object).
    """
    after = body_of(rec)
    if model is not None:
        after["model"] = model
    after = edit(after) if edit is not None else after
    return _streamify(rec.api, after)


def infer_parent(
    body: dict[str, Any], prior: Sequence[Record], *, api: str = ANTHROPIC
) -> tuple[str | None, list[str]]:
    """Find a parent among ``prior`` records of the same API: context (system / instructions, tools) matches, and "its request +
    its response" after normalization is a prefix of this request; take the longest. If multiple tie, return (None, list of candidate ids)."""
    same = [p for p in prior if p.api == api]
    return _PARENT.get(api, anthropic.infer_parent)(body, same)


def stale_thinking(
    before: dict[str, Any],
    after: dict[str, Any],
    reply: list[dict[str, Any]] | None = None,
    *,
    api: str = ANTHROPIC,
) -> int:
    """Number of reasoning entries in ``after`` that may be stale. Anthropic: if model, system, or tools change, count all thinking blocks;
    otherwise count from the first message that differs (``reply`` is before's response and is treated as part of before when concatenating).
    Responses: count all reasoning entries only when the model changes. Only counts, does not modify the request."""
    if api == RESPONSES:
        return openai_responses.stale_reasoning(before, after)
    if api == CHAT:
        return 0  # OpenAI Chat does not carry reasoning across turns.
    return anthropic.stale_thinking(before, after, reply)


def _streamify(api: str, body: dict[str, Any]) -> dict[str, Any]:
    """If the original request has stream: false (common in SDK-recorded non-streaming calls), switch to streaming: callkit only sends
    streaming requests and fills per-API defaults (such as Chat's ``include_usage``). The change appears in the new record's ``edits``."""
    if body.get("stream") is not False:
        return body
    body["stream"] = True
    for k, v in get_adapter(api).stream_defaults(body).items():
        body.setdefault(k, v)
    return body
