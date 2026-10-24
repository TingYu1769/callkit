"""Anthropic Messages adapter (the only implementation in version 1)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from callkit.adapters.base import HttpRequest, Judgement, bad_json, build_request
from callkit.config import Endpoint
from callkit.record import Outcome, Usage
from callkit.sse import SseEvent

_TOOL_BLOCKS = {"tool_use", "server_tool_use", "mcp_tool_use"}
# delta type -> (field to append in the block, field in the delta)
_APPEND_DELTAS = {
    "text_delta": ("text", "text"),
    "thinking_delta": ("thinking", "thinking"),
    "signature_delta": ("signature", "signature"),
}


class AnthropicMessages:
    """Failures: no message_stop; an error event mid-stream; assembly mismatch (a block starts
    without stopping, tool_use input is not valid JSON); thinking signature empty. When
    stop_reason is max_tokens and the last tool_use input is incomplete, record success with
    flag truncated_tool_input. refusal, max_tokens, and pause_turn are all recorded as success.
    """

    api = "anthropic.messages"
    path = "/v1/messages"  # default path, appended after the endpoint base_url

    def http_request(
        self,
        body: bytes,
        endpoint: Endpoint,
        key: str,
        *,
        path: str | None = None,
        inherit_headers: Mapping[str, str] | None = None,
    ) -> HttpRequest:
        return build_request(
            body,
            endpoint,
            key,
            path=path or self.path,
            default_auth="x-api-key",
            fixed_headers={"anthropic-version": endpoint.anthropic_version},
            inherit_headers=inherit_headers,
        )

    def assemble(self, events: Sequence[SseEvent]) -> dict[str, Any] | None:
        return _assemble(events).message

    def judge(
        self,
        status: int | None,
        transport_end: str,
        events: Sequence[SseEvent],
        message: dict[str, Any] | None,
        error_body: str | None,
        request: Mapping[str, Any] | None = None,
    ) -> Judgement:
        if status is None:
            return _failed(transport_end, "failed", transport_end, True, error_body)
        if status != 200:
            reason = _error_type(error_body) or f"http_{status}"
            retryable = status in (408, 429) or status >= 500
            return _failed(transport_end, "failed", reason, retryable, (error_body or "")[:500])
        a = _assemble(events)
        if a.errors:
            first = a.errors[0]
            detail = json.dumps(first, ensure_ascii=False)[:500]
            return _failed(transport_end, "failed", str(first.get("type", "error")), True, detail)
        if not a.stopped or a.message is None:
            return _failed(
                transport_end,
                "incomplete",
                "no_message_stop",
                True,
                "stream ended without message_stop",
            )
        return _judge_content(transport_end, a)

    def stream_defaults(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return {"stream": True}

    def from_json(
        self,
        status: int | None,
        transport_end: str,
        text: str,
        request: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Judgement, Usage | None]:
        """Non-streaming response: ``type: "message"`` is the answer, ``type: "error"`` is an error."""
        if status != 200:
            return None, self.judge(status, transport_end, [], None, text, request), None
        body = _json(text)
        if body is None:
            return None, bad_json(transport_end), None
        if body.get("type") != "message":
            nested = body.get("error")
            err: dict[str, Any] = nested if isinstance(nested, dict) else body
            reason = str(err.get("type") or "unexpected_body")
            retryable = reason in ("overloaded_error", "api_error", "rate_limit_error")
            return None, _failed(transport_end, "failed", reason, retryable, text[:500]), None
        judgement = _judge_content(transport_end, _Assembly(message=body, stopped=True))
        raw = body.get("usage")
        return body, judgement, (_to_usage(raw) if isinstance(raw, dict) and raw else None)

    def usage(self, events: Sequence[SseEvent]) -> Usage | None:
        """Start with the usage in message_start, then overwrite with the accumulated usage from each message_delta."""
        raw: dict[str, Any] | None = None
        for ev in events:
            d = _json(ev.data)
            if d is None:
                continue
            if d.get("type") == "message_start":
                raw = dict((d.get("message") or {}).get("usage") or {})
            elif d.get("type") == "message_delta" and raw is not None and d.get("usage"):
                raw.update(d["usage"])
        # No usage fields means no usage, not all zeros.
        return _to_usage(raw) if raw else None


@dataclass
class _Assembly:
    message: dict[str, Any] | None = None
    open_blocks: set[int] = field(default_factory=set)
    partial_json: dict[int, str] = field(default_factory=dict)
    bad_json: list[int] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)  # error events mid-stream
    stopped: bool = False  # received message_stop


def _assemble(events: Sequence[SseEvent]) -> _Assembly:
    a = _Assembly()
    for ev in events:
        d = _json(ev.data)
        if d is None:
            continue  # non-JSON data lines are not assembled; raw text is stored separately
        handler = _HANDLERS.get(str(d.get("type") or ev.event))
        if handler is not None:
            handler(a, d)
    return a


def _on_message_start(a: _Assembly, d: dict[str, Any]) -> None:
    a.message = dict(d.get("message") or {})
    a.message["content"] = []


def _on_block_start(a: _Assembly, d: dict[str, Any]) -> None:
    if a.message is None:
        return
    idx, block = d["index"], dict(d["content_block"])
    content = a.message["content"]
    while len(content) <= idx:
        content.append({})
    content[idx] = block
    a.open_blocks.add(idx)
    if block.get("type") in _TOOL_BLOCKS:
        a.partial_json[idx] = ""


def _on_block_delta(a: _Assembly, d: dict[str, Any]) -> None:
    block = _block(a, d["index"])
    if block is None:
        return
    delta = d.get("delta") or {}
    kind = delta.get("type")
    if kind in _APPEND_DELTAS:
        key, src = _APPEND_DELTAS[kind]
        block[key] = block.get(key, "") + delta.get(src, "")
    elif kind == "input_json_delta":
        a.partial_json[d["index"]] = a.partial_json.get(d["index"], "") + delta["partial_json"]
    elif kind == "citations_delta":
        block.setdefault("citations", []).append(delta["citation"])


def _on_block_stop(a: _Assembly, d: dict[str, Any]) -> None:
    idx = d["index"]
    a.open_blocks.discard(idx)
    block = _block(a, idx)
    if block is None or idx not in a.partial_json:
        return
    raw = a.partial_json.pop(idx)
    try:
        block["input"] = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        block["input"] = raw  # keep the raw string; do not treat it as valid arguments
        a.bad_json.append(idx)


def _on_message_delta(a: _Assembly, d: dict[str, Any]) -> None:
    if a.message is None:
        return
    delta = d.get("delta") or {}
    for key in ("stop_reason", "stop_sequence", "stop_details"):
        if key in delta:
            a.message[key] = delta[key]
    if d.get("usage"):
        a.message.setdefault("usage", {}).update(d["usage"])


def _on_message_stop(a: _Assembly, d: dict[str, Any]) -> None:
    a.stopped = True


def _on_error(a: _Assembly, d: dict[str, Any]) -> None:
    a.errors.append(d.get("error") or d)


_HANDLERS: dict[str, Callable[[_Assembly, dict[str, Any]], None]] = {
    "message_start": _on_message_start,
    "content_block_start": _on_block_start,
    "content_block_delta": _on_block_delta,
    "content_block_stop": _on_block_stop,
    "message_delta": _on_message_delta,
    "message_stop": _on_message_stop,
    "error": _on_error,
}


def _judge_content(transport_end: str, a: _Assembly) -> Judgement:
    assert a.message is not None
    content: list[dict[str, Any]] = a.message["content"]
    stop = a.message.get("stop_reason")
    flags: list[str] = []
    problems: list[str] = []
    if a.open_blocks:
        problems.append(f"block {sorted(a.open_blocks)} started but never stopped")
    bad = a.bad_json
    if bad and stop == "max_tokens" and bad == [len(content) - 1]:
        # normal behavior when eager_input_streaming is enabled
        flags.append("truncated_tool_input")
        bad = []
    if bad:
        problems.append(f"tool input of block {bad} is not valid JSON")
    empty = [
        i for i, b in enumerate(content) if b.get("type") == "thinking" and not b.get("signature")
    ]
    if empty:
        problems.append(f"thinking block {empty} has an empty signature")
    ok = not problems
    error = None if ok else "; ".join(problems)
    outcome = Outcome(ok, transport_end, "completed", stop, error, flags)
    return Judgement(outcome, retryable=False, reason="completed" if ok else "bad_assembly")


def _failed(
    transport_end: str, protocol: str, reason: str, retryable: bool, error: str | None
) -> Judgement:
    return Judgement(Outcome(False, transport_end, protocol, None, error), retryable, reason)


def _to_usage(raw: dict[str, Any]) -> Usage:
    cc = raw.get("cache_creation") or {}
    w5, w1 = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
    by_ttl = {ttl: n for ttl, n in (("5m", w5), ("1h", w1)) if n}
    details = raw.get("output_tokens_details") or {}
    return Usage(
        input_tokens=raw.get("input_tokens") or 0,
        cache_read_tokens=raw.get("cache_read_input_tokens") or 0,
        cache_write_tokens=raw.get("cache_creation_input_tokens") or (w5 + w1),
        # empty when no per-TTL breakdown; billing uses the default TTL (5m) price
        cache_write_by_ttl=by_ttl,
        output_tokens=raw.get("output_tokens") or 0,
        thinking_tokens=details.get("thinking_tokens"),
        raw=raw,
    )


def _block(a: _Assembly, idx: int) -> dict[str, Any] | None:
    if a.message is None or idx >= len(a.message["content"]):
        return None
    return a.message["content"][idx]


def _json(data: str) -> dict[str, Any] | None:
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _error_type(body: str | None) -> str | None:
    """Error bodies from Anthropic and new-api both have the form {"error": {"type": ...}}."""
    d = _json(body or "")
    err = (d or {}).get("error")
    return err.get("type") if isinstance(err, dict) else None
