"""OpenAI Responses adapter (Codex interface). Only performs offline conversion.

Assembly: take the ``response`` object from the terminal event (completed / incomplete / failed).
Rebuild its ``output`` by ``output_index`` from each ``output_item.done`` (Codex replays this exact
copy), and include items present in the terminal event but missing from done. Without a terminal
event, use ``response.created`` / ``in_progress`` as a base to assemble a partial result with
completed items.

Judgement: no terminal event means the stream did not finish (retryable); ``response.failed`` and
``error`` events are failures, and retry only when the error code is in ``RETRYABLE_CODES`` (the
``error`` event is checked before the missing terminal event); ``status=incomplete`` means the
response was not fully written (possible reasons include max_output_tokens, max_messages,
content_filter, steered), which is a stop by the model or server. Record success and keep the stop
reason as ``incomplete_details.reason``. When fully written, record ``"completed"``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from callkit.adapters.base import RETRYABLE_CODES, HttpRequest, Judgement, bad_json, build_request
from callkit.config import Endpoint
from callkit.record import Outcome, Usage
from callkit.sse import SseEvent

TERMINAL = ("response.completed", "response.incomplete", "response.failed")
SNAPSHOT = ("response.created", "response.in_progress", "response.queued")
ENCRYPTED = "reasoning.encrypted_content"


class OpenAIResponses:
    api = "openai.responses"
    path = "/v1/responses"  # Default path, appended to the endpoint base_url.

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
            default_auth="authorization",
            fixed_headers={},
            inherit_headers=inherit_headers,
        )

    def assemble(self, events: Sequence[SseEvent]) -> dict[str, Any] | None:
        return _assemble(events).response

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
            reason = _error_code(_json(error_body or "") or {}) or f"http_{status}"
            retryable = status in (408, 429) or status >= 500
            return _failed(transport_end, "failed", reason, retryable, (error_body or "")[:500])
        a = _assemble(events)
        if a.errors:
            return _error_judgement(transport_end, a.errors[0])
        if a.terminal is None or a.response is None:
            return _failed(
                transport_end,
                "incomplete",
                "no_terminal_event",
                True,
                "stream ended without a terminal event",
            )
        if a.terminal == "response.failed":
            return _error_judgement(
                transport_end, a.response.get("error") or {"code": "response_failed"}
            )
        return _judge_output(transport_end, a, request)

    def stream_defaults(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return {"stream": True}

    def from_json(
        self,
        status: int | None,
        transport_end: str,
        text: str,
        request: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Judgement, Usage | None]:
        """Non-streaming response: ``object: "response"`` is the answer; judge it by status with the same rules as streaming terminal events."""
        if status != 200:
            return None, self.judge(status, transport_end, [], None, text, request), None
        body = _json(text)
        if body is None:
            return None, bad_json(transport_end), None
        nested = body.get("error")
        if body.get("object") != "response":
            err: dict[str, Any] = (
                nested if isinstance(nested, dict) else {"code": "unexpected_body"}
            )
            return None, _error_judgement(transport_end, err), None
        usage = _to_usage(body["usage"]) if isinstance(body.get("usage"), dict) else None
        if body.get("status") == "failed":
            err = nested if isinstance(nested, dict) else {"code": "response_failed"}
            return body, _error_judgement(transport_end, err), usage
        a = _Assembly(response=body, terminal="response.completed")
        a.done = dict(enumerate(body.get("output") or []))
        a.final = set(a.done)
        return body, _judge_output(transport_end, a, request), usage

    def usage(self, events: Sequence[SseEvent]) -> Usage | None:
        """Take usage only from the terminal event's ``response.usage``: Responses does not emit usage mid-stream, so it is unavailable when the stream is cut off."""
        a = _assemble(events)
        raw = (a.response or {}).get("usage") if a.terminal else None
        return _to_usage(raw) if isinstance(raw, dict) else None


@dataclass
class _Assembly:
    response: dict[str, Any] | None = None
    terminal: str | None = None
    added: set[int] = field(default_factory=set)
    done: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Which positions are in the assembled output (from done plus terminal fallback).
    final: set[int] = field(default_factory=set)
    errors: list[dict[str, Any]] = field(default_factory=list)


def _assemble(events: Sequence[SseEvent]) -> _Assembly:
    a = _Assembly()
    snapshot: dict[str, Any] | None = None
    for ev in events:
        d = _json(ev.data)
        if d is None:
            continue
        kind = str(d.get("type") or ev.event)
        if kind in SNAPSHOT and isinstance(d.get("response"), dict):
            snapshot = d["response"]
        elif kind == "response.output_item.added":
            a.added.add(int(d.get("output_index", -1)))
        elif kind == "response.output_item.done" and isinstance(d.get("item"), dict):
            a.done[int(d.get("output_index", len(a.done)))] = d["item"]
        elif kind in TERMINAL and isinstance(d.get("response"), dict):
            a.terminal, snapshot = kind, d["response"]
        elif kind == "error":
            nested = d.get("error")
            a.errors.append(nested if isinstance(nested, dict) else d)
    if snapshot is not None:
        # Use each output_item.done as the source of truth (Codex replays this copy; encrypted
        # content for the same reasoning item in the terminal event may be re-encrypted, so bytes
        # can differ). Include terminal items missing from done so nothing is lost.
        merged = dict(enumerate(snapshot.get("output") or [])) if a.terminal else {}
        merged.update(a.done)
        a.final = set(merged)
        a.response = {**snapshot, "output": [merged[i] for i in sorted(merged)]}
    return a


def _judge_output(transport_end: str, a: _Assembly, request: Mapping[str, Any] | None) -> Judgement:
    assert a.response is not None
    resp = a.response
    # Background task was just created and has no answer yet.
    if resp.get("status") in ("queued", "in_progress"):
        detail = f"response is not finished yet (status={resp.get('status')}); no answer"
        outcome = Outcome(False, transport_end, "completed", resp.get("status"), detail)
        return Judgement(outcome, retryable=False, reason="not_finished")
    output: list[dict[str, Any]] = resp["output"]
    incomplete = resp.get("status") == "incomplete"
    stop = (
        ((resp.get("incomplete_details") or {}).get("reason") or "incomplete")
        if incomplete
        else "completed"
    )
    flags: list[str] = []
    problems: list[str] = []
    missing = sorted(a.added - a.final)
    if missing:
        problems.append(f"items {missing} were added but neither done nor in the terminal response")
    bad = [
        i
        for i, item in enumerate(output)
        if item.get("type") == "function_call" and not _is_json(item.get("arguments"))
    ]
    if bad and stop == "max_output_tokens" and bad == [len(output) - 1]:
        # The last call's arguments were cut off at the limit; this is expected.
        flags.append("truncated_tool_input")
        bad = []
    if bad:
        problems.append(f"function arguments of items {bad} are not valid JSON")
    if request is not None and ENCRYPTED in (request.get("include") or []):
        lacking = [
            i
            for i, item in enumerate(output)
            if item.get("type") == "reasoning" and not item.get("encrypted_content")
        ]
        if lacking:
            problems.append(
                f"request asked for encrypted reasoning but items {lacking} have no encrypted_content"
            )
    ok = not problems
    outcome = Outcome(
        ok, transport_end, "completed", stop, None if ok else "; ".join(problems), flags
    )
    return Judgement(outcome, retryable=False, reason="completed" if ok else "bad_assembly")


def _error_judgement(transport_end: str, err: Mapping[str, Any]) -> Judgement:
    code = _error_code(err) or "error"
    detail = json.dumps(err, ensure_ascii=False)[:500]
    return _failed(transport_end, "failed", code, code in RETRYABLE_CODES, detail)


def _failed(
    transport_end: str, protocol: str, reason: str, retryable: bool, error: str | None
) -> Judgement:
    return Judgement(Outcome(False, transport_end, protocol, None, error), retryable, reason)


def _to_usage(raw: dict[str, Any]) -> Usage:
    """Responses input_tokens includes cached reads and writes; subtract them to get the portion that did not use cache."""
    details = raw.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    write = (
        details.get("cache_write_tokens")
        or details.get("cache_creation_tokens")
        or raw.get("cache_write_tokens")
        or 0
    )
    ttl = details.get("cache_write_tokens_details") or raw.get("cache_write_tokens_details") or {}
    by_ttl = {
        label: n
        for label, n in (("30m", ttl.get("ttl_30m_tokens")), ("12h", ttl.get("ttl_12h_tokens")))
        if n
    }
    return Usage(
        input_tokens=max((raw.get("input_tokens") or 0) - cached - write, 0),
        cache_read_tokens=cached,
        cache_write_tokens=write,
        cache_write_by_ttl=by_ttl,
        output_tokens=raw.get("output_tokens") or 0,
        thinking_tokens=(raw.get("output_tokens_details") or {}).get("reasoning_tokens"),
        raw=raw,
    )


def _error_code(err: Mapping[str, Any]) -> str | None:
    nested = err.get("error")
    inner: Mapping[str, Any] = nested if isinstance(nested, dict) else err
    value = inner.get("code") or inner.get("type")
    return str(value) if value else None


def _is_json(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True


def _json(data: str) -> dict[str, Any] | None:
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
