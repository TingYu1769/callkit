"""Anthropic adapter: assembly, two-level judgment, usage, and request headers."""

import json
from pathlib import Path

from callkit.adapters.anthropic import AnthropicMessages
from callkit.config import Endpoint
from callkit.sse import SseEvent, parse

DATA = Path(__file__).resolve().parents[1] / "data"
A = AnthropicMessages()


def load(name: str) -> list[SseEvent]:
    text = (DATA / name).read_text(encoding="utf-8")
    return parse((0.0, line) for line in text.split("\n"))


def edit(events, fn):
    """Edit data for each event; returning None drops the event, returning a list replaces it with multiple events."""
    out = []
    for ev in events:
        got = fn(json.loads(ev.data))
        for d in [] if got is None else got if isinstance(got, list) else [got]:
            out.append(SseEvent(d["type"], json.dumps(d), ev.t))
    return out


def judge(events, status=200, error_body=None):
    return A.judge(status, "done", events, A.assemble(events), error_body)


def test_assemble_thinking_text_tool_use():
    msg = A.assemble(load("stream_tool_use.sse"))
    assert msg is not None
    assert [b["type"] for b in msg["content"]] == ["thinking", "text", "tool_use"]
    assert msg["content"][0]["signature"]
    tool = msg["content"][2]
    assert isinstance(tool["input"], dict) and tool["input"]
    assert msg["stop_reason"] == "tool_use"


def test_good_stream_judged_ok():
    j = judge(load("stream_tool_use.sse"))
    assert j.outcome.ok and j.outcome.protocol == "completed" and not j.retryable
    assert j.outcome.stop_reason == "tool_use" and j.reason == "completed"


def test_pings_ignored_and_usage_merged():
    evs = load("stream_thinking_text.sse")
    msg = A.assemble(evs)
    assert msg is not None and [b["type"] for b in msg["content"]] == ["thinking", "text"]
    assert msg["content"][0]["thinking"] == "" and "138" in msg["content"][1]["text"]
    u = A.usage(evs)
    assert u is not None
    assert (u.input_tokens, u.output_tokens, u.thinking_tokens) == (68, 4138, 3437)


def test_missing_message_stop_is_incomplete_and_retryable():
    evs = edit(load("stream_tool_use.sse"), lambda d: None if d["type"] == "message_stop" else d)
    j = judge(evs)
    assert not j.outcome.ok and j.outcome.protocol == "incomplete"
    assert j.retryable and j.reason == "no_message_stop"


def test_error_event_after_200_is_retryable():
    err = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    evs = edit(
        load("stream_tool_use.sse"), lambda d: [err, d] if d["type"] == "message_delta" else d
    )
    j = judge(evs)
    assert not j.outcome.ok and j.outcome.protocol == "failed"
    assert j.retryable and j.reason == "overloaded_error"


def test_empty_signature_fails_without_retry():
    def blank(d):
        if d["type"] == "content_block_delta" and d["delta"]["type"] == "signature_delta":
            d["delta"]["signature"] = ""
        return d

    j = judge(edit(load("stream_tool_use.sse"), blank))
    assert not j.outcome.ok and not j.retryable and "empty signature" in (j.outcome.error or "")


def test_unclosed_block_fails():
    seen = []

    def drop_first_stop(d):
        if d["type"] == "content_block_stop" and not seen:
            seen.append(d)
            return None
        return d

    j = judge(edit(load("stream_tool_use.sse"), drop_first_stop))
    assert not j.outcome.ok and j.reason == "bad_assembly"


def truncate_tool(stop_reason):
    def fn(d):
        if d["type"] == "content_block_delta" and d["delta"]["type"] == "input_json_delta":
            d["delta"]["partial_json"] = d["delta"]["partial_json"][:10]
        if d["type"] == "message_delta":
            d["delta"]["stop_reason"] = stop_reason
        return d

    return edit(load("stream_tool_use.sse"), fn)


def test_truncated_tool_input_ok_only_with_max_tokens():
    j = judge(truncate_tool("max_tokens"))
    assert j.outcome.ok and j.outcome.flags == ["truncated_tool_input"]
    j = judge(truncate_tool("tool_use"))
    assert not j.outcome.ok and "not valid JSON" in (j.outcome.error or "")


def test_http_errors():
    body = json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "x"}})
    j = A.judge(529, "done", [], None, body)
    assert j.retryable and j.reason == "overloaded_error" and j.outcome.protocol == "failed"
    body = json.dumps({"error": {"type": "invalid_request_error", "message": "x"}})
    assert not A.judge(400, "done", [], None, body).retryable
    j = A.judge(None, "connect_error", [], None, "ConnectError")
    assert j.retryable and j.reason == "connect_error"


def test_http_request_headers():
    ep = Endpoint("gw", "http://gw:3000/", "GW_KEY", extra_headers={"X-Team": "a"})
    req = A.http_request(b"{}", ep, "sk-1")
    assert req.url == "http://gw:3000/v1/messages" and req.body == b"{}"
    assert req.headers["anthropic-version"] == "2023-06-01" and req.headers["x-api-key"] == "sk-1"
    assert req.headers["x-team"] == "a"
    inherited = {
        "anthropic-version": "2026-10-21",
        "anthropic-beta": "a,b",
        "x-api-key": "old",
        "Content-Length": "9",
    }
    req = A.http_request(
        b"{}", ep, "sk-1", path="/v1/messages?beta=true", inherit_headers=inherited
    )
    assert req.url.endswith("/v1/messages?beta=true")
    assert (
        req.headers["anthropic-version"] == "2026-10-21" and req.headers["anthropic-beta"] == "a,b"
    )
    assert req.headers["x-api-key"] == "sk-1" and "content-length" not in req.headers
    bearer = Endpoint("gw", "http://gw", "GW_KEY", auth_header="Authorization")
    assert A.http_request(b"{}", bearer, "sk-1").headers["authorization"] == "Bearer sk-1"


def test_usage_cache_write_by_ttl():
    start = {
        "type": "message_start",
        "message": {
            "id": "m",
            "content": [],
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 30,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 10,
                    "ephemeral_1h_input_tokens": 20,
                },
                "output_tokens": 1,
            },
        },
    }
    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 9},
    }
    u = A.usage(
        [
            SseEvent("message_start", json.dumps(start), 0),
            SseEvent("message_delta", json.dumps(delta), 0),
        ]
    )
    assert u is not None
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens) == (10, 100, 30)
    assert u.cache_write_by_ttl == {"5m": 10, "1h": 20} and u.output_tokens == 9
    plain = {
        "type": "message_start",
        "message": {
            "id": "m",
            "content": [],
            "usage": {"input_tokens": 1, "cache_creation_input_tokens": 5, "output_tokens": 1},
        },
    }
    u = A.usage([SseEvent("message_start", json.dumps(plain), 0)])
    assert u is not None and (u.cache_write_tokens, u.cache_write_by_ttl) == (5, {})


def test_from_json():
    msg = A.assemble(load("stream_thinking_text.sse"))
    assert msg is not None
    body, j, u = A.from_json(200, "done", json.dumps(msg))
    assert body == msg and j.outcome.ok and j.outcome.stop_reason == msg["stop_reason"]
    assert u is not None and u.output_tokens == msg["usage"]["output_tokens"]
    unsigned = {**msg, "content": [{**b, "signature": ""} for b in msg["content"]]}
    _, j, _ = A.from_json(200, "done", json.dumps(unsigned))
    assert not j.outcome.ok and "empty signature" in (j.outcome.error or "")
    body, j, _ = A.from_json(200, "eof", json.dumps(msg)[:50])
    assert body is None and j.outcome.protocol == "incomplete" and j.retryable
    assert j.reason == "bad_json" and j.outcome.transport_end == "eof"
    overloaded = json.dumps({"type": "error", "error": {"type": "overloaded_error"}})
    body, j, _ = A.from_json(200, "done", overloaded)
    assert body is None and j.retryable and j.reason == "overloaded_error"
    body, j, _ = A.from_json(529, "done", overloaded)
    assert body is None and j.retryable and j.reason == "overloaded_error"
    bad_request = json.dumps({"type": "error", "error": {"type": "invalid_request_error"}})
    _, j, _ = A.from_json(400, "done", bad_request)
    assert not j.retryable and j.reason == "invalid_request_error"


def test_stream_defaults():
    assert A.stream_defaults({}) == {"stream": True}
