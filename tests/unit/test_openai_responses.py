"""OpenAI Responses adapter: assembly, judgment, usage, request headers. Sample streams from a real relay gateway."""

import json
from pathlib import Path

import pytest

from callkit.adapters import get_adapter
from callkit.adapters.openai_responses import OpenAIResponses
from callkit.config import Endpoint
from callkit.sse import SseEvent, parse

DATA = Path(__file__).resolve().parents[1] / "data"
A = OpenAIResponses()


def load(name="responses_tool_calls.sse"):
    text = (DATA / name).read_text(encoding="utf-8")
    return parse((0.0, line) for line in text.split("\n"))


def edit(events, fn):
    out = []
    for ev in events:
        got = fn(json.loads(ev.data))
        for d in [] if got is None else got if isinstance(got, list) else [got]:
            out.append(SseEvent(d["type"], json.dumps(d), ev.t))
    return out


def judge(events, status=200, error_body=None, request=None):
    return A.judge(status, "done", events, A.assemble(events), error_body, request)


def to_incomplete(d):
    if d["type"] == "response.completed":
        d["type"] = "response.incomplete"
        d["response"]["status"] = "incomplete"
        d["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    return d


def test_assemble_from_done_items():
    resp = A.assemble(load())
    assert resp is not None
    assert [i["type"] for i in resp["output"]] == ["reasoning", "message"] + ["function_call"] * 4
    assert resp["output"][0]["encrypted_content"] and resp["status"] == "completed"


def test_output_rebuilt_when_terminal_output_empty():
    def blank(d):
        if d["type"] == "response.completed":
            d["response"]["output"] = []
        return d

    resp = A.assemble(edit(load(), blank))
    assert resp is not None and len(resp["output"]) == 6


def test_good_stream_ok():
    j = judge(load())
    assert (
        j.outcome.ok and j.outcome.protocol == "completed" and j.outcome.stop_reason == "completed"
    )
    assert not j.retryable and j.reason == "completed"


def test_incomplete_is_model_truncation_not_stream_cut():
    j = judge(edit(load(), to_incomplete))
    assert (
        j.outcome.ok
        and j.outcome.protocol == "completed"
        and j.outcome.stop_reason == "max_output_tokens"
    )


def truncate_last_call(d):
    if d["type"] == "response.output_item.done" and d["output_index"] == 5:
        d["item"]["arguments"] = '{"city":"上'
    return d


def test_truncated_arguments_ok_only_when_cut_by_limit():
    j = judge(edit(edit(load(), truncate_last_call), to_incomplete))
    assert j.outcome.ok and j.outcome.flags == ["truncated_tool_input"]
    j = judge(edit(load(), truncate_last_call))
    assert (
        not j.outcome.ok
        and j.reason == "bad_assembly"
        and "not valid JSON" in (j.outcome.error or "")
    )


def test_no_terminal_event_is_retryable():
    j = judge(edit(load(), lambda d: None if d["type"] == "response.completed" else d))
    assert not j.outcome.ok and j.outcome.protocol == "incomplete" and j.retryable
    assert j.reason == "no_terminal_event"


@pytest.mark.parametrize(("code", "retryable"), [("server_error", True), ("invalid_prompt", False)])
def test_response_failed(code, retryable):
    def fail(d):
        if d["type"] == "response.completed":
            d["type"] = "response.failed"
            d["response"]["status"] = "failed"
            d["response"]["error"] = {"code": code, "message": "x"}
        return d

    j = judge(edit(load(), fail))
    assert (
        not j.outcome.ok
        and j.outcome.protocol == "failed"
        and j.retryable is retryable
        and j.reason == code
    )


def test_error_event_mid_stream():
    err = {"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}
    j = judge(edit(load(), lambda d: [err, d] if d["type"] == "response.completed" else d))
    assert not j.outcome.ok and j.retryable and j.reason == "rate_limit_exceeded"


def drop_done(d):
    return None if d["type"] == "response.output_item.done" and d["output_index"] == 2 else d


def test_item_only_in_terminal_is_kept():
    """An output that appears only in the terminal state without output_item.done must still be kept."""
    resp = A.assemble(edit(load(), drop_done))
    assert (
        resp is not None
        and len(resp["output"]) == 6
        and resp["output"][2]["type"] == "function_call"
    )
    assert judge(edit(load(), drop_done)).outcome.ok


def test_done_version_preferred_over_terminal():
    def change_terminal(d):
        if d["type"] == "response.completed":
            d["response"]["output"][0]["encrypted_content"] = "terminal-version"
        return d

    resp = A.assemble(edit(load(), change_terminal))
    assert resp is not None and resp["output"][0]["encrypted_content"] != "terminal-version"


def test_added_without_done_and_not_in_terminal_fails():
    def drop_both(d):
        if d["type"] == "response.completed":
            d["response"]["output"] = d["response"]["output"][:2]
        return drop_done(d)

    j = judge(edit(load(), drop_both))
    assert not j.outcome.ok and "neither done" in (j.outcome.error or "")


def test_missing_encrypted_reasoning_fails_only_if_requested():
    def strip(d):
        if d["type"] == "response.output_item.done" and d["item"]["type"] == "reasoning":
            d["item"].pop("encrypted_content")
        return d

    events = edit(load(), strip)
    assert judge(events).outcome.ok
    j = judge(events, request={"include": ["reasoning.encrypted_content"]})
    assert not j.outcome.ok and "encrypted_content" in (j.outcome.error or "")


def test_http_errors():
    body = json.dumps(
        {
            "error": {
                "message": "not implemented",
                "type": "new_api_error",
                "code": "convert_request_failed",
            }
        }
    )
    j = A.judge(500, "done", [], None, body)
    assert j.retryable and j.reason == "convert_request_failed"
    j = A.judge(
        400,
        "done",
        [],
        None,
        json.dumps({"error": {"type": "invalid_request_error", "code": None}}),
    )
    assert not j.retryable and j.reason == "invalid_request_error"


def test_usage_subtracts_cache_from_input():
    u = A.usage(load("responses_reasoning_text.sse"))
    assert u is not None
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens, u.cache_write_by_ttl) == (
        3,
        0,
        3992,
        {},
    )
    assert (u.output_tokens, u.thinking_tokens) == (38, 31)

    def ttl(d):
        if d["type"] == "response.completed":
            d["response"]["usage"]["input_tokens_details"]["cache_write_tokens_details"] = {
                "ttl_30m_tokens": 3000,
                "ttl_12h_tokens": 992,
            }
        return d

    u = A.usage(edit(load("responses_reasoning_text.sse"), ttl))
    assert u is not None and u.cache_write_by_ttl == {"30m": 3000, "12h": 992}
    assert A.usage(edit(load(), lambda d: None if d["type"] == "response.completed" else d)) is None


def test_http_request():
    ep = Endpoint("gw", "https://api.example/", "K", api="openai.responses")
    req = A.http_request(b"{}", ep, "sk-1")
    assert (
        req.url == "https://api.example/v1/responses"
        and req.headers["authorization"] == "Bearer sk-1"
    )
    assert "anthropic-version" not in req.headers
    req = A.http_request(
        b"{}", ep, "sk-1", inherit_headers={"session-id": "s1", "Authorization": "Bearer old"}
    )
    assert req.headers["session-id"] == "s1" and req.headers["authorization"] == "Bearer sk-1"
    xk = Endpoint("gw", "https://api.example", "K", api="openai.responses", auth_header="x-api-key")
    assert A.http_request(b"{}", xk, "sk-1").headers["x-api-key"] == "sk-1"


def test_registry():
    assert isinstance(get_adapter("openai.responses"), OpenAIResponses)
    with pytest.raises(ValueError, match="openai.responses"):
        get_adapter("google.gemini")


def test_from_json():
    resp = A.assemble(load())
    assert resp is not None
    body, j, u = A.from_json(200, "done", json.dumps(resp))
    assert body == resp and j.outcome.ok and j.outcome.stop_reason == "completed"
    assert u is not None and u.output_tokens == resp["usage"]["output_tokens"]
    cut = {**resp, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    _, j, _ = A.from_json(200, "done", json.dumps(cut))
    assert j.outcome.ok and j.outcome.stop_reason == "max_output_tokens"
    failed = {**resp, "status": "failed", "error": {"code": "server_error", "message": "x"}}
    body, j, _ = A.from_json(200, "done", json.dumps(failed))
    assert body == failed and not j.outcome.ok and j.retryable and j.reason == "server_error"
    bad = {**resp, "output": [{**resp["output"][2], "arguments": '{"city":'}]}
    _, j, _ = A.from_json(200, "done", json.dumps(bad))
    assert not j.outcome.ok and j.reason == "bad_assembly"
    lacking = {**resp, "output": [{**resp["output"][0], "encrypted_content": None}]}
    request = {"include": ["reasoning.encrypted_content"]}
    _, j, _ = A.from_json(200, "done", json.dumps(lacking), request)
    assert not j.outcome.ok and "encrypted_content" in (j.outcome.error or "")
    body, j, _ = A.from_json(200, "eof", json.dumps(resp)[:80])
    assert body is None and j.outcome.protocol == "incomplete" and j.retryable
    err = json.dumps({"error": {"code": "invalid_prompt", "message": "x"}})
    body, j, _ = A.from_json(200, "done", err)
    assert body is None and not j.retryable and j.reason == "invalid_prompt"
    body, j, _ = A.from_json(500, "done", err)
    assert body is None and j.retryable


def test_stream_defaults():
    assert A.stream_defaults({}) == {"stream": True}


def test_background_response_not_finished():
    """A non-streaming POST with background=True returning queued / in_progress means no answer yet: not success, not retryable."""
    resp = A.assemble(load())
    assert resp is not None
    for status in ("queued", "in_progress"):
        body = {**resp, "status": status, "output": [], "background": True}
        _, j, _ = A.from_json(200, "done", json.dumps(body))
        assert not j.outcome.ok and not j.retryable and j.reason == "not_finished"
        assert j.outcome.stop_reason == status
