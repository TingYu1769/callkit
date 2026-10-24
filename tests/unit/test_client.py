"""Client.call: send, record creation, disk persistence, pricing, cap, fork inheritance."""

import json

import pytest

from callkit import Client, Endpoint, Price, transport
from callkit.errors import CallkitError, CapReached, MissingPrice, StreamRequired
from tests.fakes import Server, error_response, good_stream, sse_response

EP = Endpoint("gw", "http://gw:3000", "CALLKIT_TEST_KEY")
PRICE = Price(input=3, output=15, cache_read=0.3, cache_write=3.75, cache_write_by_ttl={"1h": 6})
BODY = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("CALLKIT_TEST_KEY", "sk-test-123")
    monkeypatch.setattr(transport.time, "sleep", lambda s: None)
    s = Server()
    monkeypatch.setattr(transport, "open_client", lambda endpoint: s.client())
    return s


def test_call_writes_record(server, tmp_path):
    server.replies.append(sse_response(good_stream()))
    with Client(EP, tmp_path / "run", prices={"claude-sonnet-5-5": PRICE}, tags={"exp": "t"}) as c:
        rec = c.call(BODY, tags={"case": "1"})
    sent = server.requests[0]
    assert sent.headers["x-api-key"] == "sk-test-123" and sent.url == "http://gw:3000/v1/messages"
    assert json.loads(sent.content)["stream"] is True
    assert (
        rec.outcome.ok
        and rec.outcome.stop_reason == "tool_use"
        and rec.cost_usd
        and rec.cost_usd > 0
    )
    assert rec.request.headers["x-api-key"] == "<redacted>" and rec.tags == {
        "exp": "t",
        "case": "1",
    }
    assert rec.message is not None and rec.response.message_id == rec.message["id"]
    back = c.store.load(rec.id)
    assert back.body == sent.content and back.to_dict() == rec.to_dict()
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["endpoint"]["name"] == "gw" and "callkit_commit" in run
    assert "sk-test-123" not in (tmp_path / "run" / "calls" / rec.id / "meta.json").read_text()


def test_bytes_body_sent_unchanged(server, tmp_path):
    server.replies.append(sse_response(good_stream()))
    raw = b'{ "stream": true,  "model": "m", "max_tokens": 1, "messages": [] }'
    rec = Client(EP, tmp_path / "run").call(raw)
    assert server.requests[0].content == raw and rec.body == raw and rec.cost_usd is None


def test_stream_false_rejected(server, tmp_path):
    with pytest.raises(StreamRequired):
        Client(EP, tmp_path / "run").call({**BODY, "stream": False})


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("CALLKIT_TEST_KEY", raising=False)
    with pytest.raises(CallkitError, match="CALLKIT_TEST_KEY"):
        Client(EP, tmp_path / "run")


def test_cap(server, tmp_path):
    with pytest.raises(MissingPrice):
        Client(EP, tmp_path / "a", cap_usd=1.0).call(BODY)
    server.replies.append(sse_response(good_stream()))
    c = Client(EP, tmp_path / "b", prices={"claude-sonnet-5-5": PRICE}, cap_usd=1e-9)
    c.call(BODY)
    with pytest.raises(CapReached):
        c.call(BODY)
    assert len(server.requests) == 1


def test_retry_recorded_and_failed_attempts_billed(server, tmp_path):
    server.replies += [
        error_response(529, "overloaded_error", retry_after="0"),
        sse_response(good_stream()),
    ]
    rec = Client(EP, tmp_path / "run", prices={"claude-sonnet-5-5": PRICE}).call(BODY)
    assert [(a.decision, a.reason) for a in rec.attempts] == [
        ("retry", "overloaded_error"),
        ("stop", "completed"),
    ]
    assert rec.attempts[0].cost_usd == 0.0 and rec.cost_usd == rec.attempts[1].cost_usd
    assert (tmp_path / "run" / "calls" / rec.id / "attempts" / "1.body").exists()


def test_failure_does_not_raise(server, tmp_path):
    server.replies.append(error_response(400, "invalid_request_error"))
    rec = Client(EP, tmp_path / "run").call(BODY)
    assert not rec.outcome.ok and rec.response.status == 400 and rec.message is None


def test_fork_inherits_path_and_headers(server, tmp_path):
    server.replies += [sse_response(good_stream())] * 2
    c = Client(EP, tmp_path / "run")
    base = c.call(BODY)
    # treat as a Claude Code call recorded by the proxy
    base.request.url = "http://gw:3000/v1/messages?beta=true"
    base.request.headers["anthropic-beta"] = "claude-code-20250219,effort-2025-11-24"
    rec = c.call({**BODY, "model": "claude-opus-5-5", "stream": True}, fork_of=base)
    sent = server.requests[1]
    assert sent.url.query == b"beta=true"
    assert sent.headers["anthropic-beta"] == "claude-code-20250219,effort-2025-11-24"
    assert sent.headers["x-api-key"] == "sk-test-123"
    assert rec.fork_of == base.id and rec.edits == ["model"]
    assert rec.request.sha256 != base.request.sha256


def test_custom_auth_header_redacted_and_not_inherited(server, tmp_path):
    server.replies += [sse_response(good_stream())] * 2
    ep = Endpoint("gw", "http://gw:3000", "CALLKIT_TEST_KEY", auth_header="X-Gw")
    base = Client(ep, tmp_path / "run").call(BODY)
    assert server.requests[0].headers["x-gw"] == "Bearer sk-test-123"
    meta = (tmp_path / "run" / "calls" / base.id / "meta.json").read_text()
    assert "sk-test-123" not in meta and base.request.headers["x-gw"] == "<redacted>"
    # switch back to the default auth-header endpoint
    Client(EP, tmp_path / "run").call(BODY, fork_of=base)
    sent = server.requests[1]
    assert "x-gw" not in sent.headers and sent.headers["x-api-key"] == "sk-test-123"


def test_fork_flags_stale_thinking(server, tmp_path):
    from callkit import extend, fork

    server.replies += [sse_response(good_stream())] * 3
    c = Client(EP, tmp_path / "run")
    first = c.call(BODY)
    tool_id = next(b["id"] for b in first.message["content"] if b["type"] == "tool_use")
    second = c.call(extend(first, tool_results={tool_id: "ok"}), parent=first)
    assert second.parent == first.id and second.edits == [] and second.outcome.flags == []

    def edit(b):
        b["messages"][0]["content"] = "bye"
        return b

    third = c.call(fork(second, edit=edit), fork_of=second)
    assert third.edits == ["messages"] and third.outcome.flags == ["thinking_may_be_invalid:1"]


def test_responses_endpoint(server, tmp_path):
    from callkit import pending_calls
    from tests.fakes import DATA

    server.replies.append(sse_response((DATA / "responses_tool_calls.sse").read_bytes()))
    ep = Endpoint("gw", "https://gw.example", "CALLKIT_TEST_KEY", api="openai.responses")
    price = Price(input=1.25, output=10, cache_read=0.125, cache_write=1.5625)
    body = {
        "model": "gpt-5.6-sol",
        "input": "hi",
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    rec = Client(ep, tmp_path / "run", prices={"gpt-5.6-sol": price}).call(body)
    sent = server.requests[0]
    assert (
        sent.url == "https://gw.example/v1/responses"
        and sent.headers["authorization"] == "Bearer sk-test-123"
    )
    assert (
        rec.api == "openai.responses" and rec.outcome.ok and rec.outcome.stop_reason == "completed"
    )
    assert len(pending_calls(rec)) == 4 and all(c.startswith("call_") for c in pending_calls(rec))
    assert rec.cost_usd and rec.cost_usd > 0 and rec.usage and rec.usage.thinking_tokens is not None
    import json as _json

    assert (
        _json.loads((tmp_path / "run" / "run.json").read_text())["endpoint"]["api"]
        == "openai.responses"
    )


def test_responses_extend_and_fork_flags(server, tmp_path):
    from callkit import extend, fork, pending_calls
    from tests.fakes import DATA

    stream = (DATA / "responses_tool_calls.sse").read_bytes()
    server.replies += [sse_response(stream)] * 3
    ep = Endpoint("gw", "https://gw.example", "CALLKIT_TEST_KEY", api="openai.responses")
    c = Client(ep, tmp_path / "run")
    first = c.call({"model": "gpt-5.6-sol", "input": "hi", "store": False})
    second = c.call(
        extend(first, tool_results={i: "ok" for i in pending_calls(first)}), parent=first
    )
    assert second.parent == first.id and second.outcome.flags == [] and second.edits == []
    third = c.call(fork(second, model="gpt-5.6-pro"), fork_of=second)
    assert third.edits == ["model"] and third.outcome.flags == ["reasoning_may_be_invalid:1"]


def test_chat_endpoint_adds_stream_defaults_and_chains(server, tmp_path):
    from callkit import extend, fork, pending_calls
    from tests.fakes import DATA

    stream = (DATA / "chat_tool_calls.sse").read_bytes()
    server.replies += [sse_response(stream)] * 4
    ep = Endpoint("gw", "https://gw.example", "CALLKIT_TEST_KEY", api="openai.chat_completions")
    price = Price(input=1.25, output=10, cache_read=0.125, cache_write=1.5625)
    c = Client(ep, tmp_path / "run", prices={"gpt-5.6-sol": price})
    body = {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "天气？"}]}
    first = c.call(body)
    sent = server.requests[0]
    assert sent.url == "https://gw.example/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer sk-test-123"
    assert json.loads(sent.content) == {
        **body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert first.api == "openai.chat_completions" and first.outcome.ok
    assert first.outcome.stop_reason == "tool_calls" and len(pending_calls(first)) == 2
    assert first.cost_usd and first.cost_usd > 0
    raw = b'{"model":"gpt-5.6-sol","stream":true,"messages":[]}'
    # if stream is set, send as-is
    assert c.call(raw).body == raw and server.requests[1].content == raw
    second = c.call(
        extend(first, tool_results={i: "晴" for i in pending_calls(first)}), parent=first
    )
    assert second.parent == first.id and second.edits == [] and second.outcome.flags == []
    # like a non-streaming call recorded by the SDK
    first.body = json.dumps({**body, "stream": False}).encode()
    third = c.call(fork(first), fork_of=first)
    assert sorted(third.edits) == ["stream", "stream_options"] and third.outcome.ok


def test_prefixed_endpoint_keeps_path_on_parent_and_fork(server, tmp_path):
    """When the endpoint URL has a prefix (such as DeepSeek's /anthropic), parent calls and forks do not duplicate the prefix; query strings are preserved as-is."""
    server.replies += [sse_response(good_stream())] * 3
    ep = Endpoint("ds", "https://api.example/anthropic", "CALLKIT_TEST_KEY")
    c = Client(ep, tmp_path / "run")
    first = c.call(BODY)
    c.call({**BODY, "max_tokens": 50}, parent=first)
    # treat as a Claude Code call recorded elsewhere
    first.request.url = "http://other-gw/v1/messages?beta=true"
    c.call(BODY, fork_of=first)
    urls = [str(r.url) for r in server.requests]
    assert urls[:2] == ["https://api.example/anthropic/v1/messages"] * 2
    assert urls[2] == "https://api.example/anthropic/v1/messages?beta=true"


def test_client_prices_go_into_run_dir(server, tmp_path):
    with Client(EP, tmp_path / "run", prices={"claude-sonnet-5-5": PRICE}):
        pass
    assert (
        json.loads((tmp_path / "run" / "prices.json").read_text())["claude-sonnet-5-5"]["output"]
        == 15
    )
    with pytest.raises(CallkitError, match="already has different prices"):
        Client(EP, tmp_path / "run", prices={"claude-sonnet-5-5": Price(1, 1, 1, 1)})


def test_client_ok_without_usage_is_unknown_cost(server, tmp_path):
    from tests.fakes import DATA

    stream = b"\n".join(
        line
        for line in (DATA / "chat_text.sse").read_bytes().split(b"\n")
        if b'"usage":{"completion_tokens"' not in line
    )
    server.replies.append(sse_response(stream))
    ep = Endpoint("gw", "https://gw.example", "CALLKIT_TEST_KEY", api="openai.chat_completions")
    price = Price(input=1.25, output=10, cache_read=0.125, cache_write=1.5625)
    with pytest.warns(RuntimeWarning, match="include_usage"):
        rec = Client(ep, tmp_path / "run", prices={"gpt-5.6-sol": price}).call(
            {
                "model": "gpt-5.6-sol",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    assert rec.outcome.ok and rec.cost_usd is None and rec.outcome.flags == ["usage_missing"]
