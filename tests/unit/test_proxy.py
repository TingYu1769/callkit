"""Recording proxy: client -> proxy -> fake upstream, all over real local sockets."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from callkit.proxy import Proxy
from callkit.store import RunStore
from tests.fakes import good_stream

KEY = "sk-proxy-test"
SESSION = "0b6c2f6e-1d2a-4c3b-9e8f-5a7d6c4b3a21"


class Upstream:
    """Fake upstream: /v1/messages replays the sample stream in two chunks (20 ms pause in the middle), /v1/models returns JSON, everything else returns 404."""

    def __init__(self, stream=None, json_reply=None):
        self.stream = stream if stream is not None else good_stream()
        self.json_reply = json_reply
        self.seen: list[tuple[str, str, dict, bytes]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append(("GET", self.path, dict(self.headers), b""))
                self._send(200, "application/json", b'{"data": []}')

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                outer.seen.append(("POST", self.path, dict(self.headers), body))
                if outer.json_reply is not None:
                    self._send(200, "application/json", json.dumps(outer.json_reply).encode())
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(outer.stream)))
                self.end_headers()
                half = len(outer.stream) // 2
                self.wfile.write(outer.stream[:half])
                self.wfile.flush()
                time.sleep(0.02)
                self.wfile.write(outer.stream[half:])

            def _send(self, status, ctype, data):
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def setup(tmp_path):
    made = []

    def factory(upstream=None, upstream_url=None):
        up = upstream or Upstream()
        proxy = Proxy(upstream_url or up.url, tmp_path / "run")
        proxy.start()
        made.append((up, proxy))
        return up, proxy

    yield factory
    for up, proxy in made:
        proxy.stop()
        up.close()


def post(proxy, body: dict, session=SESSION, path="/v1/messages?beta=true"):
    data = json.dumps(body, ensure_ascii=False).encode()
    headers = {
        "x-api-key": KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "claude-code-20250219",
    }
    if session:
        headers["X-Claude-Code-Session-Id"] = session
    with httpx.Client(trust_env=False) as c:
        return c.post(proxy.url + path, content=data, headers=headers), data


BODY = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 100,
    "stream": True,
    "system": "s",
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
}


def test_bytes_pass_through_and_record(setup):
    up, proxy = setup()
    resp, sent = post(proxy, BODY)
    assert resp.status_code == 200 and resp.content == good_stream()
    _method, path, headers, body = up.seen[0]
    assert path == "/v1/messages?beta=true" and body == sent
    assert headers["x-api-key"] == KEY and headers["anthropic-beta"] == "claude-code-20250219"
    assert {k.lower(): v for k, v in headers.items()}["accept-encoding"] == "identity"
    (row,) = proxy.store.index()
    rec = proxy.store.load(row.id)
    assert rec.source == "proxy" and rec.outcome.ok and rec.body == sent
    assert rec.request.url == up.url + "/v1/messages?beta=true" and rec.tags == {"session": SESSION}
    assert rec.request.headers["x-api-key"] == "<redacted>"
    assert (proxy.store.calls / rec.id / "response.sse").read_bytes() == good_stream()


def test_parent_inferred_within_session(setup):
    _up, proxy = setup()
    post(proxy, BODY)
    first = proxy.store.load(proxy.store.index()[0].id)
    assert first.message
    tool_id = next(b["id"] for b in first.message["content"] if b["type"] == "tool_use")
    nxt = {
        **BODY,
        "messages": [
            *BODY["messages"],
            {"role": "assistant", "content": first.message["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
            },
        ],
    }
    post(proxy, nxt)
    post(proxy, nxt, session="another-session")
    rows = proxy.store.index()
    assert [r.parent for r in rows] == [None, first.id, None]


def test_unrecorded_paths_pass_through(setup):
    up, proxy = setup()
    with httpx.Client(trust_env=False) as c:
        r = c.get(proxy.url + "/v1/models", headers={"x-api-key": KEY})
    assert r.status_code == 200 and r.json() == {"data": []}
    assert up.seen[0][:2] == ("GET", "/v1/models") and proxy.store.index() == []


def test_non_streaming_json_response_recorded(setup):
    reply = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "好"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    _up, proxy = setup(Upstream(json_reply=reply))
    resp, _ = post(proxy, {**BODY, "stream": False})
    assert resp.json() == reply
    rec = proxy.store.load(proxy.store.index()[0].id)
    assert rec.outcome.ok and rec.outcome.stop_reason == "end_turn" and rec.message == reply
    assert (proxy.store.calls / rec.id / "response.body").exists()


def test_upstream_down_gives_502_and_is_recorded(setup, tmp_path):
    dead = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    url = f"http://127.0.0.1:{dead.server_address[1]}"
    dead.server_close()  # Port is closed, so connections are refused.
    _, proxy = setup(upstream_url=url)
    resp, _ = post(proxy, BODY)
    assert resp.status_code == 502 and resp.json()["error"]["type"] == "api_error"
    rec = RunStore(tmp_path / "run").load(proxy.store.index()[0].id)
    assert not rec.outcome.ok and rec.outcome.transport_end == "connect_error"


def codex_headers(session="019a-session"):
    meta = {"request_kind": "turn", "turn_id": "t1"}
    return (
        {
            "authorization": f"Bearer {KEY}",
            "session-id": session,
            "thread-id": session,
            "originator": "codex_exec",
        },
        {"x-codex-turn-metadata": json.dumps(meta), "session_id": session},
    )


def post_responses(proxy, body: dict, headers: dict):
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    with httpx.Client(trust_env=False) as c:
        return c.post(proxy.url + "/v1/responses", content=data, headers=headers), data


def test_codex_responses_recorded_with_tags_and_parent(setup):
    from tests.fakes import DATA

    up, proxy = setup(Upstream(stream=(DATA / "responses_tool_calls.sse").read_bytes()))
    headers, meta = codex_headers()
    first_body = {
        "model": "gpt-5.6-sol",
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "查天气"}],
            }
        ],
        "store": False,
        "client_metadata": meta,
    }
    resp, sent = post_responses(proxy, first_body, headers)
    assert resp.status_code == 200 and up.seen[0][1] == "/v1/responses" and up.seen[0][3] == sent
    first = proxy.store.load(proxy.store.index()[0].id)
    assert first.api == "openai.responses" and first.outcome.ok and first.message
    assert first.tags == {
        "session": "019a-session",
        "thread_id": "019a-session",
        "request_kind": "turn",
    }
    outputs = [
        {"type": "function_call_output", "call_id": i["call_id"], "output": "晴"}
        for i in first.message["output"]
        if i["type"] == "function_call"
    ]
    replay = [{k: v for k, v in i.items() if k != "metadata"} for i in first.message["output"]]
    second_body = {**first_body, "input": [*first_body["input"], *replay, *outputs]}
    post_responses(proxy, second_body, headers)
    post_responses(proxy, second_body, codex_headers(session="other")[0])
    assert [r.parent for r in proxy.store.index()] == [None, first.id, None]


def test_websocket_upgrade_gets_426(setup):
    up, proxy = setup()
    with httpx.Client(trust_env=False) as c:
        r = c.get(
            proxy.url + "/v1/responses", headers={"upgrade": "websocket", "connection": "Upgrade"}
        )
    assert r.status_code == 426 and up.seen == [] and proxy.store.index() == []


def test_prompt_cache_key_is_session_fallback(setup):
    from tests.fakes import DATA

    _up, proxy = setup(Upstream(stream=(DATA / "responses_tool_calls.sse").read_bytes()))
    body = {"model": "gpt-5.6-sol", "input": "hi", "stream": True, "prompt_cache_key": "pck-1"}
    post_responses(proxy, body, {"authorization": f"Bearer {KEY}"})
    rec = proxy.store.load(proxy.store.index()[0].id)
    assert rec.tags == {"session": "pck-1"}


def post_chat(proxy, body: dict, session="sess-chat", path="/v1/chat/completions"):
    data = json.dumps(body, ensure_ascii=False).encode()
    headers = {"authorization": f"Bearer {KEY}"}
    if session:
        headers["session-id"] = session
    with httpx.Client(trust_env=False) as c:
        return c.post(proxy.url + path, content=data, headers=headers), data


def chat_upstream(json_reply=None):
    from tests.fakes import DATA

    if json_reply is not None:
        return Upstream(json_reply=json_reply)
    return Upstream(stream=(DATA / "chat_tool_calls.sse").read_bytes())


CHAT_BODY = {
    "model": "gpt-5.6-sol",
    "stream": True,
    "stream_options": {"include_usage": True},
    "messages": [{"role": "user", "content": "北京和上海今天天气怎么样？"}],
}


def chat_next(first, body=CHAT_BODY):
    raw = first.message["choices"][0]["message"]
    tools = [
        {"role": "tool", "tool_call_id": tc["id"], "content": "晴"} for tc in raw["tool_calls"]
    ]
    return {**body, "messages": [*body["messages"], raw, *tools]}


def test_chat_stream_recorded_with_parent_within_session(setup):
    up, proxy = setup(chat_upstream())
    resp, sent = post_chat(proxy, CHAT_BODY)
    assert resp.status_code == 200 and up.seen[0][1] == "/v1/chat/completions"
    assert up.seen[0][3] == sent
    first = proxy.store.load(proxy.store.index()[0].id)
    assert first.api == "openai.chat_completions" and first.outcome.ok and first.message
    assert first.outcome.stop_reason == "tool_calls" and first.usage and first.usage.output_tokens
    assert first.tags == {"session": "sess-chat"}
    assert first.response.message_id == first.message["id"]
    nxt = chat_next(first)
    post_chat(proxy, nxt)
    post_chat(proxy, nxt, session="other")
    assert [r.parent for r in proxy.store.index()] == [None, first.id, None]


def test_chat_json_on_prefixed_path(setup):
    from tests.fakes import DATA

    reply = json.loads((DATA / "chat_tool_calls.json").read_text(encoding="utf-8"))
    up, proxy = setup(chat_upstream(json_reply=reply))
    path = "/compatible-mode/v1/chat/completions"
    resp, _ = post_chat(proxy, {**CHAT_BODY, "stream": False}, path=path)
    assert resp.json() == reply and up.seen[0][1] == path
    rec = proxy.store.load(proxy.store.index()[0].id)
    assert rec.api == "openai.chat_completions" and rec.outcome.ok and rec.message == reply
    assert rec.outcome.stop_reason == "tool_calls" and rec.usage and rec.usage.input_tokens == 139


def test_no_session_key_records_candidates_not_parent(setup):
    _up, proxy = setup(chat_upstream())
    post_chat(proxy, CHAT_BODY, session=None)
    first = proxy.store.load(proxy.store.index()[0].id)
    post_chat(proxy, chat_next(first), session=None)
    second = proxy.store.load(proxy.store.index()[1].id)
    assert first.tags == {} and second.parent is None and second.parent_candidates == [first.id]


def test_threads_paths_not_recorded(setup):
    up, proxy = setup(chat_upstream())
    post_chat(proxy, {"role": "user", "content": "x"}, path="/v1/threads/t1/messages")
    assert up.seen[0][1] == "/v1/threads/t1/messages" and proxy.store.index() == []


def test_proxy_shares_recorder_with_same_run_dir(setup, tmp_path):
    from callkit.recorder import Recorder

    _up, proxy = setup()
    # SDK writes to the same run directory without ID collisions.
    assert proxy.recorder is Recorder.shared(tmp_path / "run")


def test_recording_failure_still_relays(setup, monkeypatch):
    _up, proxy = setup()

    def boom(*a, **k):
        raise OSError("磁盘满了")

    monkeypatch.setattr(proxy.store, "write", boom)
    with pytest.warns(RuntimeWarning, match="callkit failed to record"):
        resp, _ = post(proxy, BODY)
    assert resp.status_code == 200 and resp.content == good_stream()


class BrokenUpstream(Upstream):
    """Sends the response headers and half of the body, then disconnects."""

    def __init__(self):
        outer = self
        self.seen = []

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                outer.seen.append(("POST", self.path, dict(self.headers), body))
                data = good_stream()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data[: len(data) // 2])
                self.wfile.flush()
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_broken_upstream_closes_downstream(setup):
    _up, proxy = setup(BrokenUpstream())
    data = json.dumps(BODY).encode()
    t0 = time.monotonic()
    with httpx.Client(trust_env=False, timeout=10) as c, pytest.raises(httpx.RemoteProtocolError):
        c.post(proxy.url + "/v1/messages", content=data, headers={"x-api-key": KEY})
    assert time.monotonic() - t0 < 5  # does not wait for the timeout
    rec = proxy.store.load(proxy.store.index()[0].id)
    assert not rec.outcome.ok and rec.outcome.transport_end == "eof"


def test_proxy_prices_and_cap(tmp_path):
    from callkit import Price

    up = Upstream()
    price = {"claude-sonnet-5-5": Price(3, 15, 0.3, 3.75)}
    proxy = Proxy(up.url, tmp_path / "run", prices=price, cap_usd=1e-9)
    proxy.start()
    try:
        first, _ = post(proxy, BODY)
        refused, _ = post(proxy, BODY)
    finally:
        proxy.stop()
        up.close()
    assert first.status_code == 200
    assert refused.status_code == 402 and refused.headers["x-should-retry"] == "false"
    assert refused.json()["error"]["type"] == "callkit_cap_reached" and len(up.seen) == 1
    (row,) = proxy.store.index()
    assert row.cost_usd and row.cost_usd > 0
