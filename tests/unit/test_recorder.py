"""Recorder: records or skips based on selection, session key, tags, sharing, non-streaming and streaming all work. No network; feed Exchange directly."""

import json

import pytest

from callkit.recorder import Exchange, Recorder, recorded_api
from tests.fakes import DATA

CHAT_BODY = {
    "model": "gpt-5.6-sol",
    "messages": [{"role": "user", "content": "北京和上海今天天气怎么样？"}],
}


def chat_exchange(headers=None, body=None, stream=False, url="https://gw/v1/chat/completions"):
    data = json.dumps(body or CHAT_BODY, ensure_ascii=False).encode()
    ex = Exchange("POST", url, {"Authorization": "Bearer sk-x", **(headers or {})}, data)
    ex.status = 200
    if stream:
        ex.response_headers = {"Content-Type": "text/event-stream"}
        ex.chunks = [(0.1, (DATA / "chat_tool_calls.sse").read_text(encoding="utf-8"))]
    else:
        ex.response_headers = {"content-type": "application/json"}
        ex.chunks = [(0.1, (DATA / "chat_tool_calls.json").read_text(encoding="utf-8"))]
    return ex


@pytest.mark.parametrize(
    ("method", "url", "api"),
    [
        ("POST", "https://gw/v1/chat/completions", "openai.chat_completions"),
        ("post", "https://gw/v1/responses?x=1", "openai.responses"),
        ("POST", "https://gw/v1/messages?beta=true", "anthropic.messages"),
        ("GET", "https://gw/v1/chat/completions", None),
        ("POST", "https://gw/v1/embeddings", None),
        ("POST", "https://gw/v1/threads/t/messages", None),
    ],
)
def test_recorded_api(method, url, api):
    assert recorded_api(method, url) == api


def test_records_json_and_stream(tmp_path):
    r = Recorder(tmp_path / "run")
    rec = r.record(chat_exchange(), source="sdk")
    assert rec is not None and rec.source == "sdk" and rec.outcome.ok
    assert rec.api == "openai.chat_completions" and rec.usage and rec.usage.input_tokens == 139
    assert rec.request.headers["Authorization"] == "<redacted>"
    assert (r.store.calls / rec.id / "response.body").exists()
    rec = r.record(chat_exchange(stream=True), source="proxy")
    assert rec is not None and rec.outcome.ok and rec.outcome.stop_reason == "tool_calls"
    assert (r.store.calls / rec.id / "response.sse").exists()
    assert r.record(chat_exchange(url="https://gw/v1/embeddings"), source="sdk") is None
    assert [row.id for row in r.store.index()] == ["c_000001", "c_000002"]


def test_session_sources_and_tags(tmp_path):
    r = Recorder(tmp_path / "run")
    rec = r.record(chat_exchange({"X-Stainless-Retry-Count": "1"}), source="sdk", session="exp-7")
    assert rec is not None and rec.tags == {"session": "exp-7", "retry_count": "1"}
    rec = r.record(chat_exchange({"Session-Id": "s-1"}), source="sdk", tags={"case": "a"})
    assert rec is not None and rec.tags == {"session": "s-1", "case": "a"}
    body = {**CHAT_BODY, "prompt_cache_key": "pck"}
    rec = r.record(chat_exchange(body=body), source="sdk")
    assert rec is not None and rec.tags == {"session": "pck"}


def test_parent_only_within_session(tmp_path):
    r = Recorder(tmp_path / "run")
    first = r.record(chat_exchange(), source="sdk", session="s")
    assert first is not None and first.message
    raw = first.message["choices"][0]["message"]
    tools = [
        {"role": "tool", "tool_call_id": tc["id"], "content": "晴"} for tc in raw["tool_calls"]
    ]
    nxt = {**CHAT_BODY, "messages": [*CHAT_BODY["messages"], raw, *tools]}
    second = r.record(chat_exchange(body=nxt), source="sdk", session="s")
    assert second is not None and second.parent == first.id
    lone = r.record(chat_exchange(), source="sdk")
    again = r.record(chat_exchange(body=nxt), source="sdk")
    assert lone is not None and again is not None
    assert again.parent is None and again.parent_candidates == [lone.id]


def test_shared_per_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = Recorder.shared("run")
    assert Recorder.shared(tmp_path / "run") is a
    assert Recorder.shared(tmp_path / "other") is not a


def test_failure_without_response_is_recorded(tmp_path):
    r = Recorder(tmp_path / "run")
    ex = Exchange("POST", "https://gw/v1/chat/completions", {}, b"{}")
    ex.end, ex.error = "connect_error", "ConnectError: refused"
    rec = r.record(ex, source="sdk")
    assert rec is not None and not rec.outcome.ok and rec.outcome.transport_end == "connect_error"
    assert rec.response.status is None and rec.message is None


def test_malformed_request_still_recorded(tmp_path):
    """Request body is malformed (messages is a string) and the server returns 400: still record, do not infer parent, and do not raise."""
    r = Recorder(tmp_path / "run")
    r.record(chat_exchange(), source="sdk", session="s")
    ex = chat_exchange(body={"model": "m", "messages": "hi"})
    ex.status, ex.chunks = 400, [(0.1, json.dumps({"error": {"code": "invalid_value"}}))]
    rec = r.record(ex, source="sdk", session="s")
    assert rec is not None and not rec.outcome.ok and rec.parent is None
    assert rec.outcome.error and "invalid_value" in rec.outcome.error


def test_record_safely_warns_instead_of_raising(tmp_path, monkeypatch):
    r = Recorder(tmp_path / "run")

    def boom(*a, **k):
        raise OSError("磁盘满了")

    monkeypatch.setattr(r.store, "write", boom)
    with pytest.warns(RuntimeWarning, match="磁盘满了"):
        assert r.record_safely(chat_exchange(), source="sdk") is None


def test_relative_run_dir_survives_chdir(tmp_path, monkeypatch):
    """A recorder created with a relative run dir writes back to the original directory even if the caller chdirs midway."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    r = Recorder("runs/E1")
    monkeypatch.chdir(tmp_path / "b")
    rec = r.record(chat_exchange(), source="sdk")
    assert (
        rec is not None
        and (tmp_path / "a" / "runs" / "E1" / "calls" / rec.id / "meta.json").exists()
    )
    assert not (tmp_path / "b" / "runs").exists()


def test_transport_error_kept_in_outcome_for_streams(tmp_path):
    """Streaming response stopped midway: keep the specific transport reason in the outcome."""
    r = Recorder(tmp_path / "run")
    ex = chat_exchange(stream=True)
    ex.chunks = [(0.1, ex.chunks[0][1][:300])]
    ex.end, ex.error = "eof", "RemoteProtocolError: peer closed connection"
    rec = r.record(ex, source="sdk")
    assert rec is not None and not rec.outcome.ok
    assert "RemoteProtocolError" in (rec.outcome.error or "")
