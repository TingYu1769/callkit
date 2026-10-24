"""Replay cache: when Client and SDK are mounted for replay reruns, identical requests get old responses, no network access, and cost is recorded as 0."""

import json

import httpx
import openai
import pytest

from callkit import Client, Endpoint, fork, recording_http_client, transport
from callkit.errors import CallkitError
from callkit.replay import ReplayCache
from callkit.store import RunStore
from tests.fakes import DATA, Server, good_stream, sse_response

EP = Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY")
A = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "A"}],
}
B = {**A, "messages": [{"role": "user", "content": "B"}]}
THINKING = (DATA / "stream_thinking_text.sse").read_bytes()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("CALLKIT_TEST_KEY", "sk-test")
    monkeypatch.setattr(transport.time, "sleep", lambda s: None)
    s = Server()
    monkeypatch.setattr(transport, "open_client", lambda endpoint: s.client())
    return s


@pytest.fixture
def old_run(server, tmp_path):
    """Old run: A was sent twice (two different samples), B once."""
    server.replies += [
        sse_response(good_stream()),
        sse_response(THINKING),
        sse_response(good_stream()),
    ]
    with Client(EP, tmp_path / "old") as c:
        for body in (A, A, B):
            c.call(body)
    server.requests.clear()
    return tmp_path / "old"


def test_client_replays_kth_sample_then_goes_online(server, old_run, tmp_path):
    server.replies += [sse_response(good_stream()), sse_response(good_stream())]
    with Client(EP, tmp_path / "new", replay_from=old_run) as c:
        recs = [c.call(body) for body in (A, A, B, A)]
        back = RunStore(tmp_path / "new").load(recs[1].id)
        # Records from replay can still be forked.
        alt = c.call(fork(back, model="claude-opus-5-5"), fork_of=back)
    old = RunStore(old_run)
    ids = [r.tags.get("replay_of", "").rsplit(":", 1)[-1] for r in recs]
    assert ids == ["c_000001", "c_000002", "c_000003", ""]
    # Only the third A (the old run has only two A records) and the fork hit the network.
    assert len(server.requests) == 2
    olds = [old.load(i).message for i in ("c_000001", "c_000002", "c_000003")]
    assert [r.message for r in recs[:3]] == olds
    assert recs[0].message != recs[1].message  # Repeated requests get different samples back.
    assert all(r.cost_usd == 0.0 and r.outcome.ok for r in recs[:3])
    assert all(r.attempts[0].reason == "replayed" for r in recs[:3])
    assert back.message == recs[1].message
    assert (tmp_path / "new" / "calls" / recs[1].id / "response.sse").exists()
    assert alt.fork_of == back.id and "replay_of" not in alt.tags


def test_same_dir_replay_uses_snapshot(server, old_run):
    server.replies.append(sse_response(good_stream()))
    with Client(EP, old_run, replay_from=old_run) as c:
        got = [c.call(A).tags.get("replay_of", "").rsplit(":", 1)[-1] for _ in range(3)]
    assert got == ["c_000001", "c_000002", ""] and len(server.requests) == 1


def sdk_client(run, transport_, **kw):
    http = recording_http_client(run, transport=transport_, **kw)
    return openai.OpenAI(
        base_url="http://gw.test/v1", api_key="sk-test", http_client=http, max_retries=0
    )


def test_sdk_replay_json_and_stream(tmp_path):
    chat_json = (DATA / "chat_tool_calls.json").read_bytes()
    chat_sse = (DATA / "chat_text.sse").read_bytes()
    replies = [
        httpx.Response(
            200,
            headers={"content-type": "application/json", "x-request-id": "r1"},
            content=chat_json,
        ),
        httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chat_sse),
    ]
    msgs = [{"role": "user", "content": "hi"}]
    old = sdk_client(tmp_path / "old", httpx.MockTransport(lambda r: replies.pop(0)))
    first = old.chat.completions.create(model="gpt-5.6-sol", messages=msgs)
    text = "".join(
        c.choices[0].delta.content or ""
        for c in old.chat.completions.create(model="gpt-5.6-sol", messages=msgs, stream=True)
        if c.choices and c.choices[0].delta
    )

    def offline(request):
        raise AssertionError("a replay hit must not hit the network")

    from callkit import Price

    price = {"gpt-5.6-sol": Price(input=1.25, output=10, cache_read=0.125, cache_write=1.5625)}
    new = sdk_client(
        tmp_path / "new", httpx.MockTransport(offline), replay_from=tmp_path / "old", prices=price
    )
    again = new.chat.completions.create(model="gpt-5.6-sol", messages=msgs)
    assert again.model_dump() == first.model_dump()
    streamed = new.chat.completions.create(model="gpt-5.6-sol", messages=msgs, stream=True)
    assert (
        "".join(
            c.choices[0].delta.content or "" for c in streamed if c.choices and c.choices[0].delta
        )
        == text
    )
    store = RunStore(tmp_path / "new")
    recs = [store.load(r.id) for r in store.index()]
    assert [r.tags["replay_of"].rsplit(":", 1)[-1] for r in recs] == ["c_000001", "c_000002"]
    assert all(
        r.cost_usd == 0.0 and r.outcome.ok and r.source == "sdk" for r in recs
    )  # Cost is recorded as 0 even when prices are provided.


def test_replay_needs_run_dir(tmp_path):
    with pytest.raises(CallkitError, match="is not a callkit run dir"):
        ReplayCache(tmp_path)
    with pytest.warns(RuntimeWarning, match="replay_from"):
        recording_http_client(None, replay_from=tmp_path)


def test_failed_records_not_replayed(server, tmp_path):
    from tests.fakes import error_response

    server.replies += [error_response(400, "invalid_request_error"), sse_response(good_stream())]
    with Client(EP, tmp_path / "old") as c:
        c.call(A)
    with Client(EP, tmp_path / "new", replay_from=tmp_path / "old") as c:
        rec = c.call(A)
    assert "replay_of" not in rec.tags and rec.outcome.ok
    assert json.loads(server.requests[-1].content)["messages"] == A["messages"]


def test_counters_shared_per_new_run(server, old_run, tmp_path):
    """Clients writing to the same new run share counters; different new runs each start from the beginning."""
    new, other = tmp_path / "new", tmp_path / "other"
    with (
        Client(EP, new, replay_from=old_run) as c1,
        Client(EP, new, replay_from=old_run) as c2,
        Client(EP, other, replay_from=old_run) as c3,
    ):
        got = [c.call(A).tags["replay_of"].rsplit(":", 1)[-1] for c in (c1, c2, c3)]
    assert got == ["c_000001", "c_000002", "c_000001"]


def test_other_gateway_or_org_not_replayed(server, old_run, tmp_path):
    """The same request body sent to a different gateway or organization is not considered identical."""
    server.replies += [sse_response(good_stream()), sse_response(good_stream())]
    other = Endpoint("gw2", "http://other-gw", "CALLKIT_TEST_KEY")
    with Client(other, tmp_path / "new", replay_from=old_run) as c:
        assert "replay_of" not in c.call(A).tags
    org = Endpoint(
        "gw", "http://gw", "CALLKIT_TEST_KEY", extra_headers={"OpenAI-Organization": "org-x"}
    )
    with Client(org, tmp_path / "new2", replay_from=old_run) as c:
        assert "replay_of" not in c.call(A).tags
    assert len(server.requests) == 2
