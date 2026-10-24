"""Replay cache: on Client and SDK reruns, identical requests get old responses, skip network access, and cost is recorded as 0."""

import json
import random
from collections import defaultdict, deque

import httpx
import openai
import pytest

from callkit import Client, Endpoint, fork, recording_http_client, transport
from callkit.errors import CallkitError
from callkit.replay import ReplayCache, replay_key
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
    """Old run: A was sent twice (two different samples), and B once."""
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
        # replayed records can still be forked
        alt = c.call(fork(back, model="claude-opus-5-5"), fork_of=back)
    old = RunStore(old_run)
    ids = [r.tags.get("replay_of", "").rsplit(":", 1)[-1] for r in recs]
    assert ids == ["c_000001", "c_000002", "c_000003", ""]
    # only the third A (the old run has only two A samples) and the fork hit the network
    assert len(server.requests) == 2
    olds = [old.load(i).message for i in ("c_000001", "c_000002", "c_000003")]
    assert [r.message for r in recs[:3]] == olds
    assert recs[0].message != recs[1].message  # repeated requests get different samples
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
    )  # cost is still recorded as 0 even with prices


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
    """Clients writing the same new run share counters; different new runs each start from the beginning."""
    new, other = tmp_path / "new", tmp_path / "other"
    with (
        Client(EP, new, replay_from=old_run) as c1,
        Client(EP, new, replay_from=old_run) as c2,
        Client(EP, other, replay_from=old_run) as c3,
    ):
        got = [c.call(A).tags["replay_of"].rsplit(":", 1)[-1] for c in (c1, c2, c3)]
    assert got == ["c_000001", "c_000002", "c_000001"]


def test_other_gateway_or_org_not_replayed(server, old_run, tmp_path):
    """Sending the same request body to a different gateway or organization is not considered the same request."""
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


def _fake_run(tmp_path, specs):
    """Write records directly from (success/failure, URL, org header, request hash) without network access."""
    from tests.unit.test_store import make

    store = RunStore(tmp_path / "src")
    for ok, url, org, sha in specs:
        rec = make(store, ok=ok)
        rec.request.url, rec.request.sha256 = url, sha
        rec.request.headers = {"openai-organization": org} if org else {}
        store.write(rec, [])
    return store.dir


def test_open_reads_only_the_index(tmp_path, monkeypatch):
    a, b = "http://a/v1/messages", "http://b/v1/messages"
    run = _fake_run(tmp_path, [(True, a, None, "h"), (True, b, None, "h"), (True, a, None, "h")])
    reads = []
    real = RunStore.load_meta
    monkeypatch.setattr(
        RunStore, "load_meta", lambda self, cid: reads.append(cid) or real(self, cid)
    )
    cache = ReplayCache(run)
    assert reads == []  # no meta.json is read on open
    got = cache.take(b, {}, "h")
    assert got is not None and got.id == "c_000002"
    assert reads == ["c_000001", "c_000002"]  # stops at the first match; c_000003 is not read
    got = cache.take(a, {}, "h")
    # records already read are not read again
    assert got is not None and got.id == "c_000001" and len(reads) == 2


def test_lazy_lookup_matches_eager_snapshot(tmp_path):
    """Compare against an eager table built by reading meta.json one by one; URL, org header, success/failure, and hash are randomly mixed."""
    rng = random.Random(7)
    urls, orgs, shas = (
        ["http://a/v1/messages", "http://b/v1/messages"],
        [None, "org-1"],
        ["x", "y", "z"],
    )
    specs = [
        (rng.random() > 0.2, rng.choice(urls), rng.choice(orgs), rng.choice(shas))
        for _ in range(60)
    ]
    run = _fake_run(tmp_path, specs)
    eager: dict[str, deque[str]] = defaultdict(deque)
    for n, (ok, url, org, sha) in enumerate(specs, 1):
        if ok:
            eager[replay_key(url, {"openai-organization": org} if org else {}, sha)].append(
                f"c_{n:06d}"
            )
    cache = ReplayCache(run)
    for _ in range(150):
        url, org, sha = rng.choice(urls), rng.choice(orgs), rng.choice(shas)
        headers = {"OpenAI-Organization": org} if org else {}
        want = eager[replay_key(url, headers, sha)]
        got = cache.take(url, headers, sha)
        assert (got.id if got else None) == (want.popleft() if want else None)


def test_unreadable_candidate_is_skipped_with_warning(tmp_path):
    run = _fake_run(tmp_path, [(True, "http://gw/v1/messages", None, "h")] * 2)
    (run / "calls" / "c_000001" / "meta.json").write_text("{broken")
    cache = ReplayCache(run)
    with pytest.warns(RuntimeWarning, match="c_000001/meta.json"):
        got = cache.take("http://gw/v1/messages", {}, "h")
    assert got is not None and got.id == "c_000002"
    assert cache.take("http://gw/v1/messages", {}, "h") is None


def test_record_taken_by_another_thread_meanwhile(tmp_path, monkeypatch):
    """Reading meta.json does not hold a lock: if another thread takes this record during a read, continue searching later; both sides cannot get the same record."""
    url = "http://gw/v1/messages"
    cache = ReplayCache(_fake_run(tmp_path, [(True, url, None, "h")] * 2))
    real = ReplayCache._key_of
    inner = []

    def key_of(self, call_id, sha256):
        if not inner:  # during the first read, simulate another thread taking a record first
            inner.append(None)
            inner[0] = self.take(url, {}, "h")
        return real(self, call_id, sha256)

    monkeypatch.setattr(ReplayCache, "_key_of", key_of)
    outer = cache.take(url, {}, "h")
    assert inner[0] is not None and outer is not None
    assert (inner[0].id, outer.id) == ("c_000001", "c_000002")
    assert cache.take(url, {}, "h") is None
