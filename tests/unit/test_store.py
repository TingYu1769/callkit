"""RunStore: persistence, read back, resume, and concurrency."""

import json
import threading

from callkit.record import Attempt, Outcome, Record, Request, Response, Timing
from callkit.store import GZIP_ABOVE, RunStore, is_secret_header, redact
from callkit.transport import RawAttempt


def make(store, *, ok=True, cost=0.1, item=None, repeat=None, body=b'{"a": 1}', message=None):
    cid = store.new_call_id()
    return Record(
        id=cid,
        api="anthropic.messages",
        source="self",
        run=str(store.dir),
        request=Request(
            "http://gw/v1/messages",
            {"x-api-key": "sk-secret", "anthropic-version": "2023-06-01"},
            "",
            f"h-{cid}",
        ),
        response=Response(200, {"set-cookie": "s", "x-oneapi-request-id": "r1"}, None, None),
        outcome=Outcome(ok, "done", "completed" if ok else "failed"),
        timing=Timing("2026-10-23T00:00:00Z", 10, 20),
        attempts=[
            Attempt(1, 529, "done", "failed", "retry", "overloaded"),
            Attempt(2, 200, "done", "completed", "stop", "completed"),
        ],
        cost_usd=cost,
        item_id=item,
        repeat=repeat,
        body=body,
        message=message,
    )


def raws():
    failed = RawAttempt(
        1, 529, {}, [], '{"error": {"type": "overloaded_error"}}', "done", "t0", 0.1, 0.1
    )
    final = RawAttempt(
        2,
        200,
        {},
        [(0.0, "event: ping"), (0.0, 'data: {"type": "ping"}'), (0.0, "")],
        None,
        "done",
        "t1",
        0.1,
        0.2,
    )
    return [failed, final]


def test_write_and_load_round_trip(tmp_path):
    store = RunStore(tmp_path / "run")
    rec = make(store, message={"id": "msg_1", "content": []})
    store.write(rec, raws())
    d = store.calls / rec.id
    assert (d / "response.sse").read_text() == 'event: ping\ndata: {"type": "ping"}\n\n'
    assert "overloaded_error" in (d / "attempts/1.body").read_text()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["request"]["headers"]["x-api-key"] == "<redacted>"
    assert meta["response"]["headers"]["set-cookie"] == "<redacted>"
    assert meta["attempts"][0]["events_file"] == "attempts/1.body"
    back = store.load(rec.id)
    assert back.body == b'{"a": 1}' and back.message == {"id": "msg_1", "content": []}
    assert back.to_dict() == meta


def test_large_body_is_gzipped(tmp_path):
    store = RunStore(tmp_path / "run")
    body = b"x" * (GZIP_ABOVE + 1)
    rec = make(store, body=body)
    store.write(rec, [])
    assert rec.request.body_file == "request.json.gz"
    assert store.load(rec.id).body == body


def test_reopen_restores_resume_state(tmp_path):
    store = RunStore(tmp_path / "run", run_meta={"endpoint": "gw"})
    a = make(store, ok=True, cost=0.5, item="q1", repeat=0)
    store.write(a, [])
    store.write(make(store, ok=False, cost=0.2, item="q1", repeat=1), [])
    with (store.dir / "index.jsonl").open("a") as f:
        f.write('{"id": "c_0000')  # Simulate a crash halfway through writing
    again = RunStore(tmp_path / "run", run_meta={"endpoint": "other"})
    assert again.done() == {("q1", 0): a.request.sha256}
    assert abs(again.spent_usd() - 0.7) < 1e-9
    assert again.new_call_id() == "c_000003"
    assert json.loads((store.dir / "run.json").read_text())["endpoint"] == "gw"


def test_concurrent_writes(tmp_path):
    store = RunStore(tmp_path / "run")

    def work():
        for _ in range(25):
            store.write(make(store, cost=0.01), [])

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [r.id for r in RunStore(tmp_path / "run").index()]
    assert len(ids) == 200 and len(set(ids)) == 200
    assert abs(store.spent_usd() - 2.0) < 1e-9


def test_two_stores_on_same_dir_skip_taken_ids(tmp_path):
    """When two RunStore instances open the same directory (such as another process), a colliding call id is skipped instead of raising."""
    a, b = RunStore(tmp_path / "run"), RunStore(tmp_path / "run")
    got = [a.new_call_id(), b.new_call_id(), a.new_call_id(), b.new_call_id()]
    assert got == ["c_000001", "c_000002", "c_000003", "c_000004"]


def test_concurrent_init_does_not_share_temp_file(tmp_path):
    """When multiple processes initialize the same directory at the same time, each uses its own temporary file."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json.tmp").mkdir()  # The old fixed temporary file name is occupied
    RunStore(run)
    assert json.loads((run / "run.json").read_text())["format"] == "callkit.run/1"


# ---- Multi-process shared ledger and price table ----


def test_spent_and_done_shared_through_index(tmp_path):
    """Two RunStore instances (equivalent to two processes) write the same directory: each sees the other's spend and resume keys."""
    a, b = RunStore(tmp_path / "run"), RunStore(tmp_path / "run")
    a.write(make(a, cost=0.25, item="q1", repeat=0), raws())
    b.write(make(b, cost=0.5), raws())
    assert a.spent_usd() == b.spent_usd() == 0.75
    assert ("q1", 0) in b.done() and len(a.index()) == len(b.index()) == 2


def test_half_written_line_is_cut_off_and_skipped(tmp_path):
    import pytest

    run = tmp_path / "run"
    s = RunStore(run)
    s.write(make(s, cost=0.1), raws())
    with (run / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "c_9999')  # Process crashed halfway through writing
    with pytest.warns(RuntimeWarning, match="unreadable line"):
        s.write(make(s, cost=0.2), raws())
    lines = (run / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines[1] == '{"id": "c_9999' and json.loads(lines[2])["cost_usd"] == 0.2
    with pytest.warns(RuntimeWarning, match="unreadable line"):
        fresh = RunStore(run)
    assert [r.cost_usd for r in fresh.index()] == [0.1, 0.2]


def test_merge_prices(tmp_path):
    import pytest

    from callkit import Price
    from callkit.errors import CallkitError

    s = RunStore(tmp_path / "run")
    p1, p2 = Price(1, 2, 0.1, 1.25), Price(3, 15, 0.3, 3.75, {"1h": 6})
    assert s.merge_prices({"a": p1}) == {"a": p1}
    assert s.merge_prices({"a": p1, "b": p2}) == {"a": p1, "b": p2}
    on_disk = json.loads((tmp_path / "run" / "prices.json").read_text())
    assert on_disk["b"]["cache_write_by_ttl"] == {"1h": 6}
    with pytest.raises(CallkitError, match="already has different prices"):
        RunStore(tmp_path / "run").merge_prices({"a": p2})
    assert s.merge_prices({}) == {"a": p1, "b": p2}


def test_merge_and_append_wait_for_dir_lock(tmp_path):
    """Price merging and ledger appends both happen under the run directory cross-process lock."""
    from callkit import Price
    from callkit.store import dir_lock

    run = tmp_path / "run"
    # Each uses its own instance, equivalent to two processes
    s, other = RunStore(run), RunStore(run)
    done = []

    def merge():
        done.append(s.merge_prices({"m": Price(1, 2, 0.1, 1.25)}))

    def append():
        done.append(other.write(make(other), raws()))

    t1, t2 = threading.Thread(target=merge), threading.Thread(target=append)
    with dir_lock(run):
        t1.start()
        t2.start()
        t1.join(0.2)
        t2.join(0.2)
        assert done == []  # The lock is held, so merge and append both wait
    t1.join(5)
    t2.join(5)
    assert len(done) == 2 and len(s.index()) == 1


def test_secret_headers_by_name():
    for name in [
        "x-api-key",
        "Authorization",
        "api-key",
        "x-goog-api-key",
        "X-Auth-Token",
        "cookie",
    ]:
        assert is_secret_header(name), name
    for name in ["anthropic-ratelimit-input-tokens-remaining", "x-request-id", "openai-project"]:
        assert not is_secret_header(name), name
    assert redact({"X-Gw": "k", "x-request-id": "r"}, "x-gw") == {
        "X-Gw": "<redacted>",
        "x-request-id": "r",
    }
