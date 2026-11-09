"""RunStore: behavior shared by both backends (write, read, ledger, resume, prices, read-only, concurrency), plus the layout details of each."""

import json
import multiprocessing
import sqlite3
import threading
import uuid

import pytest

from callkit import Price
from callkit.errors import CallkitError
from callkit.record import Attempt, Outcome, Record, Request, Response, Timing
from callkit.store import (
    GZIP_ABOVE,
    FileStore,
    RunStore,
    SqliteStore,
    is_run_dir,
    is_secret_header,
    redact,
)
from callkit.store import sqlite as sqlite_store
from callkit.transport import RawAttempt


def make(store, *, ok=True, cost=0.1, item=None, repeat=None, body=b'{"a": 1}', message=None):
    """A record ready to be written; the call id is left empty and assigned by store.write."""
    return Record(
        id="",
        api="anthropic.messages",
        source="self",
        run=str(store.dir),
        request=Request(
            "http://gw/v1/messages",
            {"x-api-key": "sk-secret", "anthropic-version": "2023-06-01"},
            "",
            f"h-{uuid.uuid4().hex}",
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


@pytest.fixture(params=["files", "sqlite"])
def backend(request):
    """Return the backend class for the parameter; both backends run the same behavior tests."""
    return FileStore if request.param == "files" else SqliteStore


# ---- Behavior contract: both backends must satisfy it ----


def test_write_and_load_round_trip(backend, tmp_path):
    store = backend(tmp_path / "run")
    rec = make(store, message={"id": "msg_1", "content": [{"type": "text", "text": "hi"}]})
    store.write(rec, raws())
    assert rec.id == "c_000001"
    back = store.load(rec.id)
    assert back.body == b'{"a": 1}' and back.message == rec.message
    assert back.to_dict() == rec.to_dict() == store.load_meta(rec.id).to_dict()
    assert back.request.headers["x-api-key"] == "<redacted>"
    assert back.response.headers["set-cookie"] == "<redacted>"
    assert back.attempts[0].events_file == "attempts/1.body"
    assert store.raw_response(rec.id) == ('event: ping\ndata: {"type": "ping"}\n\n', True)
    assert store.load_message(rec.id) == rec.message


def test_ids_allocated_on_write_and_preset_ids_kept(backend, tmp_path):
    store = backend(tmp_path / "run")
    first, second = make(store), make(store)
    store.write(first, [])
    store.write(second, [])
    preset = make(store)
    preset.id = "c_000010"
    store.write(preset, [])
    assert [first.id, second.id, preset.id] == ["c_000001", "c_000002", "c_000010"]
    assert [r.id for r in store.index()] == ["c_000001", "c_000002", "c_000010"]
    assert store.raw_response("c_000001") == ("", False) and store.load_message("c_000001") is None


def test_reopen_restores_resume_state(backend, tmp_path):
    store = backend(tmp_path / "run", run_meta={"endpoint": "gw"})
    a = make(store, ok=True, cost=0.5, item="q1", repeat=0)
    store.write(a, [])
    store.write(make(store, ok=False, cost=0.2, item="q1", repeat=1), [])
    again = RunStore(tmp_path / "run", run_meta={"endpoint": "other"})
    assert type(again) is backend
    assert again.done() == {("q1", 0): a.request.sha256}
    assert abs(again.spent_usd() - 0.7) < 1e-9
    nxt = make(again)
    again.write(nxt, [])
    assert nxt.id == "c_000003"
    assert again.run_meta()["endpoint"] == "gw" and "created_at" in again.run_meta()


def test_later_success_overwrites_resume_key(backend, tmp_path):
    store = backend(tmp_path / "run")
    store.write(make(store, item="q", repeat=0), [])
    later = make(store, item="q", repeat=0)
    store.write(later, [])
    assert backend(tmp_path / "run").done() == {("q", 0): later.request.sha256}


def test_concurrent_writes(backend, tmp_path):
    store = backend(tmp_path / "run")
    errors = []

    def work():
        try:
            for _ in range(25):
                store.write(make(store, cost=0.01), [])
        # collect exceptions from the thread and assert on them in the main thread
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    ids = [r.id for r in RunStore(tmp_path / "run").index()]
    assert len(ids) == 200 and len(set(ids)) == 200
    assert abs(store.spent_usd() - 2.0) < 1e-9


def test_two_stores_share_ids_spend_and_resume_keys(backend, tmp_path):
    """Two instances on the same directory (equivalent to two processes): call ids do not collide, and each sees the other's spend and resume keys."""
    a, b = backend(tmp_path / "run"), backend(tmp_path / "run")
    recs = [make(a, cost=0.25, item="q1", repeat=0), make(b, cost=0.5), make(a, cost=0.25)]
    a.write(recs[0], raws())
    b.write(recs[1], raws())
    a.write(recs[2], raws())
    assert [r.id for r in recs] == ["c_000001", "c_000002", "c_000003"]
    assert a.spent_usd() == b.spent_usd() == 1.0
    assert ("q1", 0) in b.done() and len(a.index()) == len(b.index()) == 3


def _write_many(run_dir, n):
    store = RunStore(run_dir)
    for _ in range(n):
        store.write(make(store, cost=0.01), [])


def test_processes_write_one_ledger(backend, tmp_path):
    run = tmp_path / "run"
    backend(run)  # Create the directory in the layout under test first
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_write_many, args=(run, 20)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]
    ids = [r.id for r in RunStore(run).index()]
    assert len(ids) == 80 and len(set(ids)) == 80
    assert abs(RunStore(run).spent_usd() - 0.8) < 1e-9


def test_merge_prices(backend, tmp_path):
    s = backend(tmp_path / "run")
    p1, p2 = Price(1, 2, 0.1, 1.25), Price(3, 15, 0.3, 3.75, {"1h": 6})
    assert s.merge_prices({"a": p1}) == {"a": p1}
    assert s.merge_prices({"a": p1, "b": p2}) == {"a": p1, "b": p2}
    with pytest.raises(CallkitError, match="already has different prices"):
        RunStore(tmp_path / "run").merge_prices({"a": p2})
    assert RunStore(tmp_path / "run").merge_prices({}) == {"a": p1, "b": p2}
    assert RunStore(tmp_path / "run", create=False).prices() == {"a": p1, "b": p2}


def test_read_only_open(backend, tmp_path):
    run = tmp_path / "run"
    s = backend(run)
    rec = make(s, message={"content": []})
    s.write(rec, raws())
    ro = RunStore(run, create=False)
    assert type(ro) is backend and [r.id for r in ro.index()] == [rec.id]
    assert ro.load(rec.id).message == {"content": []} and ro.summaries()[0].id == rec.id
    with pytest.raises(CallkitError, match="not a callkit run dir"):
        RunStore(tmp_path / "nothing", create=False)
    assert not (tmp_path / "nothing").exists()


def test_summaries_match_records(backend, tmp_path):
    from callkit.summary import row_of

    s = backend(tmp_path / "run")
    body = json.dumps({"model": "m-1", "messages": []}).encode()
    msg = {"content": [{"type": "tool_use", "name": "get", "input": {"q": 1}}]}
    rec = make(s, item="q", repeat=2, body=body, message=msg)
    rec.tags = {"session": "s1"}
    s.write(rec, raws())
    (row,) = s.summaries()
    assert row == row_of(s.load(rec.id))
    assert (row.model, row.item, row.session, row.said) == ("m-1", "q#2", "s1", 'get({"q": 1})')


# ---- Backend detection ----


def test_backend_choice(tmp_path):
    assert type(RunStore(tmp_path / "new")) is SqliteStore
    assert is_run_dir(tmp_path / "new") and (tmp_path / "new" / "run.db").exists()
    FileStore(tmp_path / "old")
    assert type(RunStore(tmp_path / "old")) is FileStore and is_run_dir(tmp_path / "old")
    assert not is_run_dir(tmp_path)


def test_mixed_layout_refused(tmp_path):
    SqliteStore(tmp_path / "run")
    (tmp_path / "run" / "index.jsonl").write_text("")  # An older callkit version wrote into it
    with pytest.raises(CallkitError, match="mixes layouts"):
        RunStore(tmp_path / "run")


# ---- SQLite layout ----


def test_sqlite_tables_and_pragmas(tmp_path):
    SqliteStore(tmp_path / "run", run_meta={"endpoint": {"name": "gw"}})
    db = sqlite3.connect(tmp_path / "run" / "run.db")
    assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert db.execute("PRAGMA user_version").fetchone() == (1,)
    assert db.execute("PRAGMA page_size").fetchone() == (65536,)
    run = dict(db.execute("SELECT key, value FROM run"))
    assert json.loads(run["format"]) == "callkit.run/2"
    assert json.loads(run["endpoint"]) == {"name": "gw"}


def test_sqlite_compresses_large_fields(tmp_path):
    s = SqliteStore(tmp_path / "run")
    body = b'{"x": "' + b"a" * 10_000 + b'"}'
    rec = make(s, body=body)
    s.write(rec, raws())
    db = sqlite3.connect(tmp_path / "run" / "run.db")
    codecs = dict(db.execute("SELECT name, codec FROM blobs"))
    assert codecs["request.json"] == "zlib" and codecs["response.sse"] == "raw"
    (size,) = db.execute("SELECT size FROM blobs WHERE name = 'request.json'").fetchone()
    assert size == len(body) and s.load(rec.id).body == body
    # No longer generates a .gz file name based on size
    assert rec.request.body_file == "request.json"


def test_sqlite_read_only_store_refuses_writes(tmp_path):
    SqliteStore(tmp_path / "run")
    ro = RunStore(tmp_path / "run", create=False)
    with pytest.raises(CallkitError, match="read-only"):
        ro.write(make(ro), [])


def test_sqlite_refuses_use_after_fork(tmp_path, monkeypatch):
    s = SqliteStore(tmp_path / "run")
    monkeypatch.setattr(sqlite_store.os, "getpid", lambda: -1)
    with pytest.raises(CallkitError, match="fork"):
        s.index()


def test_sqlite_begin_retries_busy(monkeypatch):
    calls = []

    class Busy:
        def execute(self, sql):
            calls.append(sql)
            if len(calls) < 3:
                e = sqlite3.OperationalError("database is locked")
                e.sqlite_errorcode = sqlite3.SQLITE_BUSY  # type: ignore[attr-defined]
                raise e

    monkeypatch.setattr(sqlite_store.time, "sleep", lambda s: None)
    sqlite_store._begin(Busy())  # type: ignore[arg-type]
    assert calls == ["BEGIN IMMEDIATE"] * 3
    monkeypatch.setattr(sqlite_store, "WRITE_DEADLINE_S", -1.0)
    calls.clear()
    with pytest.raises(sqlite3.OperationalError):
        sqlite_store._begin(Busy())  # type: ignore[arg-type]
    assert len(calls) == 1


def test_sqlite_failed_write_rolls_back(tmp_path):
    s = SqliteStore(tmp_path / "run")
    s.write(make(s), [])
    dup = make(s)
    dup.id = "c_000001"  # The call id collides: the whole record is not persisted
    with pytest.raises(sqlite3.IntegrityError):
        s.write(dup, raws())
    db = sqlite3.connect(tmp_path / "run" / "run.db")
    assert db.execute("SELECT count(*) FROM blobs").fetchone() == (
        2,
    )  # Only the first record's request and meta
    s.write(make(s), [])  # The write connection is still usable after the rollback
    assert len(s.index()) == 2


# ---- Legacy file layout ----


def test_files_large_body_is_gzipped(tmp_path):
    store = FileStore(tmp_path / "run")
    body = b"x" * (GZIP_ABOVE + 1)
    rec = make(store, body=body)
    store.write(rec, [])
    assert rec.request.body_file == "request.json.gz"
    assert store.load(rec.id).body == body


def test_files_layout_on_disk(tmp_path):
    store = FileStore(tmp_path / "run")
    rec = make(store, message={"id": "msg_1", "content": []})
    store.write(rec, raws())
    d = store.calls / rec.id
    assert (d / "response.sse").read_text() == 'event: ping\ndata: {"type": "ping"}\n\n'
    assert "overloaded_error" in (d / "attempts/1.body").read_text()
    assert json.loads((d / "meta.json").read_text()) == rec.to_dict()
    assert json.loads((tmp_path / "run" / "run.json").read_text())["format"] == "callkit.run/1"


def test_files_two_stores_skip_taken_ids(tmp_path):
    """When two FileStore instances open the same directory, a colliding claimed id is skipped instead of raising."""
    a, b = FileStore(tmp_path / "run"), FileStore(tmp_path / "run")
    got = [a._claim_id(), b._claim_id(), a._claim_id(), b._claim_id()]
    assert got == ["c_000001", "c_000002", "c_000003", "c_000004"]


def test_files_concurrent_init_does_not_share_temp_file(tmp_path):
    """When multiple processes initialize the same directory at the same time, each uses its own temporary file."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json.tmp").mkdir()  # The old fixed temporary file name is occupied
    FileStore(run)
    assert json.loads((run / "run.json").read_text())["format"] == "callkit.run/1"


def test_files_half_written_line_is_cut_off_and_skipped(tmp_path):
    run = tmp_path / "run"
    s = FileStore(run)
    s.write(make(s, cost=0.1), raws())
    with (run / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "c_9999')  # Process crashed halfway through writing
    with pytest.warns(RuntimeWarning, match="unreadable line"):
        s.write(make(s, cost=0.2), raws())
    lines = (run / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines[1] == '{"id": "c_9999' and json.loads(lines[2])["cost_usd"] == 0.2
    with pytest.warns(RuntimeWarning, match="unreadable line"):
        fresh = FileStore(run)
    assert [r.cost_usd for r in fresh.index()] == [0.1, 0.2]


def test_files_merge_and_append_wait_for_dir_lock(tmp_path):
    """Price merging and ledger appends both happen under the run directory cross-process lock."""
    from callkit.store import dir_lock

    run = tmp_path / "run"
    # Each uses its own instance, equivalent to two processes
    s, other = FileStore(run), FileStore(run)
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
    assert json.loads((run / "prices.json").read_text())["m"]["input"] == 1


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


def test_item_ledger(backend, tmp_path):
    s = backend(tmp_path / "run")
    assert s.items() == [] and s.items_done() == set()
    s.append_item({"item_id": "a", "repeat": 0, "ok": True, "error": None, "call_ids": ["c_1"]})
    s.append_item({"item_id": "a", "repeat": 0, "ok": False, "error": "boom", "call_ids": []})
    s.append_item({"item_id": "中\u2028", "repeat": 1, "ok": False, "error": "x", "call_ids": []})
    again = RunStore(tmp_path / "run")
    assert again.items_done() == {("a", 0)}  # A later failure does not erase an existing success
    rows = again.items()
    assert [(r["item_id"], r["ok"], r["call_ids"]) for r in rows] == [
        ("a", True, ["c_1"]),
        ("a", False, []),
        ("中\u2028", False, []),
    ]
    assert rows[1]["error"] == "boom"


def test_sqlite_checkpoints_in_background(tmp_path, monkeypatch):
    """Writers do not checkpoint themselves; a background thread, after seeing new writes, writes the pages in the WAL back to run.db.
    Only one thread runs per database, and it exits once every store writing to it has been collected.
    If time.sleep is replaced with a no-op elsewhere (common in tests), the thread does not spin."""
    import gc
    import os as _os
    import time as _time

    def no_sleep(seconds):
        raise AssertionError("the checkpoint thread must not use time.sleep")

    monkeypatch.setattr(sqlite_store, "CHECKPOINT_EVERY_S", 0.05)
    monkeypatch.setattr(sqlite_store.time, "sleep", no_sleep)
    s, twin = SqliteStore(tmp_path / "run"), SqliteStore(tmp_path / "run")
    db_file = (tmp_path / "run" / "run.db").resolve()
    before = db_file.stat().st_size
    # 2 MB that does not compress; before the checkpoint it exists only in the WAL
    s.write(make(s, body=_os.urandom(2_000_000)), [])
    twin.write(make(twin), [])
    deadline = _time.monotonic() + 5
    while db_file.stat().st_size < before + 2_000_000 and _time.monotonic() < deadline:
        threading.Event().wait(0.05)
    assert db_file.stat().st_size >= before + 2_000_000
    assert len(sqlite_store._CHECKPOINTERS[db_file]) == 2  # Two stores share one thread
    del s, twin
    gc.collect()
    deadline = _time.monotonic() + 5
    while db_file in sqlite_store._CHECKPOINTERS and _time.monotonic() < deadline:
        threading.Event().wait(0.05)
    assert db_file not in sqlite_store._CHECKPOINTERS  # The thread has exited


def test_sqlite_write_waits_for_dir_lock(tmp_path):
    """Write transactions run under the run directory cross-process lock: while another process holds it, writes queue up."""
    from callkit.store import dir_lock

    run = tmp_path / "run"
    s = SqliteStore(run)
    done = []
    t = threading.Thread(target=lambda: done.append(s.write(make(s), [])))
    with dir_lock(run):
        t.start()
        t.join(0.3)
        assert done == []
    t.join(5)
    assert done == [None] and len(s.index()) == 1
