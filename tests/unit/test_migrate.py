"""Migration from the old file layout to SQLite, plus the cat, unpack and backup commands."""

import gzip
import json
import sqlite3

import pytest

from callkit import Price
from callkit import migrate as migrate_mod
from callkit.cli import main
from callkit.cli_store import unpack
from callkit.errors import CallkitError
from callkit.migrate import migrate
from callkit.record import Record
from callkit.store import GZIP_ABOVE, FileStore, RunStore, SqliteStore, dir_lock
from tests.unit.test_store import make, raws

PRICE = Price(1, 2, 0.1, 1.25)
WHEN = {"started_at": "2026-11-09T00:00:00Z", "finished_at": "2026-11-09T00:00:01Z"}


@pytest.fixture
def old(tmp_path):
    """Old-layout directory: a plain record, one with retries, a large request body (.gz), references pointing into this directory and elsewhere, the item ledger, and the price table."""
    run = (tmp_path / "old").resolve()
    s = FileStore(run, run_meta={"endpoint": {"name": "gw"}})
    s.merge_prices({"m": PRICE})
    first = make(
        s, cost=0.5, item="q", repeat=0, message={"content": [{"type": "text", "text": "a"}]}
    )
    s.write(first, raws())
    big = make(s, cost=0.25, body=b'{"x": "' + b"a" * GZIP_ABOVE + b'"}')
    s.write(big, [])
    linked = make(s, cost=0.25, item="q", repeat=0)
    linked.parent = f"{run}:{first.id}"
    linked.fork_of = "/elsewhere/run:c_000009"
    linked.tags = {"replay_of": f"{run}:{big.id}", "session": "s1"}
    s.write(linked, raws())
    s.append_item(
        {"item_id": "q", "repeat": 0, "ok": True, "error": None, **WHEN, "call_ids": [first.id]}
    )
    s.append_item({"item_id": "r", "repeat": 0, "ok": False, "error": "x", **WHEN, "call_ids": []})
    return run


def _files(store, cid):
    return {n: store.read_file(cid, n) for n in store.file_names(cid)}


def test_round_trip(old, tmp_path):
    new = (tmp_path / "new").resolve()
    report = migrate(old, new)
    assert (report.migrated, report.resumed_from, report.skipped) == (3, 0, [])
    assert (report.items, report.prices) == (2, 1)
    a, b = FileStore(old, create=False), RunStore(new)
    assert type(b) is SqliteStore
    assert [r.id for r in a.index()] == [r.id for r in b.index()]
    assert a.done() == b.done() and a.spent_usd() == b.spent_usd() == 1.0
    assert a.items() == b.items() and b.merge_prices({}) == {"m": PRICE}
    meta = b.run_meta()
    assert meta["endpoint"] == {"name": "gw"} and meta["migrated_from"] == str(old)
    assert meta["format"] == "callkit.run/2" and "migrate_progress" not in meta
    page = sqlite3.connect(new / "run.db").execute("PRAGMA page_size").fetchone()
    assert page == (65536,)
    for row in a.index():
        x, y = a.load(row.id), b.load(row.id)
        assert x.body == y.body and x.message == y.message and y.run == str(new)
    linked = b.load("c_000003")
    assert linked.parent == f"{new}:c_000001" and linked.fork_of == "/elsewhere/run:c_000009"
    assert linked.tags["replay_of"] == f"{new}:c_000002"
    assert b.load("c_000002").request.body_file == "request.json"
    assert b.summaries()[0].item == "q#0"


def test_unpack_restores_old_files(old, tmp_path):
    new, out = tmp_path / "new", tmp_path / "out"
    migrate(old, new)
    unpack(RunStore(new), None, out)
    a, c = FileStore(old, create=False), RunStore(out)
    assert type(c) is FileStore and [r.id for r in c.index()] == [r.id for r in a.index()]
    assert c.items() == a.items() and c.merge_prices({}) == {"m": PRICE}
    for row in a.index():
        before, after = _files(a, row.id), _files(c, row.id)
        if "request.json.gz" in before:  # the large request body was decompressed during migration
            before["request.json"] = gzip.decompress(before.pop("request.json.gz"))
        assert set(before) == set(after)
        for name in before:
            if name != "meta.json":
                assert before[name] == after[name], (row.id, name)
        assert c.load(row.id).body == a.load(row.id).body


def test_resume_after_crash(old, tmp_path, monkeypatch):
    new = tmp_path / "new"
    monkeypatch.setattr(migrate_mod, "BATCH", 1)
    real = migrate_mod.insert_record
    seen = []

    def crash_on_third(db, rec, files):
        seen.append(rec.id)
        if len(seen) == 3:
            raise KeyboardInterrupt  # the process is interrupted while moving the third record
        real(db, rec, files)

    monkeypatch.setattr(migrate_mod, "insert_record", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        migrate(old, new)
    with pytest.raises(CallkitError, match="still being migrated"):
        RunStore(new)
    monkeypatch.setattr(migrate_mod, "insert_record", real)
    report = migrate(old, new)
    assert (report.resumed_from, report.migrated) == (2, 1)
    assert [r.id for r in RunStore(new).index()] == ["c_000001", "c_000002", "c_000003"]


def test_refuses_locked_or_occupied(old, tmp_path):
    with dir_lock(old), pytest.raises(CallkitError, match="locked by another process"):
        migrate(old, tmp_path / "new")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    with pytest.raises(CallkitError, match="not empty"):
        migrate(old, tmp_path / "busy")
    SqliteStore(tmp_path / "other")
    with pytest.raises(CallkitError, match="not an unfinished migration"):
        migrate(old, tmp_path / "other")
    with pytest.raises(CallkitError, match="not a file-layout run dir"):
        migrate(tmp_path / "other", tmp_path / "new2")


def test_unreadable_record_skipped(old, tmp_path):
    (old / "calls" / "c_000002" / "meta.json").unlink()
    with pytest.warns(RuntimeWarning, match="unreadable record c_000002"):
        report = migrate(old, tmp_path / "new")
    assert report.skipped == ["c_000002"] and report.migrated == 2


def test_check_catches_corruption(old, tmp_path, monkeypatch):
    real = migrate_mod.insert_record

    def corrupt(db, rec, files):
        real(db, rec, {**files, "response.sse": b"tampered"} if "response.sse" in files else files)

    monkeypatch.setattr(migrate_mod, "insert_record", corrupt)
    with pytest.raises(CallkitError, match="response.sse differs"):
        migrate(old, tmp_path / "new")


def test_rewrite_ref(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    assert migrate_mod.rewrite_ref(f"{src}:c_000001", src, dst) == f"{dst}:c_000001"
    assert migrate_mod.rewrite_ref(f"{src}x:c_000001", src, dst) == f"{src}x:c_000001"
    assert migrate_mod.rewrite_ref("c_000001", src, dst) == "c_000001"
    assert migrate_mod.rewrite_ref(None, src, dst) is None


# ---- command line ----


def test_cli_migrate_cat_unpack_backup(old, tmp_path, capfdbinary):
    new = tmp_path / "new"
    assert main(["migrate", str(old), str(new)]) == 0
    assert b"migrated 3 calls" in capfdbinary.readouterr().err
    assert main(["cat", str(new), "c_000001"]) == 0
    assert capfdbinary.readouterr().out == b'{"a": 1}'
    assert main(["cat", str(new), "c_000001", "response"]) == 0
    assert capfdbinary.readouterr().out == b'event: ping\ndata: {"type": "ping"}\n\n'
    assert main(["cat", str(new), "c_000001", "meta"]) == 0
    assert Record.from_dict(json.loads(capfdbinary.readouterr().out)).id == "c_000001"
    assert main(["cat", str(new), "c_000001", "attempts/1.body"]) == 0
    assert b"overloaded_error" in capfdbinary.readouterr().out
    with pytest.raises(SystemExit, match="has no message"):
        main(["cat", str(new), "c_000002", "message"])
    with pytest.raises(SystemExit, match="cannot read"):
        main(["cat", str(new), "c_000099"])
    assert main(["unpack", str(new), "c_000001", "-o", str(tmp_path / "one")]) == 0
    assert [r.id for r in RunStore(tmp_path / "one").index()] == ["c_000001"]
    assert main(["backup", str(new), str(tmp_path / "copy")]) == 0
    copy = RunStore(tmp_path / "copy")
    assert [r.id for r in copy.index()] == ["c_000001", "c_000002", "c_000003"]
    with pytest.raises(SystemExit, match="already exists"):
        main(["backup", str(new), str(tmp_path / "copy")])
    with pytest.raises(SystemExit, match="only handles SQLite"):
        main(["backup", str(old), str(tmp_path / "copy2")])


def test_backup_sees_committed_wal_data(tmp_path):
    """Committed data that is only in the WAL and not yet checkpointed must also end up in the backup."""
    s = SqliteStore(tmp_path / "run")
    s.write(make(s), [])
    db = sqlite3.connect(tmp_path / "run" / "run.db")
    db.execute("PRAGMA wal_autocheckpoint=0")  # this connection does not trigger a checkpoint
    s.write(make(s), [])
    assert main(["backup", str(tmp_path / "run"), str(tmp_path / "copy")]) == 0
    assert len(RunStore(tmp_path / "copy").index()) == 2


def test_check_catches_missing_field(old, tmp_path, monkeypatch):
    real = migrate_mod.insert_record

    def drop(db, rec, files):
        real(db, rec, {n: b for n, b in files.items() if not n.startswith("attempts/")})

    monkeypatch.setattr(migrate_mod, "insert_record", drop)
    with pytest.raises(CallkitError, match="fields"):
        migrate(old, tmp_path / "new")


def test_resume_refuses_other_source(old, tmp_path, monkeypatch):
    new = tmp_path / "new"
    monkeypatch.setattr(
        migrate_mod,
        "insert_record",
        lambda db, rec, files: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(KeyboardInterrupt):
        migrate(old, new)
    other = tmp_path / "other"
    FileStore(other)
    monkeypatch.undo()
    with pytest.raises(CallkitError, match="not an unfinished migration"):
        migrate(other, new)


def test_cli_unpack_all_from_read_only_store(old, tmp_path):
    new, out = tmp_path / "new", tmp_path / "out"
    migrate(old, new)
    assert main(["unpack", str(new), "-o", str(out)]) == 0
    c = RunStore(out, create=False)
    assert len(c.index()) == 3 and c.prices() == {"m": PRICE} and len(c.items()) == 2
