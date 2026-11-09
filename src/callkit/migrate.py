"""Move the older file layout into a single SQLite file.

Offline migration: while migrating, no other process may write to the old directory (at the start it tries to take the old directory's lock, errors out if it cannot, and holds it until the end). The new directory must not exist,
or must be the same migration left unfinished last time; while unfinished, the run table holds ``migrating_from``, and other processes opening the new directory get an error.

- Records are moved one by one in the row order of the old ``index.jsonl``, with seq increasing in turn and call numbers kept as they are: the ledger order and the result of "a later write overrides an earlier one" for resumption keys both stay the same.
- Each record's ``run`` is changed to the new directory; "directory:call number" references to the old directory in ``parent``, ``fork_of``, and the ``replay_of`` tag are changed to the new directory,
  while references to other runs are left alone.
- ``request.json.gz`` is decompressed and stored as ``request.json``, and ``body_file`` in meta is changed to match.
- After each batch is committed, the batch is read back from the new database and compared byte for byte with the old files (meta may only be rewritten in the ways above); it stops on any mismatch.
- After an interruption, running the same command again continues from the last committed position. Old records that cannot be read are skipped, counted, and reported with a warning.
"""

from __future__ import annotations

import fcntl
import json
import sqlite3
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from callkit.errors import CallkitError
from callkit.pricing import load_prices
from callkit.record import Record
from callkit.store.base import DB_FILE
from callkit.store.common import LOCK_FILE
from callkit.store.files import PRICES_FILE, FileStore
from callkit.store.sqlite_rows import (
    create_schema,
    insert_item,
    insert_prices,
    insert_record,
    prepare_new_db,
    unpack,
)

BATCH = 200  # records moved per transaction; also the granularity of the read-back comparison


@dataclass
class MigrateReport:
    migrated: int = 0  # number of records moved in this run
    resumed_from: int = 0  # number of records already moved when resuming
    # old records that could not be read and were not moved
    skipped: list[str] = field(default_factory=list)
    items: int = 0
    prices: int = 0


def migrate(
    old: str | Path, new: str | Path, *, progress: Callable[[int, int], None] | None = None
) -> MigrateReport:
    """Move the older-layout directory ``old`` into ``new`` (SQLite). ``progress(moved, total)`` is called once per batch."""
    src, dst = Path(old).resolve(), Path(new).resolve()
    if not (src / "run.json").exists() or (src / DB_FILE).exists():
        raise CallkitError(f"{src} is not a file-layout run dir (needs run.json and no {DB_FILE})")
    with _hold_lock(src):
        store = FileStore(src, create=False)
        db = _open_target(src, dst, store.run_meta())
        report = MigrateReport(resumed_from=_progress(db))
        rows = store.index()
        for start in range(report.resumed_from, len(rows), BATCH):
            batch = rows[start : start + BATCH]
            _move_batch(db, store, [r.id for r in batch], (src, dst), report, start + len(batch))
            if progress is not None:
                progress(start + len(batch), len(rows))
        _finish(db, store, report)
        db.close()
    return report


def rewrite_ref(value: str | None, src: Path, dst: Path) -> str | None:
    """Change "old directory:call number" to "new directory:call number"; anything else is returned unchanged."""
    prefix = f"{src}:"
    if value and value.startswith(prefix):
        return f"{dst}:{value[len(prefix) :]}"
    return value


@contextmanager
def _hold_lock(src: Path) -> Iterator[None]:
    with (src / LOCK_FILE).open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise CallkitError(
                f"{src} is locked by another process; stop everything writing to it before migrating"
            ) from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _open_target(src: Path, dst: Path, run_meta: dict[str, Any]) -> sqlite3.Connection:
    """The new directory: if it does not exist, create the database (recording migrating_from in the run table); if it exists, it must be the same unfinished migration."""
    fresh = not (dst / DB_FILE).exists()
    if fresh and dst.exists() and any(dst.iterdir()):
        raise CallkitError(f"{dst} already exists and is not empty; migrate into a new directory")
    dst.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(dst / DB_FILE, autocommit=True)
    prepare_new_db(db)
    db.execute("PRAGMA synchronous=NORMAL")
    if fresh:
        meta = {k: v for k, v in run_meta.items() if k != "format"}
        db.execute("BEGIN IMMEDIATE")
        create_schema(db, {**meta, "migrating_from": str(src), "migrate_progress": 0})
        db.execute("COMMIT")
        return db
    got = db.execute("SELECT value FROM run WHERE key = 'migrating_from'").fetchone()
    if got is None or json.loads(got[0]) != str(src):
        db.close()
        raise CallkitError(f"{dst} is not an unfinished migration from {src}")
    return db


def _progress(db: sqlite3.Connection) -> int:
    got = db.execute("SELECT value FROM run WHERE key = 'migrate_progress'").fetchone()
    return int(json.loads(got[0])) if got else 0


def _set_progress(db: sqlite3.Connection, n: int) -> None:
    db.execute("UPDATE run SET value = ? WHERE key = 'migrate_progress'", (json.dumps(n),))


def _move_batch(
    db: sqlite3.Connection,
    store: FileStore,
    ids: list[str],
    dirs: tuple[Path, Path],
    report: MigrateReport,
    done_after: int,
) -> None:
    """A batch of records and the progress after moving them are committed in one transaction (a crash in the middle cannot cause records to be moved twice), then read back and compared after the commit."""
    src, dst = dirs
    moved: list[tuple[str, dict[str, bytes]]] = []
    db.execute("BEGIN IMMEDIATE")
    try:
        for cid in ids:
            got = _read_old(store, cid)
            if got is None:
                report.skipped.append(cid)
                continue
            rec, files, expected = got
            _rewrite(rec, src, dst)
            insert_record(db, rec, files)
            moved.append((cid, expected))
        _set_progress(db, done_after)
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
    for cid, expected in moved:
        _verify(db, cid, expected, src, dst)
    report.migrated += len(moved)


def _read_old(
    store: FileStore, cid: str
) -> tuple[Record, dict[str, bytes], dict[str, bytes]] | None:
    """Read one old record: the Record, the fields to store (without meta.json, with .gz already decompressed), and the original old files for comparison."""
    try:
        rec = store.load(cid)
        expected = {name: store.read_file(cid, name) for name in store.file_names(cid)}
    except (OSError, ValueError, KeyError, TypeError) as e:
        warnings.warn(
            f"skipped unreadable record {cid}: {type(e).__name__}: {e}",
            RuntimeWarning,
            stacklevel=4,
        )
        return None
    files = {n: b for n, b in expected.items() if n != "meta.json"}
    if rec.request.body_file.endswith(".gz"):
        files.pop(rec.request.body_file)
        files["request.json"] = rec.body
        expected["request.json"] = rec.body
        expected.pop(rec.request.body_file)
        rec.request.body_file = "request.json"
    return rec, files, expected


def _rewrite(rec: Record, src: Path, dst: Path) -> None:
    rec.run = str(dst)
    rec.parent = rewrite_ref(rec.parent, src, dst)
    rec.fork_of = rewrite_ref(rec.fork_of, src, dst)
    if "replay_of" in rec.tags:
        rec.tags["replay_of"] = rewrite_ref(rec.tags["replay_of"], src, dst) or ""


def _verify(
    db: sqlite3.Connection, cid: str, expected: dict[str, bytes], src: Path, dst: Path
) -> None:
    (seq,) = db.execute("SELECT seq FROM calls WHERE id = ?", (cid,)).fetchone()
    got = {
        n: unpack(c, d)
        for n, c, d in db.execute("SELECT name, codec, data FROM blobs WHERE call_seq = ?", (seq,))
    }
    if set(got) != set(expected):
        raise CallkitError(
            f"migration check failed for {cid}: fields {sorted(got)} != {sorted(expected)}"
        )
    for name, data in expected.items():
        same = _same_meta(data, got[name], src, dst) if name == "meta.json" else data == got[name]
        if not same:
            raise CallkitError(f"migration check failed for {cid}: {name} differs")


def _same_meta(old: bytes, new: bytes, src: Path, dst: Path) -> bool:
    """After the old meta is rewritten by the migration rules, it should be exactly the same as the meta in the new database."""
    rec = Record.from_dict(json.loads(old))
    _rewrite(rec, src, dst)
    if rec.request.body_file.endswith(".gz"):
        rec.request.body_file = "request.json"
    return rec.to_dict() == json.loads(new)


def _finish(db: sqlite3.Connection, store: FileStore, report: MigrateReport) -> None:
    """Move the item ledger and the price table in one go, remove migrating_from, and record migrated_from."""
    items = store.items()
    path = (
        store.dir / PRICES_FILE
        # bypass merge_prices: it would take the old directory's lock, which this process already holds on another handle
    )
    prices = load_prices(path) if path.exists() else {}
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("DELETE FROM items")
        db.execute("DELETE FROM prices")
        for line in items:
            insert_item(db, line)
        insert_prices(db, prices)
        db.execute("UPDATE run SET key = 'migrated_from' WHERE key = 'migrating_from'")
        db.execute("DELETE FROM run WHERE key = 'migrate_progress'")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
    report.items, report.prices = len(items), len(prices)
