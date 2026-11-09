"""SQLite single-file layout: one ``run.db`` per run directory.

- ``calls`` is both the ledger and the summary, and holds no large fields; each row in ``blobs`` corresponds to one file of the older layout (meta.json, request.json,
  response.sse, ...), compressed with zlib when over 4 KB. ``seq`` is the write order, append-only (incremental ledger sync relies on it); ``num``/``id``
  is the call number, taken as ``max(num)+1`` inside the write transaction.
- Connection contract: ``autocommit=True``, and transactions are always written explicitly as ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK``; every connection sets
  ``synchronous=NORMAL`` and ``busy_timeout``, and WAL is set once when the database is created (that setting persists).
- One write connection per process, with a thread lock guarding the whole write transaction; each thread opens its own read connection. Continuing to use a store in a child process after fork is not supported.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from callkit.config import Price, PriceTable
from callkit.errors import CallkitError
from callkit.record import IndexRow, Record
from callkit.store.base import DB_FILE, RunStore
from callkit.store.common import dir_lock, redact
from callkit.store.sqlite_rows import (
    INDEX_COLUMNS,
    ROW_COLUMNS,
    create_schema,
    insert_item,
    insert_prepared,
    insert_prices,
    prepare_new_db,
    prepare_record,
    record_files,
    summary_row,
    unpack,
)
from callkit.summary import Row
from callkit.transport import RawAttempt

BUSY_TIMEOUT_MS = 60_000
# Checkpoints run in a background thread (PASSIVE, so writers are not blocked); writers no longer do one as part of committing: the commit that did it had to wait for an fsync, which on a spinning disk
# stalls all the writers queued behind it. The writer's own auto-checkpoint is only a fallback (when the WAL grows to 4096 pages, i.e. 256 MB at 64 KB per page)
CHECKPOINT_EVERY_S = 0.5
CHECKPOINT_PAGES = 4096
# total time limit for acquiring the write lock: besides busy_timeout, SQLite sometimes returns BUSY immediately, so retry outside it
WRITE_DEADLINE_S = 300.0
MIN_SQLITE = (3, 35, 0)  # INSERT … RETURNING
# traces of the older layout: appearing together with run.db means the layouts are mixed
LEGACY_MARKERS = ("calls", "index.jsonl")


class SqliteStore(RunStore):
    def __init__(
        self, run_dir: str | Path, *, run_meta: dict[str, Any] | None = None, create: bool = True
    ) -> None:
        """Open or create a run directory in the SQLite layout. When creating, write the run table (format name, creation time, and ``run_meta``); an existing one is not overwritten.
        ``create=False`` opens read-only: the database is not created or written, and connections use ``mode=ro``."""
        if sqlite3.sqlite_version_info < MIN_SQLITE:
            raise CallkitError(
                f"callkit needs SQLite >= {'.'.join(map(str, MIN_SQLITE))}, found {sqlite3.sqlite_version}"
            )
        super().__init__(run_dir)
        self.path = self.dir / DB_FILE
        mixed = [m for m in LEGACY_MARKERS if (self.dir / m).exists()]
        if mixed:
            raise CallkitError(
                f"run dir {self.dir} mixes layouts: it has {DB_FILE} and also {mixed} from the "
                "file layout (probably written by callkit <= 0.2.1); keep one layout per run dir"
            )
        self._readonly = not create
        self._pid = os.getpid()
        # guards the write connection and the whole write transaction
        self._wlock = threading.Lock()
        self._writer: sqlite3.Connection | None = None
        self._local = threading.local()
        self._seq = 0  # the seq the ledger has been synced up to
        self._dirty = False
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._init_db(run_meta or {})
        self._refuse_unfinished_migration()
        with self._lock:
            self._sync()

    # ---- writes ----

    def write(self, rec: Record, attempts: Sequence[RawAttempt]) -> None:
        """One transaction per record: allocate the call number (when ``rec.id`` is empty), insert one calls row and several blobs rows."""
        rec.request.headers = redact(rec.request.headers)
        rec.response.headers = redact(rec.response.headers)
        # compression and summary computation both happen outside the lock
        prepared = prepare_record(rec, record_files(rec, attempts))
        with self._txn() as db:
            if not rec.id:
                (n,) = db.execute("SELECT coalesce(max(num), 0) + 1 FROM calls").fetchone()
                rec.id = f"c_{n:06d}"
            insert_prepared(db, rec, prepared)
        self._dirty = True  # the background checkpoint thread only runs when it sees new writes
        with self._lock:
            self._sync()

    def prices(self) -> PriceTable:
        cur = self._reader().execute("SELECT model, price FROM prices")
        return {m: Price(**json.loads(p)) for m, p in cur}

    def merge_prices(self, prices: Mapping[str, Price]) -> PriceTable:
        with self._txn() as db:
            table = {
                m: Price(**json.loads(p)) for m, p in db.execute("SELECT model, price FROM prices")
            }
            clash = sorted(m for m, p in prices.items() if m in table and table[m] != p)
            if clash:
                raise CallkitError(
                    f"run dir {self.dir} already has different prices for {clash}; "
                    "use a new run dir to change prices"
                )
            added = {m: p for m, p in prices.items() if m not in table}
            insert_prices(db, added)
            return {**table, **added}

    def append_item(self, line: dict[str, Any]) -> None:
        with self._txn() as db:
            insert_item(db, line)

    # ---- reads ----

    def items(self) -> list[dict[str, Any]]:
        cur = self._reader().execute(
            "SELECT item_id, repeat, ok, error, started_at, finished_at, call_ids FROM items ORDER BY seq"
        )
        keys = ("item_id", "repeat", "ok", "error", "started_at", "finished_at")
        return [
            {**dict(zip(keys, r[:6], strict=True)), "ok": bool(r[2]), "call_ids": json.loads(r[6])}
            for r in cur
        ]

    def load(self, call_id: str) -> Record:
        seq = self._seq_of(call_id)
        rec = Record.from_dict(json.loads(self._blob(seq, "meta.json")))
        rec.body = self._blob(seq, rec.request.body_file)
        if rec.response.message_file:
            rec.message = json.loads(self._blob(seq, rec.response.message_file))
        return rec

    def load_meta(self, call_id: str) -> Record:
        return Record.from_dict(json.loads(self._blob(self._seq_of(call_id), "meta.json")))

    def load_message(self, call_id: str) -> Any:
        seq = self._seq_of(call_id)
        name = Record.from_dict(json.loads(self._blob(seq, "meta.json"))).response.message_file
        return json.loads(self._blob(seq, name)) if name else None

    def raw_response(self, call_id: str) -> tuple[str, bool]:
        seq = self._seq_of(call_id)
        name = Record.from_dict(json.loads(self._blob(seq, "meta.json"))).response.events_file
        if not name:
            return "", False
        return self._blob(seq, name).decode("utf-8"), name.endswith(".sse")

    def read_file(self, call_id: str, name: str) -> bytes:
        return self._blob(self._seq_of(call_id), name)

    def max_seq(self) -> int:
        """The current maximum seq of the ledger (used as a snapshot when the replay cache opens)."""
        (n,) = self._reader().execute("SELECT coalesce(max(seq), 0) FROM calls").fetchone()
        return n

    def replay_ids(self, key: str, max_seq: int) -> list[str]:
        """Records with the same replay key that succeeded and whose seq does not exceed the snapshot, ordered by call number."""
        cur = self._reader().execute(
            "SELECT id FROM calls WHERE replay_key = ? AND ok = 1 AND seq <= ? ORDER BY num",
            (key, max_seq),
        )
        return [r[0] for r in cur]

    def file_names(self, call_id: str) -> list[str]:
        cur = self._reader().execute(
            "SELECT name FROM blobs WHERE call_seq = ? ORDER BY name", (self._seq_of(call_id),)
        )
        return [r[0] for r in cur]

    def run_meta(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self._reader().execute("SELECT key, value FROM run")}

    def summaries(self) -> list[Row]:
        cur = self._reader().execute(f"SELECT {ROW_COLUMNS} FROM calls ORDER BY seq")
        return [summary_row(r) for r in cur]

    # ---- internals ----

    def _init_db(self, run_meta: dict[str, Any]) -> None:
        prepare_new_db(self._write_conn())
        with self._txn() as tx:
            create_schema(tx, run_meta)

    def _refuse_unfinished_migration(self) -> None:
        got = (
            self._reader().execute("SELECT value FROM run WHERE key = 'migrating_from'").fetchone()
        )
        if got is not None:
            raise CallkitError(
                f"run dir {self.dir} is still being migrated from {json.loads(got[0])}; "
                "run `callkit migrate` again to finish it before using the run dir"
            )

    def _sync(self) -> None:
        cur = self._reader().execute(
            f"SELECT {INDEX_COLUMNS} FROM calls WHERE seq > ? ORDER BY seq", (self._seq,)
        )
        for seq, cid, ok, sha, created, item_id, repeat, parent, fork_of, cost in cur:
            self._count(
                IndexRow(cid, bool(ok), sha, created, item_id, repeat, parent, fork_of, cost)
            )
            self._seq = seq

    def _seq_of(self, call_id: str) -> int:
        got = self._reader().execute("SELECT seq FROM calls WHERE id = ?", (call_id,)).fetchone()
        if got is None:
            raise KeyError(f"no call {call_id} in {self.path}")
        return got[0]

    def _blob(self, seq: int, name: str) -> bytes:
        got = (
            self._reader()
            .execute("SELECT codec, data FROM blobs WHERE call_seq = ? AND name = ?", (seq, name))
            .fetchone()
        )
        if got is None:
            raise KeyError(f"no {name} for call seq {seq} in {self.path}")
        return unpack(got[0], got[1])

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        if self._readonly:
            raise CallkitError(f"run dir {self.dir} was opened read-only")
        # Queue within this process first, then across processes with flock: the kernel wakes a waiting process as soon as the lock is released; with only SQLite's busy wait, a process that fails to get the lock
        # backs off and sleeps longer each time, so under high concurrency an individual process can starve for ten-odd seconds
        with self._wlock, dir_lock(self.dir):
            db = self._write_conn()
            _begin(db)
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    def _write_conn(self) -> sqlite3.Connection:
        self._check_pid()
        if self._writer is None:
            self._writer = self._connect()
            _start_checkpointer(self)
        return self._writer

    def _reader(self) -> sqlite3.Connection:
        self._check_pid()
        db = getattr(self._local, "db", None)
        if db is None:
            db = self._local.db = self._connect()
        return db

    def _connect(self) -> sqlite3.Connection:
        if self._readonly:
            target, uri = f"{self.path.resolve().as_uri()}?mode=ro", True
        else:
            target, uri = str(self.path), False
        db = sqlite3.connect(target, uri=uri, autocommit=True, check_same_thread=False)
        db.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute(f"PRAGMA wal_autocheckpoint={CHECKPOINT_PAGES}")
        return db

    def _check_pid(self) -> None:
        if os.getpid() != self._pid:
            raise CallkitError(
                f"run dir {self.dir} was opened in a parent process; SQLite connections cannot cross "
                "fork, so open it again in this process (e.g. use multiprocessing's spawn start method)"
            )


# database file -> the stores in this process that write it
_CHECKPOINTERS: dict[Path, weakref.WeakSet[SqliteStore]] = {}
_CHECKPOINTERS_LOCK = threading.Lock()


def _start_checkpointer(store: SqliteStore) -> None:
    """One background thread per database file in this process: every ``CHECKPOINT_EVERY_S`` seconds, if there have been new writes, it runs a PASSIVE checkpoint (which does not block writers
    or wait for readers). The thread holds only weak references to stores: once every store writing this database has been garbage collected, the thread exits. The interval is waited with Event.wait rather than
    time.sleep (so that if time.sleep is replaced elsewhere, the thread does not turn into a busy loop)."""
    path = store.path.resolve()
    with _CHECKPOINTERS_LOCK:
        stores = _CHECKPOINTERS.get(path)
        if stores is not None:
            stores.add(store)
            return
        stores = _CHECKPOINTERS[path] = weakref.WeakSet([store])

    def loop() -> None:
        db: sqlite3.Connection | None = None
        tick = threading.Event()
        while True:
            tick.wait(CHECKPOINT_EVERY_S)
            dirty = _take_dirty(stores, path)
            if dirty is None:
                break
            if not dirty:
                continue
            try:
                db = db or sqlite3.connect(path, autocommit=True, check_same_thread=False)
                db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchall()
            except sqlite3.Error:
                pass  # try again next round; the writer's own auto-checkpoint is the fallback
        if db is not None:
            db.close()

    threading.Thread(target=loop, name="callkit-checkpoint", daemon=True).start()


def _take_dirty(stores: weakref.WeakSet[SqliteStore], path: Path) -> bool | None:
    """Whether any store has written something new (clearing the flag along the way); returns None and unregisters when no store is left. It is a separate function so that the thread loop
    does not keep a strong reference to a store (the loop variable would hold on to the last one forever, and the thread could never exit)."""
    with _CHECKPOINTERS_LOCK:
        alive = list(stores)
        if not alive:
            del _CHECKPOINTERS[path]
            return None
    dirty = any(st._dirty for st in alive)
    for st in alive:
        st._dirty = False
    return dirty


def _begin(db: sqlite3.Connection) -> None:
    """``BEGIN IMMEDIATE``; when the write lock cannot be obtained, back off and retry, with a total time limit of ``WRITE_DEADLINE_S``."""
    deadline = time.monotonic() + WRITE_DEADLINE_S
    delay = 0.05
    while True:
        try:
            db.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as e:
            busy = getattr(e, "sqlite_errorcode", None) in (
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            )
            if not busy or time.monotonic() > deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
