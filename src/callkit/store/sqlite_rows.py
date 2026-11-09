"""Table schema and row encoding for the SQLite layout: the CREATE statements, splitting one record into a calls row plus several blobs rows, and compression of large fields."""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from callkit.config import Price
from callkit.errors import CallkitError
from callkit.record import Record
from callkit.store.common import attempt_text, now_iso, replay_key
from callkit.summary import Row, row_of
from callkit.transport import RawAttempt

RUN_FORMAT = "callkit.run/2"
SCHEMA_VERSION = 1
# The page size only takes effect when the database is created. With 4 KB pages, each time a record is written, its summary row and several hundred KB of large-field pages are appended to the end of the file in turn, so the summary table's leaf pages
# end up scattered across the whole file, and scanning the summaries on a spinning disk takes thousands of seeks; a 64 KB page holds about 150 summary rows (measured on 11,000 records: ls 9.0 s -> 0.8 s,
# a random single-record read 33 ms -> 16 ms).
PAGE_SIZE = 65536
COMPRESS_ABOVE = 4096  # bytes; compressing smaller fields is not worth it

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS run (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS prices (model TEXT PRIMARY KEY, price TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS calls (
        seq INTEGER PRIMARY KEY,
        num INTEGER NOT NULL,  -- numeric part of the call id; id is the unique key
        id TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        api TEXT NOT NULL,
        source TEXT NOT NULL,
        model TEXT,
        ok INTEGER NOT NULL,
        transport_end TEXT,
        protocol TEXT,
        stop_reason TEXT,
        reason TEXT,
        sha256 TEXT NOT NULL,
        replay_key TEXT NOT NULL,
        session TEXT,
        item_id TEXT,
        repeat INTEGER,
        parent TEXT,
        fork_of TEXT,
        replay_of TEXT,
        cost_usd REAL,
        usage_missing INTEGER NOT NULL DEFAULT 0,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        said TEXT NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX IF NOT EXISTS calls_replay ON calls(replay_key, num) WHERE ok = 1",
    """CREATE TABLE IF NOT EXISTS items (
        seq INTEGER PRIMARY KEY,
        item_id TEXT NOT NULL,
        repeat INTEGER NOT NULL,
        ok INTEGER NOT NULL,
        error TEXT,
        started_at TEXT,
        finished_at TEXT,
        call_ids TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS items_key ON items(item_id, repeat)",
    """CREATE TABLE IF NOT EXISTS blobs (
        call_seq INTEGER NOT NULL,
        name TEXT NOT NULL,
        codec TEXT NOT NULL,
        size INTEGER NOT NULL,
        data BLOB NOT NULL,
        PRIMARY KEY (call_seq, name)
    )""",
)
CALL_COLUMNS = (
    "num, id, created_at, api, source, model, ok, transport_end, protocol, stop_reason, reason, "
    "sha256, replay_key, session, item_id, repeat, parent, fork_of, replay_of, cost_usd, "
    "usage_missing, input_tokens, cache_read_tokens, output_tokens, said"
)
INDEX_COLUMNS = "seq, id, ok, sha256, created_at, item_id, repeat, parent, fork_of, cost_usd"
ROW_COLUMNS = (
    "id, created_at, api, source, model, ok, stop_reason, reason, input_tokens, cache_read_tokens, "
    "output_tokens, cost_usd, parent, fork_of, item_id, repeat, session, usage_missing, said"
)


def record_files(rec: Record, attempts: Sequence[RawAttempt]) -> dict[str, bytes]:
    """The fields of a record to store besides meta.json, keyed by file name in the older layout; also fills in the file names on ``rec``."""
    rec.request.body_file = "request.json"
    out = {"request.json": rec.body}
    if attempts:
        rec.response.events_file = put_file(out, attempt_text(attempts[-1], "response"))
        by_n = {a.n: a for a in rec.attempts}
        for raw in attempts[:-1]:
            name = put_file(out, attempt_text(raw, f"attempts/{raw.n}"))
            if raw.n in by_n:
                by_n[raw.n].events_file = name
    if rec.message is not None:
        out["message.json"] = json.dumps(rec.message, ensure_ascii=False, indent=1).encode()
        rec.response.message_file = "message.json"
    return out


def put_file(out: dict[str, bytes], got: tuple[str, str] | None) -> str | None:
    if got is None:
        return None
    name, text = got
    out[name] = text.encode("utf-8")
    return name


@dataclass
class Prepared:
    """The part computed before taking the write lock: the summary columns (without the call number) and the compressed large fields (without meta.json, which has to wait until a call number is assigned)."""

    values: tuple[Any, ...]
    blobs: list[tuple[str, str, int, bytes]]  # (name, codec, size, data)


def prepare_record(rec: Record, files: dict[str, bytes]) -> Prepared:
    row = row_of(rec)
    values = (
        rec.timing.started_at,
        rec.api,
        rec.source,
        row.model,
        int(rec.outcome.ok),
        rec.outcome.transport_end,
        rec.outcome.protocol,
        rec.outcome.stop_reason,
        row.reason,
        rec.request.sha256,
        replay_key(rec.request.url, rec.request.headers, rec.request.sha256),
        row.session,
        rec.item_id,
        rec.repeat,
        rec.parent,
        rec.fork_of,
        rec.tags.get("replay_of"),
        rec.cost_usd,
        int(row.usage_missing),
        row.input_tokens,
        row.cache_read_tokens,
        row.output_tokens,
        row.said,
    )
    return Prepared(values, [(name, *pack(data)) for name, data in files.items()])


def insert_prepared(db: sqlite3.Connection, rec: Record, prepared: Prepared) -> None:
    """Insert one calls row and several blobs rows inside the write transaction. ``rec.id`` must already be set by now."""
    values = (call_num(rec.id), rec.id, *prepared.values)
    marks = ", ".join("?" * len(values))
    (seq,) = db.execute(
        f"INSERT INTO calls ({CALL_COLUMNS}) VALUES ({marks}) RETURNING seq", values
    ).fetchone()
    meta = json.dumps(rec.to_dict(), ensure_ascii=False, indent=1).encode()
    db.executemany(
        "INSERT INTO blobs (call_seq, name, codec, size, data) VALUES (?, ?, ?, ?, ?)",
        [(seq, *b) for b in [*prepared.blobs, ("meta.json", *pack(meta))]],
    )


def insert_record(db: sqlite3.Connection, rec: Record, files: dict[str, bytes]) -> None:
    insert_prepared(db, rec, prepare_record(rec, files))


def call_num(call_id: str) -> int:
    if call_id[:2] != "c_" or not call_id[2:].isdigit():
        raise CallkitError(f"bad call id {call_id!r}; expected c_ followed by digits")
    return int(call_id[2:])


def pack(data: bytes) -> tuple[str, int, bytes]:
    if len(data) > COMPRESS_ABOVE:
        packed = zlib.compress(data, 6)
        if len(packed) < len(data):
            return "zlib", len(data), packed
    return "raw", len(data), data


def unpack(codec: str, data: bytes) -> bytes:
    if codec == "raw":
        return bytes(data)
    if codec == "zlib":
        try:
            return zlib.decompress(data)
        except zlib.error as e:
            raise ValueError(f"corrupt zlib blob: {e}") from e
    raise ValueError(f"unknown blob codec {codec!r}")


def summary_row(r: tuple[Any, ...]) -> Row:
    cid, started, api, source, model, ok, stop, reason, inp, cread, out, cost = r[:12]
    parent, fork_of, item_id, repeat, session, missing, said = r[12:]
    return Row(
        id=cid,
        started_at=started,
        api=api,
        source=source,
        model=model or "",
        ok=bool(ok),
        stop=stop,
        reason=reason or "",
        input_tokens=inp,
        cache_read_tokens=cread,
        output_tokens=out,
        cost_usd=cost,
        parent=parent,
        fork_of=fork_of,
        item=None if item_id is None else f"{item_id}#{repeat}",
        session=session,
        usage_missing=bool(missing),
        said=said,
    )


def prepare_new_db(db: sqlite3.Connection) -> None:
    """Call before creating tables in a new database: set the page size (setting it after tables exist has no effect) and WAL. Calling it on an existing database is harmless."""
    db.execute(f"PRAGMA page_size={PAGE_SIZE}")
    db.execute("PRAGMA journal_mode=WAL")


def create_schema(db: sqlite3.Connection, run_meta: Mapping[str, Any]) -> None:
    """Create the tables inside the write transaction; when the run table is empty, write the format name, creation time, and ``run_meta`` (existing values are not overwritten)."""
    for stmt in SCHEMA:
        db.execute(stmt)
    (have,) = db.execute("SELECT count(*) FROM run").fetchone()
    if not have:
        meta = {"format": RUN_FORMAT, "created_at": now_iso(), **run_meta}
        db.executemany(
            "INSERT INTO run (key, value) VALUES (?, ?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items()],
        )
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def insert_item(db: sqlite3.Connection, line: Mapping[str, Any]) -> None:
    db.execute(
        "INSERT INTO items (item_id, repeat, ok, error, started_at, finished_at, call_ids) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            line["item_id"],
            line["repeat"],
            int(line["ok"]),
            line.get("error"),
            line.get("started_at"),
            line.get("finished_at"),
            json.dumps(line.get("call_ids", [])),
        ),
    )


def insert_prices(db: sqlite3.Connection, prices: Mapping[str, Price]) -> None:
    db.executemany(
        "INSERT INTO prices (model, price) VALUES (?, ?)",
        [(m, json.dumps(asdict(p), sort_keys=True)) for m, p in sorted(prices.items())],
    )
