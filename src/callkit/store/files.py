"""The older file layout: run.json, index.jsonl, prices.json, and one directory per call under calls/<id>/.

Since 0.2.2, new directories default to SQLite (``callkit.store.sqlite``); existing older directories keep being read and written here. Shared state (index, resumption keys,
accumulated spend) is managed by RunStore under a single lock; this is what makes client.call safe under multiple threads. Individual call directories do not
affect each other, so file writes are not locked.
"""

from __future__ import annotations

import gzip
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from callkit.config import Price, PriceTable
from callkit.errors import CallkitError
from callkit.pricing import dump_prices, load_prices
from callkit.record import IndexRow, Record
from callkit.store.base import RunStore
from callkit.store.common import (
    append_line,
    attempt_text,
    dir_lock,
    index_row,
    now_iso,
    redact,
    write_json,
)
from callkit.transport import RawAttempt

RUN_FORMAT = "callkit.run/1"
GZIP_ABOVE = 1 << 20  # gzip request bodies over 1 MiB
PRICES_FILE = "prices.json"  # price table for the run directory
ITEMS_FILE = "items.jsonl"  # the scheduler's item ledger


class FileStore(RunStore):
    def __init__(
        self, run_dir: str | Path, *, run_meta: dict[str, Any] | None = None, create: bool = True
    ) -> None:
        """Open or create a run directory in the older layout; when creating, write run.json (config, start time, cap, callkit commit).

        An existing directory is treated as resuming: restore the index, completed resumption keys, and accumulated spend from index.jsonl; run.json is not overwritten.
        ``create=False`` opens read-only: it does not create the directory or write run.json (for CLI inspection and replay sources).
        """
        super().__init__(run_dir)
        self.calls = self.dir / "calls"
        run_json = self.dir / "run.json"
        if create:
            self.calls.mkdir(parents=True, exist_ok=True)
        if create and not run_json.exists():
            write_json(
                run_json, {"format": RUN_FORMAT, "created_at": now_iso(), **(run_meta or {})}
            )
        self._index = self.dir / "index.jsonl"
        self._offset = 0  # bytes already read from index.jsonl
        with self._lock:
            self._sync()
        self._next = _max_call_number(self.calls) + 1 if self.calls.exists() else 1

    def write(self, rec: Record, attempts: Sequence[RawAttempt]) -> None:
        """Write calls/<id>/: request.json, response.sse, message.json, attempts/<n>.sse, then meta.json;
        then append one row to index.jsonl. If ``rec.id`` is empty, create the directory first to reserve a call number."""
        if not rec.id:
            rec.id = self._claim_id()
        d = self.calls / rec.id
        d.mkdir(parents=True, exist_ok=True)
        rec.request.body_file = _write_body(d, rec.body)
        rec.request.headers = redact(rec.request.headers)
        rec.response.headers = redact(rec.response.headers)
        if attempts:
            rec.response.events_file = _write_attempt(d, attempts[-1], "response")
            by_n = {a.n: a for a in rec.attempts}
            for raw in attempts[:-1]:
                name = _write_attempt(d, raw, f"attempts/{raw.n}")
                if raw.n in by_n:
                    by_n[raw.n].events_file = name
        if rec.message is not None:
            write_json(d / "message.json", rec.message)
            rec.response.message_file = "message.json"
        write_json(d / "meta.json", rec.to_dict())
        with self._lock:
            append_line(self._index, index_row(rec).to_dict())
            self._sync()

    def load_meta(self, call_id: str) -> Record:
        return Record.from_dict(
            json.loads((self.calls / call_id / "meta.json").read_text(encoding="utf-8"))
        )

    def load(self, call_id: str) -> Record:
        d = self.calls / call_id
        rec = Record.from_dict(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        raw = (d / rec.request.body_file).read_bytes()
        rec.body = gzip.decompress(raw) if rec.request.body_file.endswith(".gz") else raw
        if rec.response.message_file:
            rec.message = json.loads((d / rec.response.message_file).read_text(encoding="utf-8"))
        return rec

    def load_message(self, call_id: str) -> Any:
        name = self.load_meta(call_id).response.message_file
        if not name:
            return None
        return json.loads((self.calls / call_id / name).read_text(encoding="utf-8"))

    def raw_response(self, call_id: str) -> tuple[str, bool]:
        name = self.load_meta(call_id).response.events_file
        if not name:
            return "", False
        return (self.calls / call_id / name).read_text(encoding="utf-8"), name.endswith(".sse")

    def read_file(self, call_id: str, name: str) -> bytes:
        return (self.calls / call_id / name).read_bytes()

    def file_names(self, call_id: str) -> list[str]:
        d = self.calls / call_id
        if not d.is_dir():
            raise FileNotFoundError(d)
        return sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())

    def run_meta(self) -> dict[str, Any]:
        return json.loads((self.dir / "run.json").read_text(encoding="utf-8"))

    def prices(self) -> PriceTable:
        path = self.dir / PRICES_FILE
        return load_prices(path) if path.exists() else {}

    def merge_prices(self, prices: Mapping[str, Price]) -> PriceTable:
        # Reading, comparing, and writing must happen under one cross-process lock.
        with self._lock, dir_lock(self.dir):
            path = self.dir / PRICES_FILE
            table = load_prices(path) if path.exists() else {}
            clash = sorted(m for m, p in prices.items() if m in table and table[m] != p)
            if clash:
                raise CallkitError(
                    f"run dir {self.dir} already has different prices for {clash}; "
                    "use a new run dir to change prices, or delete its prices.json first"
                )
            added = {m: p for m, p in prices.items() if m not in table}
            if added:
                table.update(added)
                write_json(path, dump_prices(table))
            return table

    def append_item(self, line: dict[str, Any]) -> None:
        append_line(self.dir / ITEMS_FILE, line)

    def items(self) -> list[dict[str, Any]]:
        """Split into lines by bytes and decode line by line: half a Chinese character (a crash mid-write) spoils only that one line, and characters such as U+2028 inside an item_id do not split a line.
        Unreadable lines trigger a warning and are skipped."""
        path = self.dir / ITEMS_FILE
        if not path.exists():
            return []
        rows = []
        for line in path.read_bytes().split(b"\n"):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                warnings.warn(f"skipped an unreadable line in {path}", RuntimeWarning, stacklevel=4)
        return rows

    def _claim_id(self) -> str:
        """Create a directory first to reserve a call number; if the number is taken (another process or RunStore is also writing this directory), move to the next one."""
        while True:
            with self._lock:
                n = self._next
                self._next += 1
            cid = f"c_{n:06d}"
            try:
                (self.calls / cid).mkdir()
            except FileExistsError:
                continue
            return cid

    def _sync(self) -> None:
        """Read new complete lines from index.jsonl, including lines written by other processes (multiple processes writing the same directory share one ledger).
        A trailing half-written line is left for the next call; unreadable lines trigger a warning and are skipped. The caller holds the lock."""
        try:
            with self._index.open("rb") as f:
                f.seek(self._offset)
                data = f.read()
        except FileNotFoundError:
            return
        end = data.rfind(b"\n") + 1
        for line in data[:end].split(b"\n"):  # Split only on \n.
            if not line.strip():
                continue
            try:
                row = IndexRow.from_dict(json.loads(line))
            except (ValueError, KeyError, TypeError):
                warnings.warn(
                    f"skipped an unreadable line in {self._index} (probably half-written by a crashed process)",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue
            self._count(row)
        self._offset += end


def _write_body(call_dir: Path, body: bytes) -> str:
    if len(body) > GZIP_ABOVE:
        (call_dir / "request.json.gz").write_bytes(gzip.compress(body, mtime=0))
        return "request.json.gz"
    (call_dir / "request.json").write_bytes(body)
    return "request.json"


def _write_attempt(call_dir: Path, raw: RawAttempt, stem: str) -> str | None:
    got = attempt_text(raw, stem)
    if got is None:
        return None
    name, text = got
    path = call_dir / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return name


def _max_call_number(calls: Path) -> int:
    nums = [int(p.name[2:]) for p in calls.iterdir() if p.name[:2] == "c_" and p.name[2:].isdigit()]
    return max(nums, default=0)
