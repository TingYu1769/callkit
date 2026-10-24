"""On-disk state for one run: run.json, index.jsonl, calls/<id>/.

Shared state (index, completed resumption keys, accumulated spend) is managed by RunStore under a single lock;
this is what makes client.call safe under multiple threads. Individual call directories do not affect each other, so file writes are not locked.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import re
import threading
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from callkit.config import Price, PriceTable
from callkit.errors import CallkitError
from callkit.pricing import dump_prices, load_prices
from callkit.record import IndexRow, Record
from callkit.transport import RawAttempt

RUN_FORMAT = "callkit.run/1"
GZIP_ABOVE = 1 << 20  # gzip request bodies over 1 MiB
PRICES_FILE = "prices.json"  # price table for the run directory
LOCK_FILE = ".callkit.lock"  # cross-process lock for the run directory
SECRET_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie"}
# Split header names on - and _; if any part matches one of these words, it is a secret header: x-api-key, api-key (Azure), x-goog-api-key, x-auth-token, etc.
SECRET_WORDS = {"key", "apikey", "token", "secret", "password", "auth", "credential", "credentials"}
REDACTED = "<redacted>"


class RunStore:
    def __init__(
        self, run_dir: str | Path, *, run_meta: dict[str, Any] | None = None, create: bool = True
    ) -> None:
        """Open or create a run directory; when creating, write run.json (config, start time, cap, callkit commit).

        An existing directory is treated as resuming: restore the index, completed resumption keys, and accumulated spend from index.jsonl; run.json is not overwritten.
        ``create=False`` opens read-only: it does not create the directory or write run.json (for CLI inspection and replay sources).
        """
        self.dir = Path(run_dir)
        self.calls = self.dir / "calls"
        self._lock = threading.Lock()
        run_json = self.dir / "run.json"
        if create:
            self.calls.mkdir(parents=True, exist_ok=True)
        if create and not run_json.exists():
            _write_json(
                run_json, {"format": RUN_FORMAT, "created_at": now_iso(), **(run_meta or {})}
            )
        self._index = self.dir / "index.jsonl"
        self._offset = 0  # bytes already read from index.jsonl
        self._rows: list[IndexRow] = []
        self._done: dict[tuple[str, int], str] = {}
        self._spent = 0.0
        with self._lock:
            self._sync()
        self._next = _max_call_number(self.calls) + 1 if self.calls.exists() else 1

    def new_call_id(self) -> str:
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

    def write(self, rec: Record, attempts: Sequence[RawAttempt]) -> None:
        """Write calls/<id>/: request.json, response.sse, message.json, attempts/<n>.sse, then meta.json;
        then append one row to index.jsonl. Fills in the file names on ``rec`` in place and redacts request and response headers."""
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
            _write_json(d / "message.json", rec.message)
            rec.response.message_file = "message.json"
        _write_json(d / "meta.json", rec.to_dict())
        self._append(_index_row(rec))

    def load_meta(self, call_id: str) -> Record:
        """Read only meta.json, without the request body or response (used when replay cache checks candidates)."""
        return Record.from_dict(
            json.loads((self.calls / call_id / "meta.json").read_text(encoding="utf-8"))
        )

    def load(self, call_id: str) -> Record:
        """Read one record back, including its body and message."""
        d = self.calls / call_id
        rec = Record.from_dict(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        raw = (d / rec.request.body_file).read_bytes()
        rec.body = gzip.decompress(raw) if rec.request.body_file.endswith(".gz") else raw
        if rec.response.message_file:
            rec.message = json.loads((d / rec.response.message_file).read_text(encoding="utf-8"))
        return rec

    def index(self) -> list[IndexRow]:
        with self._lock:
            self._sync()
            return list(self._rows)

    def done(self) -> dict[tuple[str, int], str]:
        """Completed resumption keys (item_id, repeat) -> request hash."""
        with self._lock:
            self._sync()
            return dict(self._done)

    def spent_usd(self) -> float:
        """Total spend across all attempts (failed attempts are counted too). Attempts written to the same directory by other processes are also included."""
        with self._lock:
            self._sync()
            return self._spent

    def merge_prices(self, prices: Mapping[str, Price]) -> PriceTable:
        """Merge ``prices`` into the run directory's prices.json by model and return the full merged table. Raises CallkitError if the same model already has a different price: records in the same directory are priced with the same table, and the first written price wins."""
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
                _write_json(path, dump_prices(table))
            return table

    def _append(self, row: IndexRow) -> None:
        with self._lock:
            append_line(self._index, row.to_dict())
            self._sync()

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
            self._rows.append(row)
            self._count(row)
        self._offset += end

    def _count(self, row: IndexRow) -> None:
        if row.ok and row.item_id is not None and row.repeat is not None:
            self._done[(row.item_id, row.repeat)] = row.sha256
        self._spent += row.cost_usd or 0.0


@contextmanager
def dir_lock(run_dir: Path) -> Iterator[None]:
    """Cross-process exclusive lock for the run directory (flock). Appending to the ledger and merging price tables are done under this lock, so multiple processes writing the same directory do not interleave."""
    with (run_dir / LOCK_FILE).open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_line(path: Path, obj: Any) -> None:
    """Append one line to a JSONL file, under the run directory lock. If the file has a trailing partial line (a process crashed while writing), break it off first; readers skip it as an unreadable line. Because the lock is held, the broken line cannot be one that another process is currently writing."""
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
    with dir_lock(path.parent), path.open("a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = b"\n" + data
        f.write(data)


def is_secret_header(name: str) -> bool:
    lower = name.lower()
    return lower in SECRET_HEADERS or not SECRET_WORDS.isdisjoint(re.split(r"[-_]", lower))


def redact(headers: Mapping[str, str], also: str | None = None) -> dict[str, str]:
    """Redact secret headers. ``also`` is an extra header name to redact (an endpoint-specific auth header whose name may not look secret)."""
    extra = also.lower() if also else None
    return {
        k: (REDACTED if is_secret_header(k) or k.lower() == extra else v)
        for k, v in headers.items()
    }


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _index_row(rec: Record) -> IndexRow:
    return IndexRow(
        id=rec.id,
        ok=rec.outcome.ok,
        sha256=rec.request.sha256,
        created_at=rec.timing.started_at,
        item_id=rec.item_id,
        repeat=rec.repeat,
        parent=rec.parent,
        fork_of=rec.fork_of,
        cost_usd=rec.cost_usd,
    )


def _write_body(call_dir: Path, body: bytes) -> str:
    if len(body) > GZIP_ABOVE:
        (call_dir / "request.json.gz").write_bytes(gzip.compress(body, mtime=0))
        return "request.json.gz"
    (call_dir / "request.json").write_bytes(body)
    return "request.json"


def _write_attempt(call_dir: Path, raw: RawAttempt, stem: str) -> str | None:
    """If there is an event stream, write <stem>.sse (verbatim lines); if there is no event stream (non-200 error body, full JSON of a non-streaming response), write <stem>.body."""
    if raw.lines:
        name, text = f"{stem}.sse", "\n".join(line for _, line in raw.lines) + "\n"
    elif raw.error_body is not None:
        name, text = f"{stem}.body", raw.error_body
    else:
        return None
    path = call_dir / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return name


def _write_json(path: Path, obj: Any) -> None:
    # Temporary file name includes PID and thread id so multiple processes initializing the same directory cannot overwrite each other.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _max_call_number(calls: Path) -> int:
    nums = [int(p.name[2:]) for p in calls.iterdir() if p.name[:2] == "c_" and p.name[2:].isdigit()]
    return max(nums, default=0)
