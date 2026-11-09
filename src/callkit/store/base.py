"""``RunStore``: picks a backend from the layout of the run directory, and holds the ledger cache shared by both backends.

Backend selection: if ``run.db`` exists, use SQLite; if there is no ``run.db`` but there is a ``run.json``, use the older file layout;
if neither exists, create a new directory with SQLite. ``RunStore(...)`` returns an instance of the matching subclass directly, so callers use it as before.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self, cast

from callkit.config import Price, PriceTable
from callkit.errors import CallkitError
from callkit.record import IndexRow, Record
from callkit.transport import RawAttempt

DB_FILE = "run.db"
LEGACY_RUN_FILE = "run.json"


def is_run_dir(path: str | Path) -> bool:
    p = Path(path)
    return (p / DB_FILE).exists() or (p / LEGACY_RUN_FILE).exists()


class RunStore:
    """One run directory. Subclasses implement the on-disk storage; the in-memory cache of the index, resumption keys, and accumulated spend lives here, and each subclass's ``_sync()`` fills it
    incrementally from its own ledger (rows written by other processes count too: multiple processes writing the same directory share one ledger)."""

    dir: Path

    def __new__(
        cls, run_dir: str | Path, *, run_meta: dict[str, Any] | None = None, create: bool = True
    ) -> Self:
        """Factory: ``RunStore(...)`` returns a FileStore or SqliteStore according to the directory layout; when a subclass is used directly, that subclass is returned."""
        target: type[RunStore] = _backend(Path(run_dir), create) if cls is RunStore else cls
        return cast(Self, object.__new__(target))

    def __init__(
        self, run_dir: str | Path, *, run_meta: dict[str, Any] | None = None, create: bool = True
    ) -> None:
        self.dir = Path(run_dir)
        self._lock = threading.Lock()
        self._rows: list[IndexRow] = []
        self._done: dict[tuple[str, int], str] = {}
        self._spent = 0.0

    # ---- implemented by subclasses ----

    def write(self, rec: Record, attempts: Sequence[RawAttempt]) -> None:
        """Persist one record. If ``rec.id`` is empty, allocate a call number here and fill it back into ``rec.id``; also fill in the file names on ``rec``
        and redact request and response headers."""
        raise NotImplementedError

    def load(self, call_id: str) -> Record:
        """Read one record back, including its body and message."""
        raise NotImplementedError

    def load_meta(self, call_id: str) -> Record:
        """Read only the record itself, without the request body or response."""
        raise NotImplementedError

    def load_message(self, call_id: str) -> Any:
        """Read only the assembled response; returns None if there is none."""
        raise NotImplementedError

    def raw_response(self, call_id: str) -> tuple[str, bool]:
        """The raw response of the last attempt, and whether it is an event stream. Returns ("", False) when there is no raw text."""
        raise NotImplementedError

    def read_file(self, call_id: str, name: str) -> bytes:
        """Fetch one field of a record by its file name in the older layout (meta.json, request.json, response.sse, attempts/1.body, ...),
        returned as is. Raises KeyError or FileNotFoundError if the field does not exist."""
        raise NotImplementedError

    def file_names(self, call_id: str) -> list[str]:
        """Which fields a record has stored (as file names of the older layout, sorted)."""
        raise NotImplementedError

    def run_meta(self) -> dict[str, Any]:
        """Run information (format name, creation time, endpoint, cap, callkit version, etc.)."""
        raise NotImplementedError

    def prices(self) -> PriceTable:
        """The run directory's current price table (read only)."""
        raise NotImplementedError

    def merge_prices(self, prices: Mapping[str, Price]) -> PriceTable:
        """Merge ``prices`` into the run directory's price table by model and return the full merged table. Raises
        CallkitError if the same model already has a different price: records in the same directory are priced with the same table, and the first written price wins."""
        raise NotImplementedError

    def append_item(self, line: dict[str, Any]) -> None:
        """Append one line to the scheduler's item ledger: item_id, repeat, ok, error, started_at, finished_at, call_ids."""
        raise NotImplementedError

    def items(self) -> list[dict[str, Any]]:
        """All lines of the item ledger, in write order."""
        raise NotImplementedError

    def items_done(self) -> set[tuple[str, int]]:
        """The (item_id, repeat) pairs that have succeeded in the item ledger: one successful line is enough, and a later failure does not erase it."""
        return {(str(r["item_id"]), int(r["repeat"])) for r in self.items() if r.get("ok")}

    def _sync(self) -> None:
        """Read in the lines newly added to the ledger. The caller holds ``self._lock``."""
        raise NotImplementedError

    # ---- shared ----

    def index(self) -> list[IndexRow]:
        with self._lock:
            self._sync()
            return list(self._rows)

    def done(self) -> dict[tuple[str, int], str]:
        """Completed resumption keys (item_id, repeat) -> request hash; for the same key, a later write overrides an earlier one."""
        with self._lock:
            self._sync()
            return dict(self._done)

    def spent_usd(self) -> float:
        """Total spend across all attempts (failed attempts are counted too). Attempts written to the same directory by other processes are also included."""
        with self._lock:
            self._sync()
            return self._spent

    def summaries(self) -> list[Any]:
        """Summary rows (``callkit.summary.Row``) used by ``callkit ls`` and ``tree``. By default reads records one by one; unreadable ones trigger a warning and are skipped."""
        from callkit.summary import row_of

        rows = []
        for row in self.index():
            try:
                rows.append(row_of(self.load(row.id)))
            except (OSError, ValueError, KeyError, TypeError) as e:
                warnings.warn(
                    f"skipped unreadable record {row.id}: {type(e).__name__}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return rows

    def _count(self, row: IndexRow) -> None:
        self._rows.append(row)
        if row.ok and row.item_id is not None and row.repeat is not None:
            self._done[(row.item_id, row.repeat)] = row.sha256
        self._spent += row.cost_usd or 0.0


def _backend(path: Path, create: bool) -> type[RunStore]:
    from callkit.store.files import FileStore
    from callkit.store.sqlite import SqliteStore

    if (path / DB_FILE).exists():
        return SqliteStore
    if (path / LEGACY_RUN_FILE).exists():
        return FileStore
    if create:
        return SqliteStore
    raise CallkitError(f"{path} is not a callkit run dir (no {DB_FILE} or {LEGACY_RUN_FILE})")
