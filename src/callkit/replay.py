"""Replay cache: when rerunning an experiment, a request exactly identical to a successful call in an old run takes the old response directly and avoids another paid call.

"Identical" means the URL, the organization/project/workspace request headers, and the request hash (raw request body bytes + behavior-affecting request headers) are all the same. When the same hash appears for the kth time in this process, take the kth successful record for that hash from the old run (sorted by call id); once old records are exhausted, fall back to the normal network call. That way a rerun of a sampling experiment can recover the same distinct samples. Opening the old run takes a snapshot; records written after opening are not replayed (if the same directory is both the old run and the new run, it will not replay records it just wrote). Clients in the same process that write the same new run and replay the same old run share this snapshot and counters.

On open it reads only ``index.jsonl`` and indexes by request hash. The URL and headers require ``meta.json``, so they are read only when a request hash hits:
read candidates with that hash in call order and stop at the first match. Reading ``meta.json`` for every record to build the table would take tens of minutes when there are many records on a spinning disk.
"""

from __future__ import annotations

import threading
import warnings
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from callkit.errors import CallkitError
from callkit.record import Record
from callkit.store import RunStore

# Headers that select organization, project, or workspace: the same request body sent to a different organization should not count as identical.
SCOPE_HEADERS = ("openai-organization", "openai-project", "anthropic-workspace-id")
_SHARED: dict[tuple[str, str], ReplayCache] = {}
_SHARED_LOCK = threading.Lock()


class ReplayCache:
    def __init__(self, run: str | Path) -> None:
        path = Path(run).resolve()
        if not (path / "run.json").exists():
            raise CallkitError(f"replay source {run} is not a callkit run dir (no run.json)")
        self.dir = path
        # Read only: do not create the directory or write run.json
        self._store = RunStore(path, create=False)
        # Request hash -> successful records not yet replayed (sorted by call id)
        self._ids: dict[str, list[str]] = defaultdict(list)
        for row in sorted(self._store.index(), key=lambda r: r.id):
            if row.ok:
                self._ids[row.sha256].append(row.id)
        self._keys: dict[
            str, str | None
        ] = {}  # Call id -> replay key; meta.json is read only when needed; None if unreadable
        self._lock = threading.Lock()

    @classmethod
    def shared(cls, run: str | Path, into: str | Path) -> ReplayCache:
        """Shared per old run -> new run pair: clients writing the same new run share counters; different new runs each start from the beginning."""
        key = (str(Path(run).resolve()), str(Path(into).resolve()))
        with _SHARED_LOCK:
            found = _SHARED.get(key)
            if found is None:
                found = _SHARED[key] = cls(run)
            return found

    def take(self, url: str, headers: Mapping[str, str], sha256: str) -> Record | None:
        """Return the next successful record that has not been replayed for the same URL and request; return None if none remain."""
        key = replay_key(url, headers, sha256)
        with self._lock:
            pending = list(self._ids.get(sha256, ()))
        # Read in call order; stop at the first record whose URL and headers match
        for call_id in pending:
            if call_id not in self._keys:
                # Reading meta.json does not hold the lock
                self._keys[call_id] = self._key_of(call_id, sha256)
            if self._keys[call_id] != key:
                continue
            with self._lock:
                ids = self._ids.get(sha256, [])
                if call_id in ids:  # Another thread may have just taken it; keep looking
                    ids.remove(call_id)
                    break
        else:
            return None
        return self._store.load(call_id)

    def _key_of(self, call_id: str, sha256: str) -> str | None:
        try:
            meta = self._store.load_meta(call_id)
        except (OSError, ValueError, KeyError, TypeError) as e:
            warnings.warn(
                f"replay source {self.dir}: cannot read {call_id}/meta.json ({e}); not replaying it",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return replay_key(meta.request.url, meta.request.headers, sha256)

    def ref(self, rec: Record) -> str:
        """Value written to the new record label replay_of: old run directory:call id."""
        return f"{self.dir}:{rec.id}"

    def raw_response(self, rec: Record) -> tuple[str, bool]:
        """Old record's raw response and whether it is an event stream."""
        name = rec.response.events_file
        if not name:
            return "", False
        path = self.dir / "calls" / rec.id / name
        return path.read_text(encoding="utf-8"), name.endswith(".sse")


def replay_key(url: str, headers: Mapping[str, str], sha256: str) -> str:
    """Replay matching key: full URL (host, path, query string) + the organization/project/workspace request headers + request hash. Comparing only the request hash would treat
    requests with a different gateway or organization as the same."""
    lower = {k.lower(): v for k, v in headers.items()}
    picked = "|".join(f"{h}={lower.get(h, '')}" for h in SCOPE_HEADERS)
    return f"{url}|{picked}|{sha256}"
