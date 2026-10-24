"""Replay cache: when rerunning an experiment, a request identical to a successful call in an old run gets the old response without spending money again.

"Identical" means the URL, the organization/project/workspace headers, and the request hash (raw request body bytes + behavior-changing headers) all match. When a hash appears for the k-th time in this process, take the k-th successful record for that hash from the old run (ordered by call id); when old records are exhausted, go online as usual: replaying a sampling experiment gets back the original different samples. On open, snapshot the old run once; records written afterwards are not replayed (if the same directory is both the old run and the new run, it will not replay what it just wrote). Clients in the same process writing the same new run and replaying the same old run share this snapshot and counters.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Mapping
from pathlib import Path

from callkit.errors import CallkitError
from callkit.record import Record
from callkit.store import RunStore

# Headers that select organization, project, and workspace: the same request body sent to different organizations is not identical.
SCOPE_HEADERS = ("openai-organization", "openai-project", "anthropic-workspace-id")
_SHARED: dict[tuple[str, str], ReplayCache] = {}
_SHARED_LOCK = threading.Lock()


class ReplayCache:
    def __init__(self, run: str | Path) -> None:
        path = Path(run).resolve()
        if not (path / "run.json").exists():
            raise CallkitError(f"replay source {run} is not a callkit run dir (no run.json)")
        self.dir = path
        # read-only: do not create the directory or write run.json
        self._store = RunStore(path, create=False)
        self._ids: dict[str, deque[str]] = defaultdict(deque)
        for row in sorted(self._store.index(), key=lambda r: r.id):
            if row.ok:
                meta = self._store.load_meta(row.id)
                self._ids[replay_key(meta.request.url, meta.request.headers, row.sha256)].append(
                    row.id
                )
        self._lock = threading.Lock()

    @classmethod
    def shared(cls, run: str | Path, into: str | Path) -> ReplayCache:
        """Shared per old run -> new run pair: clients writing the same new run share counters, and
        different new runs each start from the beginning."""
        key = (str(Path(run).resolve()), str(Path(into).resolve()))
        with _SHARED_LOCK:
            found = _SHARED.get(key)
            if found is None:
                found = _SHARED[key] = cls(run)
            return found

    def take(self, url: str, headers: Mapping[str, str], sha256: str) -> Record | None:
        """Return the next not yet replayed successful record for the same URL and request; return None if none are left."""
        with self._lock:
            ids = self._ids.get(replay_key(url, headers, sha256))
            if not ids:
                return None
            call_id = ids.popleft()
        return self._store.load(call_id)

    def ref(self, rec: Record) -> str:
        """Value written into the new record's replay_of tag: old run directory:call id."""
        return f"{self.dir}:{rec.id}"

    def raw_response(self, rec: Record) -> tuple[str, bool]:
        """The old record's raw response text and whether it is an event stream."""
        name = rec.response.events_file
        if not name:
            return "", False
        path = self.dir / "calls" / rec.id / name
        return path.read_text(encoding="utf-8"), name.endswith(".sse")


def replay_key(url: str, headers: Mapping[str, str], sha256: str) -> str:
    """Replay match key: full URL (including host, path, query string) + organization/project/workspace headers + request hash. Comparing only the request hash would
    treat requests sent through a different gateway or organization as identical."""
    lower = {k.lower(): v for k, v in headers.items()}
    picked = "|".join(f"{h}={lower.get(h, '')}" for h in SCOPE_HEADERS)
    return f"{url}|{picked}|{sha256}"
