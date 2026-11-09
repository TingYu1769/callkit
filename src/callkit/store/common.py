"""Small pieces shared by both storage layouts: header redaction, cross-process lock, JSONL append, timestamps, index rows, replay keys, and raw event-stream text."""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from callkit.record import IndexRow, Record
from callkit.transport import RawAttempt

LOCK_FILE = ".callkit.lock"  # cross-process lock for the run directory
SECRET_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie"}
# Split header names on - and _; if any part matches one of these words, it is a secret header: x-api-key, api-key (Azure), x-goog-api-key, x-auth-token, etc.
SECRET_WORDS = {"key", "apikey", "token", "secret", "password", "auth", "credential", "credentials"}
REDACTED = "<redacted>"
# Request headers that select the organization, project, or workspace: the same request body sent to different organizations must not count as identical
SCOPE_HEADERS = ("openai-organization", "openai-project", "anthropic-workspace-id")


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


def index_row(rec: Record) -> IndexRow:
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


def replay_key(url: str, headers: Mapping[str, str], sha256: str) -> str:
    """Matching key for replay: the full URL (host, path, and query string) + the headers that select the organization, project, or workspace + the request hash. Comparing only the request hash
    would treat requests sent through a different gateway or organization as identical."""
    lower = {k.lower(): v for k, v in headers.items()}
    picked = "|".join(f"{h}={lower.get(h, '')}" for h in SCOPE_HEADERS)
    return f"{url}|{picked}|{sha256}"


def attempt_text(raw: RawAttempt, stem: str) -> tuple[str, str] | None:
    """The raw text to store for one attempt: if there is an event stream, store <stem>.sse (verbatim lines, joined back with \\n and a trailing newline added); if there is no event stream (non-200
    error body, full JSON of a non-streaming response), store <stem>.body; if neither exists, return None."""
    if raw.lines:
        return f"{stem}.sse", "\n".join(line for _, line in raw.lines) + "\n"
    if raw.error_body is not None:
        return f"{stem}.body", raw.error_body
    return None


def write_json(path: Path, obj: Any) -> None:
    # Temporary file name includes PID and thread id so multiple processes initializing the same directory cannot overwrite each other.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
