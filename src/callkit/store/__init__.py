"""On-disk state for one run. Two layouts: a single SQLite file ``run.db`` (the default for new directories since 0.2.2), and the older file layout (``run.json``,
``index.jsonl``, ``calls/<id>/``; existing directories keep being read and written). ``RunStore(path)`` picks the backend by what the directory contains.
"""

from callkit.store.base import DB_FILE, LEGACY_RUN_FILE, RunStore, is_run_dir
from callkit.store.common import (
    LOCK_FILE,
    REDACTED,
    SCOPE_HEADERS,
    SECRET_HEADERS,
    SECRET_WORDS,
    append_line,
    dir_lock,
    is_secret_header,
    now_iso,
    redact,
    replay_key,
)
from callkit.store.files import GZIP_ABOVE, PRICES_FILE, FileStore
from callkit.store.sqlite import SqliteStore

__all__ = [
    "DB_FILE",
    "GZIP_ABOVE",
    "LEGACY_RUN_FILE",
    "LOCK_FILE",
    "PRICES_FILE",
    "REDACTED",
    "SCOPE_HEADERS",
    "SECRET_HEADERS",
    "SECRET_WORDS",
    "FileStore",
    "RunStore",
    "SqliteStore",
    "append_line",
    "dir_lock",
    "is_run_dir",
    "is_secret_header",
    "now_iso",
    "redact",
    "replay_key",
]
