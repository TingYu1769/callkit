"""Four subcommands related to the storage layout: ``cat``, ``unpack``, ``migrate``, ``backup``."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from callkit.errors import CallkitError

# Field aliases for cat -> which file name to take from meta
ALIASES: dict[str, Callable[[Any], str | None]] = {
    "request": lambda rec: rec.request.body_file,
    "response": lambda rec: rec.response.events_file,
    "message": lambda rec: rec.response.message_file,
}


def add_commands(sub: Any) -> dict[str, Callable[[argparse.Namespace], int]]:
    p = sub.add_parser("cat", help="write one stored field of a call to stdout, byte for byte")
    p.add_argument("run", help="run directory")
    p.add_argument("call_id", help="call id, e.g. c_000007")
    p.add_argument(
        "field",
        nargs="?",
        default="request",
        help="request (default), response, message, meta, or a stored file name such as attempts/1.sse",
    )
    p = sub.add_parser(
        "unpack", help="write calls out in the file layout (run.json, index.jsonl, calls/<id>/)"
    )
    p.add_argument("run", help="run directory")
    p.add_argument("call_ids", nargs="*", help="calls to write; defaults to all")
    p.add_argument("-o", "--output", required=True, help="new directory to write into")
    p = sub.add_parser(
        "migrate", help="move a file-layout run dir into a new SQLite run dir (offline)"
    )
    p.add_argument("old", help="file-layout run directory; nothing may write to it meanwhile")
    p.add_argument("new", help="new run directory; rerun the same command to resume")
    p = sub.add_parser("backup", help="copy a SQLite run dir to a consistent snapshot")
    p.add_argument("run", help="SQLite run directory")
    p.add_argument("dest", help="new directory for the copy")
    return {"cat": _cat, "unpack": _unpack, "migrate": _migrate, "backup": _backup}


def _store(run: str) -> Any:
    from callkit.store import RunStore, is_run_dir

    if not is_run_dir(run):
        sys.exit(f"{run} is not a callkit run dir (no run.db or run.json)")
    return RunStore(run, create=False)


def _cat(args: argparse.Namespace) -> int:
    store = _store(args.run)
    try:
        name = args.field
        if name in ALIASES:
            name = ALIASES[name](store.load_meta(args.call_id))
        elif name == "meta":
            name = "meta.json"
        if not name:
            sys.exit(f"{args.call_id} has no {args.field}")
        data = store.read_file(args.call_id, name)
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f"cannot read {args.field} of {args.call_id}: {type(e).__name__}: {e}")
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return 0


def _unpack(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already exists and is not empty")
    n = unpack(_store(args.run), args.call_ids or None, out)
    print(f"wrote {n} call{'' if n == 1 else 's'} to {out}", file=sys.stderr)
    return 0


def unpack(store: Any, call_ids: list[str] | None, out: Path) -> int:
    """Write the records into ``out`` in the old file layout: run.json, index.jsonl, prices.json, items.jsonl, calls/<id>/.
    The resulting directory can be read directly by a RunStore using the old layout."""
    from callkit.pricing import dump_prices
    from callkit.store.common import append_line, index_row, write_json
    from callkit.store.files import ITEMS_FILE, PRICES_FILE, FileStore

    meta = {k: v for k, v in store.run_meta().items() if k != "format"}
    target = FileStore(out, run_meta=meta)
    ids = call_ids if call_ids is not None else [r.id for r in store.index()]
    for cid in ids:
        for name in store.file_names(cid):
            path = target.calls / cid / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(store.read_file(cid, name))
        append_line(out / "index.jsonl", index_row(store.load_meta(cid)).to_dict())
    prices = store.prices() if call_ids is None else {}
    if prices:
        write_json(out / PRICES_FILE, dump_prices(prices))
    for line in store.items() if call_ids is None else []:
        append_line(out / ITEMS_FILE, line)
    return len(ids)


def _migrate(args: argparse.Namespace) -> int:
    from callkit.migrate import migrate

    def show(done: int, total: int) -> None:
        print(f"\rmigrated {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        report = migrate(args.old, args.new, progress=show)
    except CallkitError as e:
        sys.exit(str(e))
    print(file=sys.stderr)
    print(
        f"migrated {report.migrated} calls (resumed after {report.resumed_from}), "
        f"{report.items} item rows, {report.prices} prices; skipped {len(report.skipped)} unreadable: "
        f"{', '.join(report.skipped[:10]) or '-'}",
        file=sys.stderr,
    )
    return 0


def _backup(args: argparse.Namespace) -> int:
    from callkit.store import DB_FILE

    src, dest = Path(args.run), Path(args.dest)
    if not (src / DB_FILE).exists():
        sys.exit(f"{src} has no {DB_FILE}; backup only handles SQLite run dirs")
    if dest.exists():
        sys.exit(f"{dest} already exists")
    dest.mkdir(parents=True)
    source = sqlite3.connect(f"{(src / DB_FILE).resolve().as_uri()}?mode=ro", uri=True)
    target = sqlite3.connect(dest / DB_FILE)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    print(f"backed up {src} to {dest}", file=sys.stderr)
    return 0
