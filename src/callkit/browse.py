"""Inspect the run directory: pure functions behind the command line ``callkit ls / show / tree / export``. Read-only, offline.

The three API formats have different request and answer shapes; here we extract only the parts people need: last user message, tool names, answer text, thinking length, tool calls.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

from callkit.record import Record
from callkit.store import RunStore
from callkit.summary import CLIP, Answer, Row, answer, clip, row_of
from callkit.summary import body_of as _body
from callkit.summary import dicts_of as _dicts

# Row, Answer, row_of, answer and clip used to be defined here; they now live in callkit.summary and can still be imported from here
__all__ = [
    "CLIP",
    "Answer",
    "Node",
    "Row",
    "answer",
    "clip",
    "export",
    "last_user",
    "records",
    "row_of",
    "tool_names",
    "totals",
    "tree",
]


def records(store: RunStore) -> Iterator[Record]:
    """Yield records one by one; warn and skip a record whose file is missing or corrupted, without interrupting later records."""
    for row in store.index():
        try:
            yield store.load(row.id)
        except (OSError, ValueError, KeyError, TypeError) as e:
            warnings.warn(
                f"skipped unreadable record {row.id}: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


def totals(rows: list[Row]) -> dict[str, Any]:
    """Summary: call count, success/failure, tokens, cost (separate counts for unpriced and unknown usage), and the same metrics grouped by model."""
    by_model: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_model[r.model].append(r)
    return {
        **_sums(rows),
        "ok": sum(r.ok for r in rows),
        "cache_read_tokens": sum(r.cache_read_tokens for r in rows),
        "by_model": {m: _sums(rs) for m, rs in sorted(by_model.items())},
    }


def _sums(rows: list[Row]) -> dict[str, Any]:
    return {
        "calls": len(rows),
        "input_tokens": sum(r.input_tokens for r in rows),
        "output_tokens": sum(r.output_tokens for r in rows),
        "cost_usd": sum(r.cost_usd or 0.0 for r in rows),
        "unpriced": sum(r.cost_usd is None and not r.usage_missing for r in rows),
        "usage_unknown": sum(r.usage_missing for r in rows),
    }


def last_user(rec: Record) -> str:
    """The last user input in the request: copied text (clipped), tool results are only counted."""
    body = _body(rec)
    convo = body.get("input" if rec.api == "openai.responses" else "messages")
    # The Responses input can be a string; a string value for messages is a bad request, but still display it
    if isinstance(convo, str):
        return clip(convo)
    if rec.api == "openai.responses":
        return _last_user_responses(_dicts(convo))
    return _last_user_messages(_dicts(convo))


def tool_names(rec: Record) -> list[str]:
    names = []
    for t in _dicts(_body(rec).get("tools")):
        inner = t.get("function")
        fn: dict[str, Any] = inner if isinstance(inner, dict) else t
        names.append(str(fn.get("name") or t.get("type") or "?"))
    return names


@dataclass
class Node:
    """A node in ``callkit tree``: continuation calls are attached under parent, forks under fork_of."""

    row: Row
    how: str = ""  # "" root; "→" continuation; "⑂" fork
    children: list[Node] = field(default_factory=list)


def tree(rows: list[Row]) -> dict[str | None, list[Node]]:
    """Forest grouped by session. Calls whose parent or fork_of is not in this run directory
    become roots; calls pointing to themselves or forming cycles also become roots, so the whole tree does not disappear."""
    ups = {r.id: ((r.parent, "→") if r.parent else (r.fork_of, "⑂")) for r in rows}
    nodes = {r.id: Node(r) for r in rows}
    roots: dict[str | None, list[Node]] = defaultdict(list)
    for r in rows:
        up, how = ups[r.id]
        if up in nodes and not _loops(r.id, ups):
            nodes[r.id].how = how
            nodes[up].children.append(nodes[r.id])
        else:
            roots[r.session].append(nodes[r.id])
    return dict(roots)


def _loops(start: str, ups: dict[str, tuple[str | None, str]]) -> bool:
    """Whether following parent / fork_of upward leads back to the starting call."""
    seen, cur = {start}, ups[start][0]
    while cur in ups:
        if cur in seen:
            return True
        seen.add(cur)
        cur = ups[cur][0]
    return False


def export(store: RunStore, out: TextIO, *, ok_only: bool = False) -> int:
    """One JSON line per call: metadata + full request body + assembled answer. Returns how many lines were written."""
    n = 0
    for rec in records(store):
        if ok_only and not rec.outcome.ok:
            continue
        line = {
            "id": rec.id,
            "api": rec.api,
            "source": rec.source,
            "model": _body(rec).get("model"),
            "item_id": rec.item_id,
            "repeat": rec.repeat,
            "parent": rec.parent,
            "fork_of": rec.fork_of,
            "tags": rec.tags,
            "outcome": asdict(rec.outcome),
            "usage": None if rec.usage is None else {**asdict(rec.usage), "raw": None},
            "cost_usd": rec.cost_usd,
            "timing": asdict(rec.timing),
            "response": rec.message,
            **_request(rec),
        }
        out.write(json.dumps(line, ensure_ascii=False) + "\n")
        n += 1
    return n


def _request(rec: Record) -> dict[str, Any]:
    """Request body for export: if it parses, return it as is (whether or not it is an object); if not,
    provide the raw text separately rather than replacing it with an empty object."""
    try:
        return {"request": json.loads(rec.body)}
    except ValueError:
        return {"request": None, "request_text": rec.body.decode("utf-8", "replace")}


def _last_user_messages(messages: list[dict[str, Any]]) -> str:
    """Anthropic and Chat: search backwards for user messages (or tool messages in Chat)."""
    tools = 0
    for m in reversed(messages):
        role = m.get("role")
        if role == "tool":
            tools += 1
            continue
        if role != "user":
            break
        content = m.get("content")
        if isinstance(content, str):
            return clip(content)
        parts = _dicts(content)
        texts = [str(c.get("text") or "") for c in parts if c.get("type") == "text"]
        tools += sum(c.get("type") == "tool_result" for c in parts)
        if texts:
            return clip(" ".join(texts))
        break
    return f"[tool results x{tools}]" if tools else ""


def _last_user_responses(items: list[dict[str, Any]]) -> str:
    tools = 0
    for it in reversed(items):
        if str(it.get("type", "")).endswith("_output"):
            tools += 1
            continue
        if it.get("role") != "user":
            break
        content = it.get("content")
        if isinstance(content, str):
            return clip(content)
        return clip(" ".join(str(c.get("text") or "") for c in _dicts(content)))
    return f"[tool results x{tools}]" if tools else ""
