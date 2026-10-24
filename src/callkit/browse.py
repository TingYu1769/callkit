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

from callkit.pricing import USAGE_MISSING
from callkit.record import Record
from callkit.store import RunStore

CLIP = 80  # maximum characters shown for one text snippet in summaries


@dataclass
class Row:
    """One row of ``callkit ls``."""

    id: str
    started_at: str
    api: str
    source: str
    model: str
    ok: bool
    stop: str | None
    reason: str
    input_tokens: int  # uncached + cache read + cache write; total input tokens for this request
    cache_read_tokens: int
    output_tokens: int
    cost_usd: float | None
    parent: str | None
    fork_of: str | None
    item: str | None  # item_id#repeat
    session: str | None
    usage_missing: bool = False  # succeeded but no usage; cost unknown


@dataclass
class Answer:
    """Human-readable parts of one answer."""

    text: str = ""
    thinking_chars: int = 0
    tool_calls: list[str] = field(default_factory=list)  # name(arguments)


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


def row_of(rec: Record) -> Row:
    u = rec.usage
    return Row(
        id=rec.id,
        started_at=rec.timing.started_at,
        api=rec.api,
        source=rec.source,
        model=str(_body(rec).get("model", "")),
        ok=rec.outcome.ok,
        stop=rec.outcome.stop_reason,
        reason=rec.attempts[-1].reason if rec.attempts else "",
        input_tokens=0
        if u is None
        else u.input_tokens + u.cache_read_tokens + u.cache_write_tokens,
        cache_read_tokens=0 if u is None else u.cache_read_tokens,
        output_tokens=0 if u is None else u.output_tokens,
        cost_usd=rec.cost_usd,
        parent=rec.parent,
        fork_of=rec.fork_of,
        item=None if rec.item_id is None else f"{rec.item_id}#{rec.repeat}",
        session=rec.tags.get("session"),
        usage_missing=USAGE_MISSING in rec.outcome.flags,
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


def answer(rec: Record) -> Answer:
    m = rec.message if isinstance(rec.message, dict) else {}
    if rec.api == "openai.responses":
        return _answer_responses(_dicts(m.get("output")))
    if rec.api == "openai.chat_completions":
        choices = _dicts(m.get("choices")) or [{}]
        msg = choices[0].get("message")
        return _answer_chat(msg if isinstance(msg, dict) else {})
    return _answer_anthropic(_dicts(m.get("content")))


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


def clip(text: str, n: int = CLIP) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _request(rec: Record) -> dict[str, Any]:
    """Request body for export: if it parses, return it as is (whether or not it is an object); if not,
    provide the raw text separately rather than replacing it with an empty object."""
    try:
        return {"request": json.loads(rec.body)}
    except ValueError:
        return {"request": None, "request_text": rec.body.decode("utf-8", "replace")}


def _dicts(value: Any) -> list[dict[str, Any]]:
    """Keep only dict elements: odd shapes in malformed requests or answers should not crash the summary."""
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _body(rec: Record) -> dict[str, Any]:
    try:
        value = json.loads(rec.body) if rec.body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _call(name: Any, args: Any) -> str:
    text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return f"{name}({clip(text, 60)})"


def _answer_anthropic(blocks: list[dict[str, Any]]) -> Answer:
    a = Answer()
    for b in blocks:
        kind = b.get("type")
        if kind == "text":
            a.text += b.get("text") or ""
        elif kind == "thinking":
            a.thinking_chars += len(b.get("thinking") or "")
        elif kind in ("tool_use", "server_tool_use"):
            a.tool_calls.append(_call(b.get("name"), b.get("input")))
    return a


def _answer_responses(items: list[dict[str, Any]]) -> Answer:
    a = Answer()
    for it in items:
        kind = it.get("type")
        if kind == "message":
            a.text += "".join(c.get("text") or "" for c in it.get("content") or [])
        elif kind == "reasoning":
            a.thinking_chars += sum(len(s.get("text") or "") for s in it.get("summary") or [])
        elif kind in ("function_call", "custom_tool_call"):
            a.tool_calls.append(_call(it.get("name"), it.get("arguments") or it.get("input")))
    return a


def _answer_chat(msg: dict[str, Any]) -> Answer:
    content = msg.get("content")
    text = content if isinstance(content, str) else ""
    thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
    fns = [tc.get("function") for tc in _dicts(msg.get("tool_calls"))]
    calls = [_call(f.get("name"), f.get("arguments")) for f in fns if isinstance(f, dict)]
    return Answer(text, len(thinking) if isinstance(thinking, str) else 0, calls)


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
