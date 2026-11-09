"""Summary of one record: what each row of ``callkit ls`` and ``tree`` displays.

The SQLite layout computes the summary at write time and stores it in the ``calls`` table, so viewing no longer reads the large fields; the older layout reads records one by one and computes the summary when viewing. Both
share the functions here, which is why this is a separate module (both store and browse depend on it, and this avoids circular imports). The three APIs have different answer shapes; here we extract only what a person wants to see:
the answer text, the thinking length, and the tool calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from callkit.pricing import USAGE_MISSING
from callkit.record import Record

CLIP = 80  # maximum characters shown for one text snippet in summaries
SAID = 50  # characters of answer summary on each tree line


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
    # answer summary used by tree: the tool calls if there are any, otherwise the start of the answer
    said: str = ""


@dataclass
class Answer:
    """Human-readable parts of one answer."""

    text: str = ""
    thinking_chars: int = 0
    tool_calls: list[str] = field(default_factory=list)  # name(arguments)


def row_of(rec: Record) -> Row:
    u = rec.usage
    a = answer(rec)
    return Row(
        id=rec.id,
        started_at=rec.timing.started_at,
        api=rec.api,
        source=rec.source,
        model=str(body_of(rec).get("model", "")),
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
        said="; ".join(a.tool_calls) if a.tool_calls else clip(a.text, SAID),
    )


def answer(rec: Record) -> Answer:
    m = rec.message if isinstance(rec.message, dict) else {}
    if rec.api == "openai.responses":
        return _answer_responses(dicts_of(m.get("output")))
    if rec.api == "openai.chat_completions":
        choices = dicts_of(m.get("choices")) or [{}]
        msg = choices[0].get("message")
        return _answer_chat(msg if isinstance(msg, dict) else {})
    return _answer_anthropic(dicts_of(m.get("content")))


def clip(text: str, n: int = CLIP) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def dicts_of(value: Any) -> list[dict[str, Any]]:
    """Keep only dict elements: odd shapes in malformed requests or answers should not crash the summary."""
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def body_of(rec: Record) -> dict[str, Any]:
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
    fns = [tc.get("function") for tc in dicts_of(msg.get("tool_calls"))]
    calls = [_call(f.get("name"), f.get("arguments")) for f in fns if isinstance(f, dict)]
    return Answer(text, len(thinking) if isinstance(thinking, str) else 0, calls)
