"""A call record: everything persisted for one API call, format name ``callkit.call/2``.

On disk a record is a directory ``calls/<id>/``: ``meta.json`` stores the dataclasses from this module, while the request body,
raw event stream, and assembled Message each have their own file; ``meta.json`` only records the filenames. Changing fields here
changes the format: bump the format version and keep old formats readable.
"""

from __future__ import annotations

import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Self

FORMAT = "callkit.call/2"
READABLE_FORMATS = ("callkit.call/1", FORMAT)  # /1 is upgraded to /2 in place when read


@dataclass
class Usage:
    """Normalized usage. The three input parts are disjoint; their sum is the total input
    (Anthropic's input_tokens only counts the part that did not use cache; OpenAI's
    input_tokens includes cache reads and writes, and adapters subtract them).

    Cache writes are bucketed by TTL: the two providers use different buckets (Anthropic
    5m/1h, OpenAI 30m/12h), so they are recorded with TTL labels rather than fixed field
    names.
    """

    input_tokens: int = 0  # input that did not use cache
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0  # input written to cache, all TTLs combined
    # TTL label to token count; empty when upstream does not break it down
    cache_write_by_ttl: dict[str, int] = field(default_factory=dict)
    output_tokens: int = 0  # includes thinking
    # thinking part of output_tokens; None when upstream does not provide it
    thinking_tokens: int | None = None
    # upstream usage as-is; may be amended when passing through a gateway
    raw: dict[str, Any] | None = None


@dataclass
class Outcome:
    """Result has two layers: how the transport ended, and whether the protocol completed."""

    ok: bool
    transport_end: str  # done / eof / timeout / connect_error
    protocol: str  # completed / failed / incomplete
    stop_reason: str | None = None  # model's own stop reason, recorded as-is
    error: str | None = None  # one-sentence description; HTTP error body, mid-stream error event
    # e.g. truncated_tool_input, thinking_may_be_invalid:3
    flags: list[str] = field(default_factory=list)


@dataclass
class Attempt:
    """A single attempt. Failed attempts are also recorded and billed by the usage they received."""

    n: int  # starts at 1
    status: int | None  # HTTP status code; None if connection failed
    transport_end: str
    protocol: str
    decision: str  # retry / stop
    reason: str  # e.g. overloaded, no_message_stop, budget_exhausted
    # usage known for this attempt; on interruption, the last cumulative value received
    usage: Usage | None = None
    cost_usd: float | None = None
    events_file: str | None = None  # raw event stream for failed attempts: attempts/<n>.sse


@dataclass
class Request:
    url: str  # includes query string, e.g. /v1/messages?beta=true
    # keys are masked; anthropic-version and anthropic-beta must be preserved
    headers: dict[str, str]
    body_file: str  # request.json or request.json.gz: request body as-is
    sha256: str  # covers request body and behavior-changing request headers, excluding keys


@dataclass
class Response:
    status: int | None
    headers: dict[str, str]  # gateway headers such as x-oneapi-request-id
    events_file: str | None  # response.sse: raw event stream of the last attempt
    message_file: str | None  # message.json: assembled Message
    message_id: str | None = None  # answer id, recorded as-is
    inference_geo: str | None = None


@dataclass
class Timing:
    started_at: str  # ISO 8601, UTC
    first_byte_ms: int | None
    total_ms: int


@dataclass
class Record:
    """A single call. ``body`` and ``message`` are only kept in memory; when persisted they are written to separate files."""

    id: str
    api: str  # anthropic.messages / openai.responses / openai.chat_completions
    source: str  # self (sent by Client) / proxy (recording proxy) / sdk (SDK mount)
    run: str
    request: Request
    response: Response
    outcome: Outcome
    timing: Timing
    usage: Usage | None = None  # usage of the last attempt
    cost_usd: float | None = None  # total for all attempts; None when prices are missing
    attempts: list[Attempt] = field(default_factory=list)
    item_id: str | None = None  # batch item id; resume key is (item_id, repeat)
    repeat: int | None = None
    tags: dict[str, str] = field(default_factory=dict)
    parent: str | None = None  # previous call in the same chain
    # when proxy inference of parent is ambiguous
    parent_candidates: list[str] = field(default_factory=list)
    fork_of: str | None = None  # points to the modified call when forking
    # which fields changed relative to parent or fork_of
    edits: list[str] = field(default_factory=list)
    format: str = FORMAT
    body: bytes = field(default=b"", repr=False, metadata={"transient": True})
    message: dict[str, Any] | None = field(default=None, repr=False, metadata={"transient": True})

    def to_dict(self) -> dict[str, Any]:
        """Content written into meta.json (excluding body and message)."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Read /2; when reading /1, usage is upgraded to the /2 representation, and the returned record has ``format`` /2."""
        if data.get("format") not in READABLE_FORMATS:
            raise ValueError(
                f"unknown record format {data.get('format')!r}; expected one of {READABLE_FORMATS}"
            )
        if data["format"] != FORMAT:
            data = _upgrade_v1(data)
        return _from_dict(cls, data)


@dataclass
class IndexRow:
    """One row in index.jsonl. Resume and cost accumulation read only the index."""

    id: str
    ok: bool
    sha256: str
    created_at: str
    item_id: str | None = None
    repeat: int | None = None
    parent: str | None = None
    fork_of: str | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return _from_dict(cls, data)


def _upgrade_v1(data: dict[str, Any]) -> dict[str, Any]:
    """/1 to /2: combine cache_write_5m_tokens and cache_write_1h_tokens into cache_write_tokens and per-TTL buckets."""
    out = {**data, "format": FORMAT}
    out["usage"] = _upgrade_usage(data.get("usage"))
    out["attempts"] = [
        {**a, "usage": _upgrade_usage(a.get("usage"))} for a in data.get("attempts") or []
    ]
    return out


def _upgrade_usage(u: dict[str, Any] | None) -> dict[str, Any] | None:
    if u is None:
        return None
    w5, w1 = u.get("cache_write_5m_tokens") or 0, u.get("cache_write_1h_tokens") or 0
    rest = {
        k: v for k, v in u.items() if k not in ("cache_write_5m_tokens", "cache_write_1h_tokens")
    }
    by_ttl = {ttl: n for ttl, n in (("5m", w5), ("1h", w1)) if n}
    return {**rest, "cache_write_tokens": w5 + w1, "cache_write_by_ttl": by_ttl}


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_dict(getattr(obj, f.name))
            for f in fields(obj)
            if not f.metadata.get("transient")
        }
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.metadata.get("transient"):
            continue
        if f.name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ValueError(f"{cls.__name__} is missing field {f.name}")
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name])
    return cls(**kwargs)


def _convert(tp: Any, value: Any) -> Any:
    """Restore JSON values into dataclasses according to type annotations; only handles the types used in this module."""
    if value is None:
        return None
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        return _convert(inner[0], value) if len(inner) == 1 else value
    if origin is list:
        (item,) = typing.get_args(tp)
        return [_convert(item, v) for v in value]
    if isinstance(tp, type) and is_dataclass(tp):
        return _from_dict(tp, value)
    return value
