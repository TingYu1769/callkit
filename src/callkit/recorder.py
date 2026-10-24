"""Records one HTTP round trip as one record. Shared by the recording proxy and SDK hooks.

Only POSTs for three APIs are recorded, recognized by path suffix (``adapters.api_for_path``); streaming
responses are assembled and judged from the event stream, and non-streaming responses (whole JSON)
go to the adapter's ``from_json``. The session key is taken from explicit ``session``, then session
headers, then request body ``prompt_cache_key``; parent is only inferred within a session, and when
there is no session key parent is left empty and matched candidates are stored in
``parent_candidates``.

Within one process and one run directory, ``Recorder.shared`` shares a single recorder (including
RunStore, lock, and session history), so call numbers do not collide.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from callkit import sse
from callkit.adapters import api_for_path, get_adapter
from callkit.adapters.base import Adapter
from callkit.chain import infer_parent
from callkit.client import request_sha256
from callkit.config import PriceTable
from callkit.context import CURRENT_ITEM, ItemContext
from callkit.pricing import cost_usd, mark_usage_missing
from callkit.record import Attempt, Outcome, Record, Request, Response, Timing, Usage
from callkit.store import RunStore, now_iso
from callkit.transport import RawAttempt

# Claude Code and Codex session headers; when neither is present use prompt_cache_key
SESSION_HEADERS = ("x-claude-code-session-id", "session-id")
HISTORY_PER_SESSION = 200  # how many recent records per session to keep for parent inference
REFUSAL_HEADERS = {
    "content-type": "application/json",
    "x-should-retry": "false",
}  # headers for the 402 response returned when a request is blocked
# retry sequence from the official SDK, stored as the retry_count tag
RETRY_HEADER = "x-stainless-retry-count"

_SHARED: dict[str, Recorder] = {}
_SHARED_LOCK = threading.Lock()


@dataclass
class Exchange:
    """What is received in one HTTP round trip. chunks is (seconds relative to start, decoded text piece)."""

    method: str
    url: str  # full URL, including query string
    headers: dict[str, str]  # request headers sent by the caller, as-is (masked when written)
    body: bytes
    t0: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=now_iso)
    status: int | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    chunks: list[tuple[float, str]] = field(default_factory=list)
    end: str = "done"  # done / eof / timeout / connect_error / client_gone
    error: str | None = None
    # item running in the scheduler when the request was sent: the stream may finish in another item,
    # so reading it when writing would record the wrong item
    item: ItemContext | None = field(default_factory=CURRENT_ITEM.get)


def recorded_api(method: str, url: str) -> str | None:
    """Return the API name if this request is recorded, otherwise None. Only POST is recorded, recognized by path suffix."""
    return api_for_path(urlsplit(url).path) if method.upper() == "POST" else None


class Recorder:
    """Thread-safe: session history is managed by one lock, and shared store state is managed by RunStore's own lock."""

    def __init__(self, run: str | Path, *, run_meta: dict[str, Any] | None = None) -> None:
        # Use an absolute path: if the caller changes directory later, writes still go to the original directory.
        self.store = RunStore(Path(run).resolve(), run_meta=run_meta)
        self._sessions: dict[str | None, deque[Record]] = defaultdict(
            lambda: deque(maxlen=HISTORY_PER_SESSION)
        )
        self._lock = threading.Lock()
        # price table already in the run directory
        self.prices: PriceTable = self.store.merge_prices({})

    def add_prices(self, prices: PriceTable | None) -> None:
        """Merge prices from one entry point into the run directory's price table; conflicting prices for the same model raise CallkitError."""
        if prices:
            # Merge and update the in-memory table together, so a concurrent merge cannot overwrite the newer table.
            with self._lock:
                self.prices = self.store.merge_prices(prices)

    def refusal(self, body: bytes, cap_usd: float | None) -> bytes | None:
        """When a cap is set, decide whether to block this request: if blocked, return a 402 response body (fake 402, no retry).
        Block when the cap is reached, or when the model has no price (cannot be accounted)."""
        if cap_usd is None:
            return None
        model = str((_json(body) or {}).get("model", ""))
        if model not in self.prices:
            code, message = (
                "callkit_missing_price",
                f"callkit: a spending cap of ${cap_usd} is set but the price table has no entry for model {model!r}",
            )
        else:
            spent = self.store.spent_usd()
            if spent < cap_usd:
                return None
            code, message = (
                "callkit_cap_reached",
                f"callkit: spent ${spent:.4f}, reached the cap of ${cap_usd}",
            )
        error = {"type": code, "code": code, "message": message}
        return json.dumps({"type": "error", "error": error}, ensure_ascii=False).encode()

    @classmethod
    def shared(cls, run: str | Path, *, run_meta: dict[str, Any] | None = None) -> Recorder:
        """Within one process, share one recorder per absolute run directory; ``run_meta`` is written to run.json only when the directory is first created."""
        key = str(Path(run).resolve())
        with _SHARED_LOCK:
            found = _SHARED.get(key)
            if found is None:
                found = _SHARED[key] = cls(run, run_meta=run_meta)
            return found

    def record(
        self,
        ex: Exchange,
        *,
        source: str,
        session: str | None = None,
        tags: Mapping[str, str] | None = None,
        replayed: bool = False,
    ) -> Record | None:
        """Record one request and write it to the store; return None if this request is not recorded (see ``recorded_api``).
        ``replayed`` means the response came from the replay cache: no money was spent, so cost is recorded as 0."""
        api = recorded_api(ex.method, ex.url)
        if api is None:
            return None
        adapter = get_adapter(api)
        parsed = _json(ex.body)
        headers = {k.lower(): v for k, v in ex.headers.items()}
        session = session or _session_key(headers, parsed)
        content_type = {k.lower(): v for k, v in ex.response_headers.items()}.get(
            "content-type", ""
        )
        is_sse = content_type.startswith("text/event-stream")
        text = "".join(piece for _, piece in ex.chunks)
        raw = RawAttempt(
            1,
            ex.status,
            ex.response_headers,
            _lines(ex.chunks) if is_sse else [],
            None if is_sse else (text or ex.error),
            ex.end,
            ex.started_at,
            ex.chunks[0][0] if ex.chunks else None,
            time.monotonic() - ex.t0,
        )
        message, outcome, usage, reason = _judge(adapter, raw, is_sse, text, parsed)
        price = self.prices.get(str((parsed or {}).get("model", "")))
        if usage is None and outcome.ok:
            # Success without usage: cost is unknown; do not record 0. Mark usage missing whether priced or replayed.
            mark_usage_missing(outcome)
            cost = None
        else:
            cost = None if price is None else (0.0 if usage is None else cost_usd(usage, price))
        if replayed:
            cost = 0.0
        # Do not lose the transport-level reason.
        if ex.error and ex.error not in (outcome.error or ""):
            outcome.error = f"{outcome.error} ({ex.error})" if outcome.error else ex.error
        with self._lock:
            prior = list(self._sessions[session])
        try:
            parent, candidates = (
                infer_parent(parsed, prior, api=api) if parsed is not None else (None, [])
            )
        except (AttributeError, TypeError, ValueError, KeyError):
            # request body has unexpected shape (server likely returned 400): record it without parent inference
            parent, candidates = None, []
        if session is None and parent is not None:
            # without a session key, do not accept parent; only record it as a candidate
            parent, candidates = None, [parent]
        rec = Record(
            id=self.store.new_call_id(),
            api=api,
            source=source,
            run=str(self.store.dir),
            request=Request(ex.url, dict(ex.headers), "", request_sha256(ex.body, ex.headers)),
            response=Response(
                ex.status,
                ex.response_headers,
                None,
                None,
                (message or {}).get("id"),
                ((message or {}).get("usage") or {}).get("inference_geo"),
            ),
            outcome=outcome,
            timing=Timing(
                ex.started_at,
                None if raw.first_byte_s is None else int(raw.first_byte_s * 1000),
                int(raw.total_s * 1000),
            ),
            usage=usage,
            cost_usd=cost,
            attempts=[Attempt(1, ex.status, ex.end, outcome.protocol, "stop", reason, usage, cost)],
            parent=parent,
            parent_candidates=candidates,
            tags={**_tags(headers, parsed, session), **(tags or {})},
            body=ex.body,
            message=message,
        )
        item = ex.item
        if item is not None:  # item running in the scheduler when the request was sent
            rec.item_id, rec.repeat = item.item_id, item.repeat
        self.store.write(rec, [raw])
        if item is not None:
            item.call_ids.append(rec.id)
        if message is not None:
            with self._lock:
                self._sessions[session].append(rec)
        return rec

    def record_safely(self, ex: Exchange, **kwargs: Any) -> Record | None:
        """Like ``record``, but recording errors only emit a warning and are not raised: recording is a side path, so it must not make the caller's request fail (for example, a malformed body that gets a 400 should be seen as the original 400, not as an exception from callkit parent inference)."""
        try:
            return self.record(ex, **kwargs)
        # side path errors do not affect the main path; the reason goes into the warning
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"callkit failed to record {ex.method} {ex.url}: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None


def _judge(
    adapter: Adapter, raw: RawAttempt, is_sse: bool, text: str, request: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, Outcome, Usage | None, str]:
    """Return (assembled answer, outcome, usage, reason). Non-streaming responses (whole JSON) go to the adapter's from_json."""
    if is_sse:
        events = sse.parse(raw.lines)
        message = adapter.assemble(events)
        j = adapter.judge(raw.status, raw.transport_end, events, message, raw.error_body, request)
        return message, j.outcome, adapter.usage(events), j.reason
    message, j, usage = adapter.from_json(
        raw.status, raw.transport_end, text or raw.error_body or "", request
    )
    return message, j.outcome, usage, j.reason


def _session_key(headers: Mapping[str, str], parsed: dict[str, Any] | None) -> str | None:
    for name in SESSION_HEADERS:
        value = headers.get(name)
        if value:
            return value
    key = (parsed or {}).get("prompt_cache_key")
    return str(key) if key else None


def _tags(
    headers: Mapping[str, str], parsed: dict[str, Any] | None, session: str | None
) -> dict[str, str]:
    """Session id; for Codex also record thread_id and the request kind (turn / prewarm / compaction / memory);
    for official SDK requests also record the retry sequence retry_count."""
    tags = {"session": session} if session else {}
    thread = headers.get("thread-id")
    if thread:
        tags["thread_id"] = thread
    kind = _request_kind(parsed)
    if kind:
        tags["request_kind"] = kind
    retry = headers.get(RETRY_HEADER)
    if retry:
        tags["retry_count"] = retry
    return tags


def _request_kind(parsed: dict[str, Any] | None) -> str | None:
    meta = (parsed or {}).get("client_metadata")
    raw = meta.get("x-codex-turn-metadata") if isinstance(meta, dict) else None
    turn = _json(raw.encode()) if isinstance(raw, str) else None
    kind = (turn or {}).get("request_kind")
    return str(kind) if kind else None


def _lines(chunks: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Split text pieces into lines by \\n, recording for each line the time its newline arrived. A final partial line without a newline is kept too.
    The storage layer joins them with \\n and appends one final newline, so streams ending with a newline round-trip byte for byte."""
    lines: list[tuple[float, str]] = []
    buf, t = "", 0.0
    for t, piece in chunks:
        buf += piece
        *complete, buf = buf.split("\n")
        lines.extend((round(t, 3), line) for line in complete)
    if buf:
        lines.append((round(t, 3), buf))
    return lines


def _json(data: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None
