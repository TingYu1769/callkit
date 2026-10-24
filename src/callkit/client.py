"""Public entry points: single call and batch call. Synchronous; concurrency via a thread pool."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

from callkit import sse, transport, version
from callkit.adapters import get_adapter
from callkit.adapters.base import Adapter, HttpRequest, Judgement
from callkit.chain import stale_thinking
from callkit.config import Endpoint, Price, PriceTable
from callkit.context import CURRENT_ITEM
from callkit.errors import CallkitError, CapReached, ItemChanged, MissingPrice, StreamRequired
from callkit.pricing import cost_usd, mark_usage_missing
from callkit.record import Attempt, Record, Request, Response, Timing
from callkit.replay import ReplayCache
from callkit.store import REDACTED, RunStore, is_secret_header, now_iso, redact
from callkit.transport import RawAttempt

Body = bytes | Mapping[str, Any]  # raw bytes, or a request body to serialize
# request headers included in the request hash
BEHAVIOR_HEADERS = ("anthropic-version", "anthropic-beta")
# fields that hold the conversation in the two API shapes
CONVERSATION_FIELDS = ("messages", "input")
_MISSING = object()


class Client:
    def __init__(
        self,
        endpoint: Endpoint,
        run: str | Path,
        *,
        prices: PriceTable | None = None,
        cap_usd: float | None = None,
        max_attempts: int = 3,
        idle_timeout_s: float = 180.0,
        tags: Mapping[str, str] | None = None,
        replay_from: str | Path | None = None,
    ) -> None:
        """If ``cap_usd`` is set and the price for the model is missing, MissingPrice is raised before the first call.
        The key is read from the environment variable ``endpoint.key_env``; if it is not set, CallkitError is raised.
        When ``replay_from`` names an old run directory, replay caching is enabled: a request that exactly matches
        a successful call in the old run gets the old response, the new record's ``replay_of`` tag points to it, and cost is recorded as 0."""
        key = os.environ.get(endpoint.key_env)
        if not key:
            raise CallkitError(
                f"environment variable {endpoint.key_env} is not set (endpoint {endpoint.name})"
            )
        self.endpoint = endpoint
        self.prices: PriceTable = dict(prices or {})
        self.cap_usd = cap_usd
        self.max_attempts = max_attempts
        self.idle_timeout_s = idle_timeout_s
        self.tags = dict(tags or {})
        self.adapter: Adapter = get_adapter(endpoint.api)
        self.store = RunStore(run, run_meta=self._run_meta())
        # Records in the same directory are priced with the same price table; use the merged table.
        self.prices = self.store.merge_prices(self.prices)
        self._key = key
        self._http = transport.open_client(endpoint)
        self._replay = (
            ReplayCache.shared(replay_from, self.store.dir) if replay_from is not None else None
        )

    def call(
        self,
        body: Body,
        *,
        parent: Record | None = None,
        fork_of: Record | None = None,
        item_id: str | None = None,
        repeat: int | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Record:
        """Send one call (with retries on failure), persist it, and return the record. A failed call does not raise; check ``rec.outcome``.

        Thread-safe. If the request body does not specify stream, true is added; if it specifies false, StreamRequired is raised.
        CapReached is raised when the cap is exceeded. When ``parent`` or ``fork_of`` is given, edits are computed and written to the record;
        with ``fork_of`` its query string and headers are inherited (for example Claude Code's ``?beta=true`` and beta header), otherwise
        ``parent``'s are inherited; the path always uses the adapter's default path.
        """
        raw_body, parsed = _prepare_body(body, self.adapter)
        base = fork_of or parent
        req = self.adapter.http_request(
            raw_body, self.endpoint, self._key, **_inherit(base, self.adapter)
        )
        old = (
            self._replay.take(req.url, req.headers, request_sha256(raw_body, req.headers))
            if self._replay
            else None
        )
        if old is not None:  # replay cache hit: no network, no cost, cap is not checked
            assert self._replay is not None
            call_id = self.store.new_call_id()
            rec, raws = _replayed(
                self._replay,
                old,
                call_id,
                str(self.store.dir),
                req,
                raw_body,
                self.endpoint.auth_header,
            )
            rec.tags = {"replay_of": self._replay.ref(old)}
        else:
            price = self._price_for(str(parsed.get("model", "")))
            if self.cap_usd is not None and self.store.spent_usd() >= self.cap_usd:
                raise CapReached(
                    f"spent ${self.store.spent_usd():.4f}, reached the cap of ${self.cap_usd}"
                )
            call_id = self.store.new_call_id()
            t0 = time.monotonic()
            results = transport.send(
                self._http,
                self.adapter,
                req,
                max_attempts=self.max_attempts,
                idle_timeout_s=self.idle_timeout_s,
                request=parsed,
            )
            rec = self._record(call_id, req, raw_body, results, price, time.monotonic() - t0)
            raws = [raw for raw, _ in results]
        rec.parent = _ref(parent, self.store.dir) if parent else None
        rec.fork_of = _ref(fork_of, self.store.dir) if fork_of else None
        rec.edits = _edits(base, parsed, ignore_messages=fork_of is None)
        stale = _stale_thinking(parent, fork_of, parsed, self.adapter.api)
        if stale:
            what = "reasoning" if self.adapter.api == "openai.responses" else "thinking"
            rec.outcome.flags.append(f"{what}_may_be_invalid:{stale}")
        item = CURRENT_ITEM.get()  # the item currently running in the scheduler
        if item is not None and item_id is None:
            item_id, repeat = item.item_id, item.repeat
        rec.item_id, rec.repeat = item_id, repeat
        rec.tags = {**self.tags, **(tags or {}), **rec.tags}
        self.store.write(rec, raws)
        if item is not None and (item_id, repeat) == (item.item_id, item.repeat):
            item.call_ids.append(rec.id)  # also count explicit calls targeting the same item
        return rec

    def batch(
        self,
        items: Mapping[str, Body],
        *,
        concurrency: int = 8,
        repeats: int = 1,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[Record]:
        """``items`` maps item_id to request body. A thread pool with ``concurrency`` threads produces records in completion order;
        already successful (item_id, repeat) pairs are skipped, and if the request hash changes for the same item_id, ItemChanged is raised;
        when cumulative spending reaches the cap, no new requests are sent, in-flight requests finish, and CapReached is raised after yielding their records.

        When the iterator is closed, queued tasks are cancelled and in-flight tasks finish (results are already persisted) before exit. If you only
        break while the iterator is still referenced, cleanup waits until it is garbage collected; to clean up immediately after break, wrap it
        with ``contextlib.closing``. Other exceptions (such as MissingPrice) also cancel queued tasks before being raised.
        """
        jobs = self._plan(items, repeats)
        capped = threading.Event()

        def work(item_id: str, repeat: int, raw: bytes) -> Record | None:
            if capped.is_set():
                return None
            try:
                return self.call(raw, item_id=item_id, repeat=repeat, tags=tags)
            except CapReached:
                capped.set()
                return None

        pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="callkit")
        try:
            futures = [pool.submit(work, *job) for job in jobs]
            for future in as_completed(futures):
                rec = future.result()
                if rec is not None:
                    yield rec
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if capped.is_set():
            raise CapReached(
                f"spent ${self.store.spent_usd():.4f}, reached the cap of ${self.cap_usd}; remaining jobs were not sent"
            )

    def _plan(self, items: Mapping[str, Body], repeats: int) -> list[tuple[str, int, bytes]]:
        """Plan the (item_id, repeat, body) jobs to send. Successful pairs are skipped; if the request hash changes for the same item,
        raise ItemChanged before sending any request so changed input is not mixed with old results."""
        done = self.store.done()
        jobs: list[tuple[str, int, bytes]] = []
        for item_id, body in items.items():
            raw, _ = _prepare_body(body, self.adapter)
            sha = request_sha256(
                raw, self.adapter.http_request(raw, self.endpoint, self._key).headers
            )
            for repeat in range(repeats):
                prev = done.get((item_id, repeat))
                if prev is None:
                    jobs.append((item_id, repeat, raw))
                elif prev != sha:
                    raise ItemChanged(
                        f"the request for item {item_id!r} changed (repeat {repeat} already succeeded with the old request); use another run dir or item_id"
                    )
        return jobs

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _price_for(self, model: str) -> Price | None:
        price = self.prices.get(model)
        if price is None and self.cap_usd is not None:
            raise MissingPrice(
                f"a spending cap of ${self.cap_usd} is set but the price table has no entry for model {model!r}"
            )
        return price

    def _record(
        self,
        call_id: str,
        req: HttpRequest,
        raw_body: bytes,
        results: Sequence[tuple[RawAttempt, Judgement]],
        price: Price | None,
        elapsed_s: float,
    ) -> Record:
        attempts = [
            self._attempt(raw, j, price, i == len(results) - 1)
            for i, (raw, j) in enumerate(results)
        ]
        final_raw, final_j = results[-1]
        message = self.adapter.assemble(sse.parse(final_raw.lines))
        msg = message or {}
        response = Response(
            status=final_raw.status,
            headers=final_raw.headers,
            events_file=None,  # filled by RunStore when persisting
            message_file=None,
            message_id=msg.get("id"),
            inference_geo=(msg.get("usage") or {}).get("inference_geo"),
        )
        first_byte = final_raw.first_byte_s
        return Record(
            id=call_id,
            api=self.adapter.api,
            source="self",
            run=str(self.store.dir),
            request=Request(
                req.url,
                redact(req.headers, self.endpoint.auth_header),
                "",
                request_sha256(raw_body, req.headers),
            ),
            response=response,
            outcome=final_j.outcome,
            timing=Timing(
                started_at=results[0][0].started_at,
                first_byte_ms=None if first_byte is None else int(first_byte * 1000),
                total_ms=int(elapsed_s * 1000),
            ),
            usage=attempts[-1].usage,
            cost_usd=_total_cost(price, attempts, final_j),
            attempts=attempts,
            body=raw_body,
            message=message,
        )

    def _attempt(self, raw: RawAttempt, j: Judgement, price: Price | None, last: bool) -> Attempt:
        """Failed attempts are also charged using the usage received; attempts with no usage (for example HTTP 529) cost 0."""
        usage = self.adapter.usage(sse.parse(raw.lines))
        cost = None if price is None else (0.0 if usage is None else cost_usd(usage, price))
        decision = "stop" if last else "retry"
        return Attempt(
            raw.n,
            raw.status,
            raw.transport_end,
            j.outcome.protocol,
            decision,
            j.reason,
            usage,
            cost,
        )

    def _run_meta(self) -> dict[str, Any]:
        ep = self.endpoint
        return {
            "callkit_version": version.version(),
            "callkit_commit": version.commit(),
            "endpoint": {
                "name": ep.name,
                "api": ep.api,
                "base_url": ep.base_url,
                "anthropic_version": ep.anthropic_version,
                # the proxy URL may contain a secret, so only record whether one is set
                "proxy": ep.proxy is not None,
            },
            "cap_usd": self.cap_usd,
            "max_attempts": self.max_attempts,
            "idle_timeout_s": self.idle_timeout_s,
        }


def request_sha256(body: bytes, headers: Mapping[str, str]) -> str:
    """Hash over the request body and behavior-changing headers (version and beta headers), excluding the API key."""
    lower = {k.lower(): v for k, v in headers.items()}
    h = hashlib.sha256()
    for name in BEHAVIOR_HEADERS:
        h.update(f"{name}:{lower.get(name, '')}\n".encode())
    h.update(body)
    return h.hexdigest()


def _prepare_body(body: Body, adapter: Adapter) -> tuple[bytes, dict[str, Any]]:
    """Raw bytes are kept as-is; only when the stream field is missing are streaming defaults added for the API
    (for example Chat also adds include_usage) and the body re-serialized (key order unchanged, additions at the end)."""
    raw = body if isinstance(body, bytes) else None
    parsed: dict[str, Any] = json.loads(body) if isinstance(body, bytes) else dict(body)
    if parsed.get("stream") is False:
        raise StreamRequired("Client only sends streaming requests; do not set stream: false")
    if "stream" not in parsed:
        parsed = {**parsed, **adapter.stream_defaults(parsed)}
        raw = None
    if raw is None:
        raw = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode()
    return raw, parsed


def _total_cost(price: Price | None, attempts: list[Attempt], final: Judgement) -> float | None:
    """Total cost across attempts. When the call succeeds but has no usage, that attempt's cost is unknown (not recorded as 0) and marked usage_missing;
    known costs from previous failed attempts are still counted, so the record cost is a lower bound, or None if no attempt cost is known."""
    if final.outcome.ok and attempts[-1].usage is None:
        mark_usage_missing(final.outcome)
        attempts[-1].cost_usd = None
        known = sum(a.cost_usd or 0.0 for a in attempts[:-1])
        return known if price is not None and known > 0 else None
    if price is None:
        return None
    return sum(a.cost_usd or 0.0 for a in attempts)


def _ref(base: Record, run_dir: Path) -> str:
    """Format for parent and fork_of references: within the same run directory store only the call id; records from another run
    directory are stored as "directory:call id" to avoid colliding with local call ids."""
    if not base.run or Path(base.run).resolve() == run_dir.resolve():
        return base.id
    return f"{Path(base.run).resolve()}:{base.id}"


def _replayed(
    cache: ReplayCache,
    old: Record,
    call_id: str,
    run: str,
    req: HttpRequest,
    raw_body: bytes,
    auth_header: str | None,
) -> tuple[Record, list[RawAttempt]]:
    """New record for a replay hit: the response body, usage, outcome, and response headers are copied from the old record, and cost is recorded as 0."""
    text, is_sse = cache.raw_response(old)
    lines = [(0.0, line) for line in text.split("\n")[:-1]] if is_sse else []
    status, headers = old.response.status, dict(old.response.headers)
    started = now_iso()
    raw = RawAttempt(1, status, headers, lines, None if is_sse else text, "done", started, 0.0, 0.0)
    outcome = replace(old.outcome, flags=list(old.outcome.flags))
    attempt = Attempt(1, status, "done", outcome.protocol, "stop", "replayed", old.usage, 0.0)
    rec = Record(
        id=call_id,
        api=old.api,
        source="self",
        run=run,
        request=Request(
            req.url, redact(req.headers, auth_header), "", request_sha256(raw_body, req.headers)
        ),
        response=Response(status, headers, None, None, old.response.message_id, None),
        outcome=outcome,
        timing=Timing(started, 0, 0),
        usage=old.usage,
        cost_usd=0.0,
        attempts=[attempt],
        body=raw_body,
        message=old.message,
    )
    return rec, [raw]


def _inherit(base: Record | None, adapter: Adapter) -> dict[str, Any]:
    """Inherit the record's query string (for example Claude Code's ?beta=true) and request headers. The path uses the adapter's default path appended to the current endpoint,
    not the full path from the record: reusing the full path would duplicate a prefix when the endpoint URL already contains one (for example DeepSeek's /anthropic/anthropic/v1/messages)."""
    if base is None:
        return {}
    query = urlsplit(base.request.url).query
    headers = {
        k: v
        for k, v in base.request.headers.items()
        # redacted headers are not inherited; auth is supplied by the new endpoint
        if not is_secret_header(k) and v != REDACTED
    }
    return {"path": adapter.path + (f"?{query}" if query else ""), "inherit_headers": headers}


def _edits(base: Record | None, parsed: dict[str, Any], *, ignore_messages: bool) -> list[str]:
    """Which top-level request body fields changed relative to base. When extending a conversation (parent only), the conversation
    fields always change and are not counted: Anthropic messages and Responses input."""
    if base is None or not base.body:
        return []
    before = json.loads(base.body)
    keys = sorted(set(before) | set(parsed))
    changed = [k for k in keys if before.get(k, _MISSING) != parsed.get(k, _MISSING)]
    return [k for k in changed if not (ignore_messages and k in CONVERSATION_FIELDS)]


def _stale_thinking(
    parent: Record | None, fork_of: Record | None, parsed: dict[str, Any], api: str
) -> int:
    """After forking or extending, how many stored reasoning items have a changed context (they are kept as-is and only flagged)."""
    if fork_of is not None and fork_of.body:
        return stale_thinking(json.loads(fork_of.body), parsed, api=api)
    if parent is not None and parent.body and parent.message is not None:
        reply = parent.message.get("content") or []
        return stale_thinking(json.loads(parent.body), parsed, reply, api=api)
    return 0
