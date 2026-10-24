"""Outbound: streaming send, line-by-line receive, retry and backoff. In callkit only this module and proxy can access the network."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from callkit import sse
from callkit.adapters.base import Adapter, HttpRequest, Judgement
from callkit.config import Endpoint

CONNECT_TIMEOUT_S = 30.0
BACKOFF_CAP_S = 60.0


@dataclass
class RawAttempt:
    """Raw data received in one attempt, handed to the adapter for judgement and to storage for persistence."""

    n: int
    status: int | None
    headers: dict[str, str]
    lines: list[tuple[float, str]]  # (seconds from request start, raw line); includes comment lines
    # response body for non-200; exception class name and description when connection fails
    error_body: str | None
    transport_end: str  # done / eof / timeout / connect_error
    started_at: str
    first_byte_s: float | None
    total_s: float


def open_client(endpoint: Endpoint) -> httpx.Client:
    """One httpx connection pool per callkit.Client, shared by multiple threads. Environment proxy variables are ignored;
    set Endpoint.proxy to use a proxy, so the same config behaves the same on any machine."""
    return httpx.Client(proxy=endpoint.proxy, trust_env=False)


def send_once(http: httpx.Client, req: HttpRequest, n: int, *, idle_timeout_s: float) -> RawAttempt:
    """Send once until the stream ends or times out; no retries and no network exceptions raised (recorded in ``transport_end``).

    transport_end: ``done`` when the server closes the connection normally; ``eof`` when the connection drops mid-stream; ``timeout`` when no data arrives between lines for more than ``idle_timeout_s``; ``connect_error`` when the connection cannot be established. Whether the stream completed is judged separately by the adapter.
    """
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    status: int | None = None
    headers: dict[str, str] = {}
    lines: list[tuple[float, str]] = []
    error_body: str | None = None
    first_byte: float | None = None
    timeout = httpx.Timeout(CONNECT_TIMEOUT_S, read=idle_timeout_s)
    try:
        with http.stream(
            req.method, req.url, headers=req.headers, content=req.body, timeout=timeout
        ) as r:
            status, headers = r.status_code, dict(r.headers)
            if status != 200:
                error_body = r.read().decode("utf-8", "replace")
            else:
                for line in r.iter_lines():
                    t = time.monotonic() - t0
                    first_byte = t if first_byte is None else first_byte
                    lines.append((round(t, 3), line))
        end = "done"
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        end, error_body = "connect_error", f"{type(e).__name__}: {e}"
    except httpx.TimeoutException:
        end = "timeout"
    # RemoteProtocolError, ReadError, etc.: connection dropped mid-stream
    except httpx.TransportError as e:
        end = "eof" if status is not None else "connect_error"
        error_body = error_body if status is not None else f"{type(e).__name__}: {e}"
    total = time.monotonic() - t0
    return RawAttempt(n, status, headers, lines, error_body, end, started_at, first_byte, total)


def send(
    http: httpx.Client,
    adapter: Adapter,
    req: HttpRequest,
    *,
    max_attempts: int,
    idle_timeout_s: float,
    request: Mapping[str, Any] | None = None,
) -> list[tuple[RawAttempt, Judgement]]:
    """Send until success, a non-retryable failure, or attempts are exhausted; return every attempt and its judgement. ``request`` is the parsed request body, provided for judgement."""
    results: list[tuple[RawAttempt, Judgement]] = []
    for n in range(1, max_attempts + 1):
        raw = send_once(http, req, n, idle_timeout_s=idle_timeout_s)
        events = sse.parse(raw.lines)
        message = adapter.assemble(events)
        judgement = adapter.judge(
            raw.status, raw.transport_end, events, message, raw.error_body, request
        )
        results.append((raw, judgement))
        if judgement.outcome.ok or not judgement.retryable or n == max_attempts:
            break
        time.sleep(backoff_s(n, raw.headers.get("retry-after")))
    return results


def backoff_s(n: int, retry_after: str | None) -> float:
    """Wait time after the nth failure: use ``retry-after`` (seconds) when present, otherwise exponential backoff with random jitter.

    Exponential backoff uses min(60, 2^n) seconds multiplied by a random factor between 0.5 and 1.0 to avoid simultaneous retries by multiple threads.
    A ``retry-after`` formatted as an HTTP date is not parsed and falls back to exponential backoff.
    """
    seconds = _seconds(retry_after)
    if seconds is not None:
        return seconds
    return min(BACKOFF_CAP_S, 2.0**n) * random.uniform(0.5, 1.0)


def _seconds(value: str | None) -> float | None:
    if value is None or not value.strip().replace(".", "", 1).isdigit():
        return None
    return float(value)
