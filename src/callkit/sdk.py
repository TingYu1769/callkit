"""Record under the official SDK: pass the callkit httpx client to the SDK's ``http_client``; requests to the three recorded APIs are saved as callkit records (``source="sdk"``), while other requests (embeddings, models, ...) pass through unrecorded.

    client = OpenAI(..., http_client=callkit.recording_http_client(run="runs/E1"))
    aclient = AsyncOpenAI(..., http_client=callkit.recording_async_http_client(run="runs/E1"))
    claude = Anthropic(..., http_client=callkit.recording_http_client(run="runs/E1", lib="httpx2"))

- ``lib``: ``"httpx"`` or ``"httpx2"``. anthropic SDK 2.x only accepts httpx2; openai SDK 3.x accepts both; older SDKs use httpx.
- ``run=None`` returns a plain client with no recording and no request header changes.
- Clients never read proxy environment variables (``trust_env=False``, same as ``Endpoint``); pass ``proxy`` to use a proxy. The proxy is only assigned to the inner transport, otherwise httpx creates a separate proxy tunnel that bypasses recording. No timeout is set; the SDK uses its own default.
- Recorded requests set ``accept-encoding`` to ``identity`` (SDK default wants gzip); if the server still compresses, gzip and deflate are decompressed before parsing, other encodings are passed through unparsed.
- Records are written when the SDK reads and closes the response; failures without a response such as connection errors and timeouts are recorded first and then re-raised. SDK retries are recorded separately.
- The same run directory shares one ``Recorder`` in the same process; it will not conflict with proxy or other recording clients.
"""

from __future__ import annotations

import codecs
import contextlib
import importlib
import time
import warnings
import zlib
from collections.abc import AsyncIterator, Iterator, Mapping
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

from callkit.client import request_sha256
from callkit.config import PriceTable
from callkit.errors import CallkitError
from callkit.recorder import REFUSAL_HEADERS, Exchange, Recorder, recorded_api
from callkit.replay import ReplayCache

LIBS = ("httpx", "httpx2")
# Response headers not copied during replay: the body is re-sent as the full original text (not compressed, not chunked), and the length is computed by the library.
_NOT_REPLAYED = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def recording_http_client(
    run: str | Path | None = None,
    *,
    lib: str = "httpx",
    session: str | None = None,
    tags: Mapping[str, str] | None = None,
    proxy: str | None = None,
    transport: Any = None,
    prices: PriceTable | None = None,
    cap_usd: float | None = None,
    replay_from: str | Path | None = None,
    **client_kwargs: Any,
) -> Any:
    """Return a synchronous client (``lib``'s ``Client``) for the ``http_client`` argument of ``OpenAI(...)`` and ``Anthropic(...)``.

    ``session`` is the session key (parent is inferred only within a session); ``tags`` are merged into each record; ``transport`` can replace the inner transport (for example MockTransport for tests), and then ``proxy`` and transport options cannot be given. Transport options such as ``verify``, ``limits``, and ``http2`` go to the inner transport; remaining options go to the client; ``mounts`` is not accepted (would bypass recording).

    ``prices`` gives per-model unit prices used to compute ``cost_usd`` for recorded calls (and is written to the run directory's prices.json); when ``cap_usd`` is set and the run directory's accumulated cost reaches the cap, or the model price is missing, the request is not sent and an HTTP 402 is returned directly (with ``x-should-retry: false``; the SDK does not retry and raises APIStatusError).

    ``replay_from`` given an old run directory enables the replay cache: a request identical to a previously successful call is not sent; the old response text is used to synthesize an HTTP response for the SDK. The new run still records one entry, with a ``replay_of`` tag pointing to the old record and cost recorded as 0."""
    mod = _lib(lib)
    inner = _inner(mod, transport, proxy, client_kwargs, mod.HTTPTransport)
    if run is not None:
        inner = _classes(lib)[0](
            inner,
            _recorder(run, prices),
            session,
            dict(tags or {}),
            cap_usd,
            _replay(replay_from, run),
        )
    else:
        _ignored(prices, cap_usd, replay_from)
    client_kwargs.setdefault("follow_redirects", True)  # match the SDK's own client
    return mod.Client(transport=inner, trust_env=False, **client_kwargs)


def recording_async_http_client(
    run: str | Path | None = None,
    *,
    lib: str = "httpx",
    session: str | None = None,
    tags: Mapping[str, str] | None = None,
    proxy: str | None = None,
    transport: Any = None,
    prices: PriceTable | None = None,
    cap_usd: float | None = None,
    replay_from: str | Path | None = None,
    **client_kwargs: Any,
) -> Any:
    """Return an async client (``lib``'s ``AsyncClient``) for ``AsyncOpenAI(...)`` and ``AsyncAnthropic(...)``. Arguments are the same as above.
    Recording writes synchronously inside the coroutine (a short local write) and briefly blocks the event loop."""
    mod = _lib(lib)
    inner = _inner(mod, transport, proxy, client_kwargs, mod.AsyncHTTPTransport)
    if run is not None:
        inner = _classes(lib)[1](
            inner,
            _recorder(run, prices),
            session,
            dict(tags or {}),
            cap_usd,
            _replay(replay_from, run),
        )
    else:
        _ignored(prices, cap_usd, replay_from)
    client_kwargs.setdefault("follow_redirects", True)  # match the SDK's own client
    return mod.AsyncClient(transport=inner, trust_env=False, **client_kwargs)


class _Tap:
    """One recorded exchange: receives response bytes (decompressing when necessary, decoding as UTF-8 into timestamped chunks), writes the record at the end, only once."""

    def __init__(
        self, recorder: Recorder, ex: Exchange, session: str | None, tags: dict[str, str]
    ) -> None:
        self.recorder, self.ex, self.session, self.tags = recorder, ex, session, tags
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._unzip: Any = None
        self._raw_deflate_retry = (
            # if the first deflate block fails to decompress, retry without the zlib wrapper (matching httpx)
            False
        )
        # opaque compressed encoding: pass through only, do not parse or save
        self._opaque: str | None = None
        self._finished = False

    def start(self, status: int, headers: Any) -> None:
        self.ex.status, self.ex.response_headers = status, dict(headers)
        encoding = str(headers.get("content-encoding", "")).lower().strip()
        if encoding in ("gzip", "x-gzip"):
            self._unzip = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            self._unzip, self._raw_deflate_retry = zlib.decompressobj(), True
        elif encoding not in ("", "identity"):
            self._opaque = encoding

    def feed(self, chunk: bytes, *, decoded: bool = False) -> None:
        if self._opaque or not chunk:
            return
        try:
            data = chunk if decoded or self._unzip is None else self._decompress(chunk)
        except zlib.error:
            self._opaque = "corrupt compressed data"
            return
        self._append(self._decoder.decode(data))

    def _decompress(self, chunk: bytes) -> bytes:
        retry, self._raw_deflate_retry = self._raw_deflate_retry, False
        try:
            return self._unzip.decompress(chunk)
        except zlib.error:
            if not retry:
                raise
            self._unzip = zlib.decompressobj(-zlib.MAX_WBITS)
            return self._unzip.decompress(chunk)

    def fail(self, exc: BaseException, mod: ModuleType) -> None:
        if isinstance(exc, (mod.ConnectError, mod.ConnectTimeout, mod.PoolTimeout, mod.ProxyError)):
            self.ex.end = "connect_error"
        elif isinstance(exc, mod.TimeoutException):
            self.ex.end = "timeout"
        elif isinstance(exc, Exception):
            # other transport errors and connection errors the library does not map (such as ssl.SSLError)
            self.ex.end = "eof"
        else:
            self.ex.end = "client_gone"  # asyncio cancellation or Ctrl-C
        self.ex.error = f"{type(exc).__name__}: {exc}"

    def finish(self, *, complete: bool = True) -> None:
        if self._finished:
            return
        self._finished = True
        if self._opaque:
            self.ex.error = (
                f"response is compressed with {self._opaque}; callkit does not decode it"
            )
        else:
            if self._unzip is not None:
                with contextlib.suppress(zlib.error):
                    self._append(self._decoder.decode(self._unzip.flush()))
            self._append(self._decoder.decode(b"", final=True))
        if not complete and self.ex.end == "done":
            self.ex.end = "client_gone"  # caller closed the response before reading it fully
        self.recorder.record_safely(self.ex, source="sdk", session=self.session, tags=self.tags)

    def _append(self, text: str) -> None:
        if text:
            self.ex.chunks.append((time.monotonic() - self.ex.t0, text))


class _SyncStream:
    """Wraps the response byte stream: each chunk read by the SDK is recorded; the record is written when the SDK closes the response."""

    _lib: ModuleType

    def __init__(self, inner: Any, tap: _Tap) -> None:
        self._inner, self._tap, self._complete = inner, tap, False

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._inner:
                self._tap.feed(chunk)
                yield chunk
        # Record immediately on any exception, including cancellation: after a stream read
        # error or cancellation while reading an error body, httpx may not close the response.
        except BaseException as e:
            self._tap.fail(e, self._lib)
            self._tap.finish()
            raise
        self._complete = True

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._tap.finish(complete=self._complete or self._tap.ex.end != "done")


class _AsyncStream:
    _lib: ModuleType

    def __init__(self, inner: Any, tap: _Tap) -> None:
        self._inner, self._tap, self._complete = inner, tap, False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                self._tap.feed(chunk)
                yield chunk
        except BaseException as e:
            self._tap.fail(e, self._lib)
            self._tap.finish()
            raise
        self._complete = True

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            self._tap.finish(complete=self._complete or self._tap.ex.end != "done")


class _SyncTransport:
    _lib: ModuleType
    _stream: type[_SyncStream]

    def __init__(
        self,
        inner: Any,
        recorder: Recorder,
        session: str | None,
        tags: dict[str, str],
        cap_usd: float | None = None,
        replay: ReplayCache | None = None,
    ) -> None:
        self._inner, self._recorder, self._session, self._tags = inner, recorder, session, tags
        self._cap_usd, self._replay = cap_usd, replay

    def handle_request(self, request: Any) -> Any:
        if recorded_api(request.method, str(request.url)) is None:
            return self._inner.handle_request(request)
        body = request.read()
        hit = _replayed(self, request, body)
        if hit is not None:  # replay cache hit: do not send
            return hit
        refused = self._recorder.refusal(body, self._cap_usd)
        if refused is not None:  # cap reached: do not send, return a fake 402 (not recorded)
            return self._lib.Response(
                402, headers=REFUSAL_HEADERS, content=refused, request=request
            )
        tap = _tap_for(request, body, self)
        try:
            response = self._inner.handle_request(request)
        # failures without a response (connection errors, timeouts, cancellation, etc.): record one entry by category and re-raise
        except BaseException as e:
            tap.fail(e, self._lib)
            tap.finish()
            raise
        return _wrap(response, tap, self._stream)

    def close(self) -> None:
        self._inner.close()


class _AsyncTransport:
    _lib: ModuleType
    _stream: type[_AsyncStream]

    def __init__(
        self,
        inner: Any,
        recorder: Recorder,
        session: str | None,
        tags: dict[str, str],
        cap_usd: float | None = None,
        replay: ReplayCache | None = None,
    ) -> None:
        self._inner, self._recorder, self._session, self._tags = inner, recorder, session, tags
        self._cap_usd, self._replay = cap_usd, replay

    async def handle_async_request(self, request: Any) -> Any:
        if recorded_api(request.method, str(request.url)) is None:
            return await self._inner.handle_async_request(request)
        body = await request.aread()
        hit = _replayed(self, request, body)
        if hit is not None:  # replay cache hit: do not send
            return hit
        refused = self._recorder.refusal(body, self._cap_usd)
        if refused is not None:  # cap reached: do not send, return a fake 402 (not recorded)
            return self._lib.Response(
                402, headers=REFUSAL_HEADERS, content=refused, request=request
            )
        tap = _tap_for(request, body, self)
        try:
            response = await self._inner.handle_async_request(request)
        # failures without a response (connection errors, timeouts, cancellation, etc.): record one entry by category and re-raise
        except BaseException as e:
            tap.fail(e, self._lib)
            tap.finish()
            raise
        return _wrap(response, tap, self._stream)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _replayed(t: _SyncTransport | _AsyncTransport, request: Any, body: bytes) -> Any:
    """On a replay cache hit, synthesize an HTTP response from the old response text and record one entry in the new run; return None on a miss."""
    if t._replay is None:
        return None
    headers_in = dict(request.headers)
    old = t._replay.take(str(request.url), headers_in, request_sha256(body, headers_in))
    if old is None:
        return None
    text, _ = t._replay.raw_response(old)
    headers = {k: v for k, v in old.response.headers.items() if k.lower() not in _NOT_REPLAYED}
    ex = Exchange(request.method, str(request.url), dict(request.headers), body)
    ex.status, ex.response_headers, ex.chunks = old.response.status, headers, [(0.0, text)]
    tags = {**t._tags, "replay_of": t._replay.ref(old)}
    t._recorder.record_safely(ex, source="sdk", session=t._session, tags=tags, replayed=True)
    status = old.response.status or 200
    return t._lib.Response(status, headers=headers, content=text.encode(), request=request)


def _tap_for(request: Any, body: bytes, t: _SyncTransport | _AsyncTransport) -> _Tap:
    """Record the SDK's original request headers and body, then change accept-encoding to identity before sending."""
    ex = Exchange(request.method, str(request.url), dict(request.headers), body)
    request.headers["accept-encoding"] = "identity"
    return _Tap(t._recorder, ex, t._session, t._tags)


def _wrap(response: Any, tap: _Tap, stream_cls: type[Any]) -> Any:
    tap.start(response.status_code, response.headers)
    # inner transport already read the full response (for example MockTransport's full content)
    if response.is_closed:
        tap.feed(response.content, decoded=True)
        tap.finish()
        return response
    response.stream = stream_cls(response.stream, tap)
    return response


@cache
def _classes(lib: str) -> tuple[type[Any], type[Any]]:
    """Generate recording transports for a library: httpx requires response streams to be instances of that library's SyncByteStream / AsyncByteStream."""
    mod = _lib(lib)
    stream = type("RecordingStream", (_SyncStream, mod.SyncByteStream), {"_lib": mod})
    astream = type("AsyncRecordingStream", (_AsyncStream, mod.AsyncByteStream), {"_lib": mod})
    sync = type(
        "RecordingTransport", (_SyncTransport, mod.BaseTransport), {"_lib": mod, "_stream": stream}
    )
    asyn = type(
        "AsyncRecordingTransport",
        (_AsyncTransport, mod.AsyncBaseTransport),
        {"_lib": mod, "_stream": astream},
    )
    return sync, asyn


def _replay(replay_from: str | Path | None, run: str | Path) -> ReplayCache | None:
    return None if replay_from is None else ReplayCache.shared(replay_from, run)


def _ignored(prices: PriceTable | None, cap_usd: float | None, replay_from: Any) -> None:
    if prices or cap_usd is not None or replay_from is not None:
        warnings.warn(
            "run=None disables recording; prices, cap_usd and replay_from are ignored",
            RuntimeWarning,
            stacklevel=3,
        )


def _recorder(run: str | Path, prices: PriceTable | None) -> Recorder:
    recorder = Recorder.shared(run)
    recorder.add_prices(prices)
    return recorder


def _lib(name: str) -> ModuleType:
    if name not in LIBS:
        raise CallkitError(f"lib must be one of {LIBS}, got {name!r}")
    return importlib.import_module(name)


# These options are only useful for the inner transport; passing them to the client would be ignored.
TRANSPORT_KWARGS = (
    "verify",
    "cert",
    "http1",
    "http2",
    "limits",
    "uds",
    "local_address",
    "retries",
    "socket_options",
)


def _inner(
    mod: ModuleType, transport: Any, proxy: str | None, kwargs: dict[str, Any], cls: Any
) -> Any:
    """Build the inner transport and take its options from ``kwargs``."""
    if "mounts" in kwargs:
        raise CallkitError(
            "recording clients do not accept mounts: requests routed through mounts would bypass recording"
        )
    if kwargs.pop("trust_env", False):
        raise CallkitError(
            "recording clients ignore proxy environment variables (trust_env=False); pass proxy= instead"
        )
    own = {k: kwargs.pop(k) for k in TRANSPORT_KWARGS if k in kwargs}
    if transport is not None:
        if proxy is not None or own:
            raise CallkitError(
                "pass either proxy or transport, not both: configure the proxy and transport options on the transport you pass"
            )
        return transport
    # Connection pool limits match the two SDKs' defaults
    own.setdefault("limits", mod.Limits(max_connections=1000, max_keepalive_connections=100))
    return cls(proxy=proxy, **own)
