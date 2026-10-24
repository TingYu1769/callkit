"""Local recording proxy: transparent byte-for-byte forwarding, recording one entry per model call. Inside callkit only this module, transport, and sdk
may access the network. Which requests to record, how to decide, and how parent is inferred live in ``callkit.recorder``; here only forwards.

WebSocket upgrade requests return 426 and are not forwarded: when Codex sees 426 it switches the whole session to HTTP.

Usage: ``callkit proxy --upstream <gateway> --run runs/X``, then
- Claude Code: start with ``ANTHROPIC_BASE_URL=http://127.0.0.1:<port>``. When launching Claude Code from an existing Claude Code session, first
  clear inherited CLAUDE* and ANTHROPIC* environment variables (otherwise Not logged in).
- Codex: use a fresh ``CODEX_HOME`` plus a custom provider (name must not be OpenAI):
  ``-c 'model_providers.callkit={name="callkit",base_url="http://127.0.0.1:<port>/v1",wire_api="responses",env_key="<variable>"}' -c model_provider=callkit``.
"""

from __future__ import annotations

import codecs
import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from callkit import version
from callkit.config import PriceTable
from callkit.recorder import Exchange, Recorder, recorded_api

# Hop-by-hop request headers are not forwarded; drop accept-encoding so the upstream returns uncompressed text.
_HOP = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "accept-encoding",
}


class Proxy:
    """Multithreaded recording proxy. ``start()`` starts a background thread, ``port`` is the actual listening port, and ``stop()`` stops it."""

    def __init__(
        self,
        upstream: str,
        run: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        upstream_proxy: str | None = None,
        idle_timeout_s: float = 600.0,
        prices: PriceTable | None = None,
        cap_usd: float | None = None,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        meta = {
            "callkit_version": version.version(),
            "callkit_commit": version.commit(),
            "proxy": {"upstream": self.upstream, "upstream_proxy": upstream_proxy is not None},
        }
        self.recorder = Recorder.shared(run, run_meta=meta)
        self.recorder.add_prices(prices)  # recorded calls are priced using this
        self.cap_usd = cap_usd
        self.store = self.recorder.store
        self._http = httpx.Client(proxy=upstream_proxy, trust_env=False)
        self._timeout = httpx.Timeout(30.0, read=idle_timeout_s)
        handler = type("Handler", (_Handler,), {"proxy": self})
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self._server.server_address[0]}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="callkit-proxy", daemon=True
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._http.close()

    def relay(self, h: _Handler) -> None:
        """Forward one request: forward each chunk to the client as it arrives; after forwarding, write a record if appropriate; only then send the final chunked terminator."""
        if h.headers.get("upgrade", "").lower() == "websocket":
            # only HTTP is recorded; Codex sees 426 and switches the whole session to HTTP
            _send_upgrade_required(h)
            return
        length = int(h.headers.get("content-length") or 0)
        body = h.rfile.read(length) if length else b""
        ex = Exchange(h.command, self.upstream + h.path, dict(h.headers), body)
        refused = (
            self.recorder.refusal(body, self.cap_usd) if recorded_api(ex.method, ex.url) else None
        )
        # cap reached: do not forward; return a fake 402 to the client (not recorded)
        if refused is not None:
            _send_json(h, 402, refused, {"x-should-retry": "false"})
            return
        headers = {k: v for k, v in h.headers.items() if k.lower() not in _HOP}
        headers["accept-encoding"] = (
            # ask upstream not to compress so recorded data is the original text (httpx defaults to requesting gzip)
            "identity"
        )
        self._forward(h, ex, headers)
        # not recorded when the request is not a POST to one of the three APIs
        self.recorder.record_safely(ex, source="proxy")
        # write the record before finishing: when the client sees the response end the record is already on disk, so the next request's parent inference will not miss it
        if ex.status is None:
            _send_bad_gateway(h, ex.error or ex.end)
        elif ex.end == "done":
            with contextlib.suppress(OSError):
                _send_chunk(h, b"")
        else:
            # upstream broke mid-response: close the downstream connection so the client knows the response is incomplete and does not wait forever
            h.close_connection = True

    def _forward(self, h: _Handler, ex: Exchange, headers: dict[str, str]) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            with self._http.stream(
                h.command,
                self.upstream + h.path,
                headers=headers,
                content=ex.body,
                timeout=self._timeout,
            ) as r:
                ex.status, ex.response_headers = r.status_code, dict(r.headers)
                _send_head(h, r)
                for chunk in r.iter_bytes():
                    ex.chunks.append((time.monotonic() - ex.t0, decoder.decode(chunk)))
                    _send_chunk(h, chunk)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            ex.end, ex.error = "connect_error", f"{type(e).__name__}: {e}"
        except httpx.TimeoutException:
            ex.end = "timeout"
        except httpx.TransportError as e:
            ex.end, ex.error = "eof", f"{type(e).__name__}: {e}"
        except OSError:
            # client disconnected first; the upstream stream closes when the with block exits
            ex.end = "client_gone"
        tail = decoder.decode(b"", final=True)
        if tail:
            ex.chunks.append((time.monotonic() - ex.t0, tail))


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    proxy: Proxy

    def log_message(self, format: str, *args: Any) -> None:
        pass  # do not write access logs to stderr

    def do_POST(self) -> None:
        self.proxy.relay(self)

    do_GET = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_POST


def serve(upstream: str, run: str | Path, *, host: str = "127.0.0.1", port: int = 0) -> None:
    """Start a multithreaded HTTP server until interrupted. Requests and responses are forwarded byte for byte unchanged, and query strings in paths are preserved;
    each chunk is forwarded as it arrives; auth headers are passed to the upstream unchanged and masked when stored; parent is inferred by chain.infer_parent,
    searching within the same session (session key is taken in order from the X-Claude-Code-Session-Id header, Session-Id header, and prompt_cache_key)."""
    proxy = Proxy(upstream, run, host=host, port=port)
    try:
        proxy.serve_forever()
    finally:
        proxy.stop()


def _send_upgrade_required(h: _Handler) -> None:
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "callkit proxy only records HTTP; please use HTTP SSE instead of WebSocket",
            },
        }
    ).encode()
    with contextlib.suppress(OSError):
        h.send_response(426)
        h.send_header("content-type", "application/json")
        h.send_header("content-length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)


def _send_json(h: _Handler, status: int, data: bytes, headers: dict[str, str]) -> None:
    with contextlib.suppress(OSError):  # ignore if the client already disconnected
        h.send_response(status)
        h.send_header("content-type", "application/json")
        h.send_header("content-length", str(len(data)))
        for k, v in headers.items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(data)


def _send_head(h: _Handler, r: httpx.Response) -> None:
    h.send_response(r.status_code)
    for k, v in r.headers.multi_items():
        if k.lower() not in _HOP and k.lower() != "content-encoding":
            h.send_header(k, v)
    h.send_header("transfer-encoding", "chunked")
    h.end_headers()


def _send_chunk(h: _Handler, chunk: bytes) -> None:
    """Write one chunk in chunked encoding; an empty chunk indicates the end."""
    h.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
    h.wfile.flush()


def _send_bad_gateway(h: _Handler, error: str) -> None:
    body = json.dumps(
        {"type": "error", "error": {"type": "api_error", "message": f"callkit proxy: {error}"}}
    )
    data = body.encode()
    with contextlib.suppress(OSError):  # ignore if the client already disconnected
        h.send_response(502)
        h.send_header("content-type", "application/json")
        h.send_header("content-length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)
