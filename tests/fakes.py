"""Fake server for tests: replays pre-arranged responses in order using httpx.MockTransport."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

DATA = Path(__file__).resolve().parent / "data"


def good_stream() -> bytes:
    return (DATA / "stream_tool_use.sse").read_bytes()


def sse_response(content: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


def error_response(status: int, err_type: str, retry_after: str | None = None) -> httpx.Response:
    body = json.dumps({"type": "error", "error": {"type": err_type, "message": "x"}})
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.Response(status, headers=headers, content=body.encode())


class BreakingStream(httpx.SyncByteStream):
    """Yields some bytes first, then raises RemoteProtocolError as if the peer cut the connection."""

    def __init__(self, head: bytes) -> None:
        self.head = head

    def __iter__(self) -> Iterator[bytes]:
        yield self.head
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )


class Server:
    """Replays ``replies`` in order; each item is a Response, or an exception to raise. Received requests are recorded in ``requests``."""

    def __init__(self, *replies: httpx.Response | Exception | Callable[[], httpx.Response]) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply() if callable(reply) else reply

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))
