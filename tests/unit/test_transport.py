"""Transport layer: four outcomes of one send, retry, backoff."""

import httpx
import pytest

from callkit import transport
from callkit.adapters.anthropic import AnthropicMessages
from callkit.adapters.base import HttpRequest
from tests.fakes import BreakingStream, Server, error_response, good_stream, sse_response

REQ = HttpRequest("POST", "http://gw/v1/messages", {"x-api-key": "sk"}, b'{"stream":true}')
A = AnthropicMessages()


def once(server):
    return transport.send_once(server.client(), REQ, 1, idle_timeout_s=5)


def test_send_once_done():
    raw = once(Server(sse_response(good_stream())))
    assert raw.status == 200 and raw.transport_end == "done" and raw.error_body is None
    assert len(raw.lines) == good_stream().decode().count("\n") and raw.first_byte_s is not None


def test_send_once_http_error_keeps_body():
    raw = once(Server(error_response(529, "overloaded_error")))
    assert raw.status == 529 and raw.lines == [] and "overloaded_error" in (raw.error_body or "")


def test_send_once_connect_error():
    raw = once(Server(httpx.ConnectError("connection refused")))
    assert raw.status is None and raw.transport_end == "connect_error"
    assert "ConnectError" in (raw.error_body or "")


def test_send_once_timeout():
    raw = once(Server(httpx.ReadTimeout("idle")))
    assert raw.status is None and raw.transport_end == "timeout"


def test_send_once_eof_mid_stream():
    head = good_stream()[:1500]
    raw = once(Server(lambda: httpx.Response(200, stream=BreakingStream(head))))
    assert raw.status == 200 and raw.transport_end == "eof"
    assert 0 < len(raw.lines) < good_stream().decode().count("\n")


@pytest.fixture
def slept(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(transport.time, "sleep", waits.append)
    return waits


def send(server, max_attempts=3):
    return transport.send(server.client(), A, REQ, max_attempts=max_attempts, idle_timeout_s=5)


def test_retry_then_success(slept):
    server = Server(
        error_response(529, "overloaded_error", retry_after="0"), sse_response(good_stream())
    )
    results = send(server)
    assert [j.outcome.ok for _, j in results] == [False, True]
    assert slept == [0.0] and len(server.requests) == 2


def test_eof_is_retried(slept):
    head = good_stream()[:1500]
    server = Server(
        lambda: httpx.Response(200, stream=BreakingStream(head)), sse_response(good_stream())
    )
    results = send(server)
    assert results[0][1].reason == "no_message_stop" and results[1][1].outcome.ok


def test_no_retry_on_400_or_bad_assembly(slept):
    assert len(send(Server(error_response(400, "invalid_request_error")))) == 1
    bad = good_stream().replace(
        b'"signature_delta","signature":"', b'"signature_delta","signature":"","x":"'
    )
    results = send(Server(sse_response(bad)))
    assert len(results) == 1 and results[0][1].reason == "bad_assembly"
    assert slept == []


def test_attempts_capped(slept):
    server = Server(*[error_response(529, "overloaded_error", retry_after="0")] * 3)
    results = send(server, max_attempts=3)
    assert len(results) == 3 and not results[-1][1].outcome.ok and len(slept) == 2


def test_backoff():
    assert transport.backoff_s(1, "2.5") == 2.5
    assert 2.0 <= transport.backoff_s(2, "Wed, 21 Oct 2026 07:28:00 GMT") <= 4.0
    assert 30.0 <= transport.backoff_s(10, None) <= 60.0
