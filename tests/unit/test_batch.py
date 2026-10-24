"""Client.batch: concurrency limit, completion order, resume support, changed input error, cap."""

import threading
import time

import httpx
import pytest

from callkit import Client, Endpoint, Price, transport
from callkit.errors import CapReached, ItemChanged, MissingPrice
from tests.fakes import error_response, good_stream, sse_response

EP = Endpoint("gw", "http://gw:3000", "CALLKIT_TEST_KEY")
PRICE = Price(input=3, output=15, cache_read=0.3, cache_write=3.75, cache_write_by_ttl={"1h": 6})


def body(text, model="claude-sonnet-5-5"):
    return {"model": model, "max_tokens": 10, "messages": [{"role": "user", "content": text}]}


class ConcurrentServer:
    """Thread-safe fake server: decides the response from each request's user message and records the maximum in-flight count."""

    def __init__(self, delay=0.0, fail=()):
        self.delay, self.fail = delay, set(fail)
        self.lock = threading.Lock()
        self.inflight = self.peak = 0
        self.seen: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        text = request.content.decode()
        with self.lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            self.seen.append(text)
        time.sleep(self.delay)
        with self.lock:
            self.inflight -= 1
        if any(f in text for f in self.fail):
            return error_response(400, "invalid_request_error")
        return sse_response(good_stream())


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLKIT_TEST_KEY", "sk-test")
    monkeypatch.setattr(
        transport, "backoff_s", lambda n, retry_after: 0.0
    )  # Do not replace time.sleep globally; the fake server needs a real sleep.

    def factory(server, run="run", **kw):
        monkeypatch.setattr(
            transport,
            "open_client",
            lambda ep: httpx.Client(transport=httpx.MockTransport(server.handler)),
        )
        return Client(EP, tmp_path / run, **kw)

    return factory


def test_concurrency_cap_and_all_yielded(make):
    server = ConcurrentServer(delay=0.05)
    c = make(server)
    recs = list(c.batch({f"q{i}": body(f"问题{i}") for i in range(8)}, concurrency=3))
    assert len(recs) == 8 and server.peak == 3
    assert {(r.item_id, r.repeat) for r in recs} == {(f"q{i}", 0) for i in range(8)}


def test_resume_skips_done_and_redoes_failed(make):
    server = ConcurrentServer(fail={"问题2"})
    items = {"q1": body("问题1"), "q2": body("问题2")}
    first = list(make(server).batch(items, repeats=2))
    assert len(first) == 4 and sum(r.outcome.ok for r in first) == 2
    server.fail.clear()
    again = list(make(server).batch(items, repeats=2))
    assert sorted((r.item_id, r.repeat) for r in again) == [("q2", 0), ("q2", 1)]
    assert list(make(server).batch(items, repeats=2)) == [] and len(server.seen) == 6


def test_changed_item_raises_before_sending(make):
    server = ConcurrentServer()
    list(make(server).batch({"q1": body("问题1")}))
    with pytest.raises(ItemChanged, match="q1"):
        list(make(server).batch({"q1": body("问题1（改过）"), "q2": body("问题2")}))
    assert len(server.seen) == 1


def test_cap_stops_new_requests(make):
    server = ConcurrentServer()
    c = make(server, prices={"claude-sonnet-5-5": PRICE}, cap_usd=1e-9)
    it = c.batch({f"q{i}": body(f"问题{i}") for i in range(5)}, concurrency=1)
    first = next(it)
    with pytest.raises(CapReached, match="remaining jobs"):
        next(it)
    assert first.outcome.ok and len(server.seen) == 1


def test_other_errors_propagate(make):
    server = ConcurrentServer()
    c = make(server, cap_usd=1.0)
    with pytest.raises(MissingPrice):
        list(c.batch({"q1": body("问题1", model="no-price")}))
    assert server.seen == []


def test_early_break_waits_for_inflight(make):
    server = ConcurrentServer(delay=0.02)
    c = make(server)
    for _ in c.batch({f"q{i}": body(f"问题{i}") for i in range(20)}, concurrency=2):
        break
    n = len(server.seen)
    time.sleep(0.1)
    assert len(server.seen) == n and n < 20 and len(c.store.index()) == n
