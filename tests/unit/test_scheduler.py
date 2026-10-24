"""Scheduler over the SDK: concurrency, completion order, resume, error handling, records attached to items, cap, stopping midway. No network."""

import asyncio
import json
import threading
import time

import httpx
import openai
import pytest

from callkit import (
    Client,
    Endpoint,
    Price,
    arun_items,
    recording_async_http_client,
    recording_http_client,
    run_items,
    transport,
)
from callkit.errors import CallkitError, CapReached
from callkit.store import RunStore
from tests.fakes import DATA, Server, good_stream, sse_response

CHAT = (DATA / "chat_tool_calls.json").read_bytes()
PRICE = {"m": Price(input=1.25, output=10, cache_read=0.125, cache_write=1.5625)}


def reply(request):
    return httpx.Response(200, headers={"content-type": "application/json"}, content=CHAT)


def ledger(run):
    path = run / "items.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_values_resume_and_errors(tmp_path):
    calls = []

    def fn(x):
        calls.append(x)
        if x == 2 and len([c for c in calls if c == 2]) == 1:
            raise ValueError("第一次坏了")
        return x * 10

    items = {"a": 1, "b": 2, "c": 3}
    got = {r.item_id: r for r in run_items(items, fn, run=tmp_path / "run", concurrency=2)}
    assert got["a"].value == 10 and got["c"].ok and not got["b"].ok
    assert got["b"].error == "ValueError: 第一次坏了" and got["b"].value is None
    again = list(run_items(items, fn, run=tmp_path / "run"))  # Resume: only rerun the failed b
    assert [(r.item_id, r.ok, r.value) for r in again] == [("b", True, 20)]
    assert list(run_items(items, fn, run=tmp_path / "run")) == []
    rows = ledger(tmp_path / "run")
    assert [(x["item_id"], x["ok"]) for x in rows].count(("b", False)) == 1 and len(rows) == 4


def test_concurrency_limit_and_completion_order(tmp_path):
    running, peak, lock = [0], [0], threading.Lock()

    def fn(delay):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(delay)
        with lock:
            running[0] -= 1
        return delay

    items = {"slow": 0.3, "fast": 0.01, "mid": 0.1, "x": 0.01, "y": 0.01}
    order = [r.item_id for r in run_items(items, fn, run=tmp_path / "run", concurrency=2)]
    assert peak[0] == 2 and order[0] == "fast" and order[-1] == "slow"


def test_records_carry_item_and_call_ids(tmp_path):
    """Records from SDK calls are attached to their own items and do not mix across threads; each repeat runs once."""
    run = tmp_path / "run"
    client = openai.OpenAI(
        base_url="http://gw.test/v1",
        api_key="sk",
        http_client=recording_http_client(run, transport=httpx.MockTransport(reply)),
    )

    def ask(text):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": text}])
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": text}])

    items = {f"q{i}": f"q{i}" for i in range(6)}
    results = list(run_items(items, ask, run=run, concurrency=4, repeats=2))
    assert len(results) == 12 and all(len(r.call_ids) == 2 for r in results)
    store = RunStore(run)
    for r in results:
        for cid in r.call_ids:
            rec = store.load(cid)
            sent = json.loads(rec.body)["messages"][0]["content"]
            assert (rec.item_id, rec.repeat) == (r.item_id, r.repeat) and sent == r.item_id
    assert ledger(run)[0]["call_ids"]


def test_client_inside_fn_gets_item(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLKIT_TEST_KEY", "sk")
    s = Server(*[sse_response(good_stream()) for _ in range(2)])
    monkeypatch.setattr(transport, "open_client", lambda endpoint: s.client())
    run = tmp_path / "run"
    with Client(Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY"), run) as c:
        body = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "x"}],
        }
        results = list(run_items({"a": body, "b": body}, c.call, run=run, concurrency=1))
    assert {(r.value.item_id, r.value.repeat) for r in results} == {("a", 0), ("b", 0)}
    assert all(r.call_ids == [r.value.id] for r in results)


def test_cap_stops_new_items(tmp_path):
    run = tmp_path / "run"
    client = openai.OpenAI(
        base_url="http://gw.test/v1",
        api_key="sk",
        http_client=recording_http_client(run, transport=httpx.MockTransport(reply), prices=PRICE),
        max_retries=0,
    )

    def ask(text):
        return client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": text}]
        )

    got = []
    with pytest.raises(CapReached, match="remaining items were not started"):
        for r in run_items(
            {"a": "a", "b": "b", "c": "c"}, ask, run=run, concurrency=1, cap_usd=1e-9
        ):
            got.append(r.item_id)
    assert got == ["a"] and len(RunStore(run).index()) == 1


def test_break_cancels_queued(tmp_path):
    started = []

    def fn(x):
        started.append(x)
        time.sleep(0.05)
        return x

    items = {str(i): i for i in range(10)}
    for _ in run_items(items, fn, run=tmp_path / "run", concurrency=1):
        break
    time.sleep(0.2)
    assert len(started) <= 2 and len(ledger(tmp_path / "run")) == len(started)


def test_item_ids_must_be_strings(tmp_path):
    with pytest.raises(CallkitError, match="must be strings"):
        list(run_items({1: "x"}, str, run=tmp_path / "run"))


def test_async_records_errors_resume(tmp_path):
    run = tmp_path / "run"
    http = recording_async_http_client(run, transport=httpx.MockTransport(reply))
    seen = []

    async def main():
        client = openai.AsyncOpenAI(base_url="http://gw.test/v1", api_key="sk", http_client=http)

        async def ask(text):
            seen.append(text)
            if text == "bad" and seen.count("bad") == 1:
                raise RuntimeError("坏了")
            await asyncio.sleep(0.01 if text != "q0" else 0.05)
            resp = await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": text}]
            )
            return resp.id

        items = {"q0": "q0", "q1": "q1", "q2": "q2", "bad": "bad"}
        first = [r async for r in arun_items(items, ask, run=run, concurrency=3)]
        second = [r async for r in arun_items(items, ask, run=run)]
        await client.close()
        return first, second

    first, second = asyncio.run(main())
    by_id = {r.item_id: r for r in first}
    assert not by_id["bad"].ok and by_id["bad"].error == "RuntimeError: 坏了"
    assert by_id["q1"].ok and by_id["q1"].value and [r.item_id for r in second] == ["bad"]
    assert [r.item_id for r in first].index("q0") > [r.item_id for r in first].index("q1")
    assert all(len(r.call_ids) == 1 for r in [*first, *second] if r.ok)
    store = RunStore(run)
    for r in [*first, *second]:
        for cid in r.call_ids:
            rec = store.load(cid)
            assert (
                rec.item_id == r.item_id
                and json.loads(rec.body)["messages"][0]["content"] == r.item_id
            )


def test_async_break_cancels_workers(tmp_path):
    started = []

    async def fn(x):
        started.append(x)
        await asyncio.sleep(0.05)
        return x

    async def main():
        gen = arun_items({str(i): i for i in range(10)}, fn, run=tmp_path / "run", concurrency=2)
        async for _ in gen:
            break
        await gen.aclose()

    asyncio.run(main())
    # After yielding a result, worker coroutines continue to the next item without waiting for the caller to consume it, so at most 2 × concurrency items have started; the rest have not.
    assert len(started) <= 4 and len(ledger(tmp_path / "run")) <= len(started)
