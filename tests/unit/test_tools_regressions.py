"""Regression tests for pricing, replay, scheduler, and CLI: each test corresponds to a fixed issue."""

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
    browse,
    fork,
    recording_async_http_client,
    recording_http_client,
    run_items,
    transport,
)
from callkit.adapters.anthropic import AnthropicMessages
from callkit.adapters.base import Judgement
from callkit.cli import _print_children, main
from callkit.client import _total_cost
from callkit.context import CURRENT_ITEM, ItemContext
from callkit.record import Attempt, Outcome, Usage
from callkit.recorder import Recorder
from callkit.store import RunStore
from tests.fakes import DATA, Server, good_stream, sse_response

CHAT_JSON = (DATA / "chat_tool_calls.json").read_bytes()
CHAT_SSE = (DATA / "chat_text.sse").read_bytes()
NO_USAGE_SSE = b"\n".join(
    l for l in CHAT_SSE.split(b"\n") if b'"usage":{"completion_tokens"' not in l
)
MSGS = [{"role": "user", "content": "hi"}]
PRICE = Price(input=1.25, output=10, cache_read=0.125, cache_write=1.5625)


def json_reply(request):
    return httpx.Response(200, headers={"content-type": "application/json"}, content=CHAT_JSON)


def sdk(run, transport_, **kw):
    http = recording_http_client(run, transport=transport_, **kw)
    return openai.OpenAI(
        base_url="http://gw.test/v1", api_key="sk", http_client=http, max_retries=0
    )


def records(run):
    store = RunStore(run)
    return [store.load(r.id) for r in store.index()]


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("CALLKIT_TEST_KEY", "sk")
    monkeypatch.setattr(transport.time, "sleep", lambda s: None)
    s = Server()
    monkeypatch.setattr(transport, "open_client", lambda endpoint: s.client())
    return s


def test_anthropic_without_usage_is_unknown():
    a = AnthropicMessages()
    msg = a.assemble(
        __import__("callkit.sse", fromlist=["parse"]).parse(
            (0.0, line)
            for line in (DATA / "stream_tool_use.sse").read_text(encoding="utf-8").split("\n")
        )
    )
    assert msg is not None
    body = {k: v for k, v in msg.items() if k != "usage"}
    _, j, usage = a.from_json(200, "done", json.dumps(body))
    assert j.outcome.ok and usage is None
    start = {"type": "message_start", "message": {"id": "m", "content": []}}
    from callkit.sse import SseEvent

    assert a.usage([SseEvent("message_start", json.dumps(start), 0.0)]) is None


def test_client_uses_prices_already_in_run_dir(server, tmp_path):
    ep = Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY")
    Client(ep, tmp_path / "run", prices={"claude-sonnet-5-5": PRICE})
    server.replies.append(sse_response(good_stream()))
    body = {"model": "claude-sonnet-5-5", "max_tokens": 10, "messages": MSGS}
    # No price passed again and no missing-price error configured.
    rec = Client(ep, tmp_path / "run", cap_usd=5).call(body)
    assert rec.cost_usd and rec.cost_usd > 0


def test_concurrent_add_prices_keep_both(tmp_path, monkeypatch):
    r = Recorder(tmp_path / "run")
    original = r.store.merge_prices

    def slow(prices):
        table = original(prices)
        if "a" in prices:
            # A has merged but not yet written back to the in-memory table when B arrives.
            time.sleep(0.2)
        return table

    monkeypatch.setattr(r.store, "merge_prices", slow)
    ta = threading.Thread(target=r.add_prices, args=({"a": PRICE},))
    tb = threading.Thread(target=r.add_prices, args=({"b": PRICE},))
    ta.start()
    time.sleep(0.05)
    tb.start()
    ta.join()
    tb.join()
    assert set(r.prices) == {"a", "b"}


def test_known_cost_of_failed_attempts_kept():
    ok = Judgement(Outcome(True, "done", "completed", "stop"), False, "completed")
    u = Usage(input_tokens=1_000_000)
    first = Attempt(1, 529, "done", "failed", "retry", "overloaded", u, 1.25)
    last = Attempt(2, 200, "done", "completed", "stop", "completed", None, 0.0)
    with pytest.warns(RuntimeWarning, match="include_usage"):
        assert _total_cost(PRICE, [first, last], ok) == 1.25 and last.cost_usd is None
    assert ok.outcome.flags == ["usage_missing"]
    alone = Judgement(Outcome(True, "done", "completed", "stop"), False, "completed")
    only = Attempt(1, 200, "done", "completed", "stop", "completed", None, 0.0)
    with pytest.warns(RuntimeWarning):
        assert _total_cost(PRICE, [only], alone) is None


def test_usage_missing_flag_without_price_and_on_sdk_replay(server, tmp_path):
    server.replies.append(sse_response(NO_USAGE_SSE))
    ep = Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY", api="openai.chat_completions")
    with pytest.warns(RuntimeWarning, match="include_usage"):
        rec = Client(ep, tmp_path / "c").call({"model": "m", "stream": True, "messages": MSGS})
    assert rec.cost_usd is None and rec.outcome.flags == ["usage_missing"]

    def sse(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=NO_USAGE_SSE
        )

    with pytest.warns(RuntimeWarning):
        for _ in sdk(tmp_path / "old", httpx.MockTransport(sse)).chat.completions.create(
            model="m", messages=MSGS, stream=True
        ):
            pass

    def offline(request):
        raise AssertionError("must not hit the network")

    client = sdk(tmp_path / "new", httpx.MockTransport(offline), replay_from=tmp_path / "old")
    with pytest.warns(RuntimeWarning):
        for _ in client.chat.completions.create(model="m", messages=MSGS, stream=True):
            pass
    (replayed,) = records(tmp_path / "new")
    assert "usage_missing" in replayed.outcome.flags and replayed.cost_usd == 0.0


def test_async_scheduler_error_surfaces(tmp_path, monkeypatch):
    import callkit.scheduler as sched

    def broken(path, obj):
        raise OSError("磁盘满了")

    monkeypatch.setattr(sched, "append_line", broken)

    async def fn(x):
        return x

    async def main_():
        return [
            r async for r in arun_items({"a": 1, "b": 2}, fn, run=tmp_path / "run", concurrency=1)
        ]

    with pytest.raises(OSError, match="磁盘满了"):
        asyncio.run(main_())


def test_cancel_inside_fn_is_item_failure(tmp_path):
    async def fn(x):
        if x == "bad":
            task = asyncio.create_task(asyncio.sleep(10))
            task.cancel()
            await task  # The task awaited inside the function was cancelled.
        return x

    async def main_():
        items = {"bad": "bad", "ok": "ok"}
        return [r async for r in arun_items(items, fn, run=tmp_path / "run", concurrency=1)]

    got = {r.item_id: r for r in asyncio.run(main_())}
    assert not got["bad"].ok and got["bad"].error == "CancelledError" and got["ok"].ok


def test_ledger_unicode(tmp_path):
    from callkit.scheduler import _done

    ledger = tmp_path / "items.jsonl"
    good = json.dumps({"item_id": "a\u2028b", "repeat": 0, "ok": True}, ensure_ascii=False).encode()
    half = json.dumps({"item_id": "中", "repeat": 0, "ok": True}, ensure_ascii=False).encode()[:-5]
    # The last line was written halfway, splitting a Chinese character.
    ledger.write_bytes(good + b"\n" + half[:-1])
    with pytest.warns(RuntimeWarning, match="unreadable line"):
        assert _done(ledger) == {("a\u2028b", 0)}


def test_explicit_same_item_id_collects_call_ids(server, tmp_path):
    server.replies += [sse_response(good_stream())]
    run = tmp_path / "run"
    body = {"model": "claude-sonnet-5-5", "max_tokens": 10, "messages": MSGS}
    with Client(Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY"), run) as c:
        (r,) = run_items({"a": body}, lambda b: c.call(b, item_id="a", repeat=0), run=run)
    assert r.call_ids == [r.value.id]


class Chunks(httpx.SyncByteStream):
    """Chunked response stream, not read ahead, simulating a real network stream."""

    def __iter__(self):
        for i in range(0, len(CHAT_SSE), 97):
            yield CHAT_SSE[i : i + 97]


def test_stream_belongs_to_item_that_sent_it(tmp_path):
    def sse(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Chunks())

    client = sdk(tmp_path / "run", httpx.MockTransport(sse))
    a, b = ItemContext("A", 0), ItemContext("B", 0)
    token = CURRENT_ITEM.set(a)
    stream = client.chat.completions.create(model="m", messages=MSGS, stream=True)
    CURRENT_ITEM.reset(token)
    token = CURRENT_ITEM.set(b)
    for _ in stream:
        pass
    CURRENT_ITEM.reset(token)
    (rec,) = records(tmp_path / "run")
    assert rec.item_id == "A" and a.call_ids == [rec.id] and b.call_ids == []


def test_cancel_while_reading_error_body_is_recorded(tmp_path):
    class SlowError(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error": {"code": "inv'
            await asyncio.sleep(10)

    def reply(request):
        return httpx.Response(400, headers={"content-type": "application/json"}, stream=SlowError())

    http = recording_async_http_client(tmp_path / "run", transport=httpx.MockTransport(reply))

    async def main_():
        client = openai.AsyncOpenAI(
            base_url="http://gw.test/v1", api_key="sk", http_client=http, max_retries=0
        )
        task = asyncio.create_task(
            client.chat.completions.create(model="m", messages=MSGS, stream=True)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main_())
    (rec,) = records(tmp_path / "run")
    assert rec.response.status == 400 and rec.outcome.transport_end == "client_gone"


def test_cross_run_fork_keeps_tree(server, tmp_path):
    server.replies += [sse_response(good_stream()), sse_response(good_stream())]
    ep = Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY")
    body = {"model": "claude-sonnet-5-5", "max_tokens": 10, "messages": MSGS}
    with Client(ep, tmp_path / "a") as c:
        base = c.call(body)
    with Client(ep, tmp_path / "b") as c:
        alt = c.call(fork(base), fork_of=base)
    assert (
        alt.id == base.id == "c_000001" and alt.fork_of == f"{(tmp_path / 'a').resolve()}:c_000001"
    )
    forest = browse.tree([browse.row_of(r) for r in records(tmp_path / "b")])
    assert [n.row.id for roots in forest.values() for n in roots] == ["c_000001"]
    looped = browse.row_of(alt)
    looped.parent = looped.id  # Pointing to itself must not make the tree disappear.
    assert sum(len(v) for v in browse.tree([looped]).values()) == 1


def test_cli_survives_bad_records(server, tmp_path, capsys):
    server.replies += [sse_response(good_stream()) for _ in range(3)]
    run = tmp_path / "run"
    with Client(Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY"), run) as c:
        c.call({"model": "m", "max_tokens": 1, "messages": MSGS})
        c.call(b'{"model": "m", "stream": true, "messages": "not a list", "tools": [1, "x"]}')
        c.call({"model": "m", "max_tokens": 1, "messages": MSGS})
    (run / "calls" / "c_000003" / "meta.json").unlink()
    with pytest.warns(RuntimeWarning, match="unreadable record c_000003"):
        assert main(["ls", str(run)]) == 0
    assert "c_000002" in capsys.readouterr().out
    assert main(["show", str(run), "c_000002"]) == 0
    assert "last user input: not a list" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="cannot read c_000003"):
        main(["show", str(run), "c_000003"])
    (run / "calls" / "c_000001" / "request.json").write_bytes(b"{broken")
    out = tmp_path / "x.jsonl"
    with pytest.warns(RuntimeWarning):
        main(["export", str(run), "-o", str(out)])
    first = json.loads(out.read_text().splitlines()[0])
    assert first["request"] is None and first["request_text"] == "{broken"


def test_ls_tells_missing_usage_from_missing_price(server, tmp_path, capsys):
    server.replies += [
        sse_response(NO_USAGE_SSE),
        sse_response((DATA / "chat_tool_calls.sse").read_bytes()),
    ]
    ep = Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY", api="openai.chat_completions")
    with Client(ep, tmp_path / "run", prices={"m": PRICE}) as c:
        with pytest.warns(RuntimeWarning):
            c.call(
                {"model": "m", "stream": True, "messages": MSGS}
            )  # Explicit stream without include_usage.
        c.call({"model": "m", "messages": MSGS})
    main(["ls", str(tmp_path / "run")])
    assert "(not counted: 1 without usage)" in capsys.readouterr().out


def test_read_only_commands_do_not_create_dirs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text('{"format": "callkit.run/1"}')
    assert main(["ls", str(run)]) == 0
    assert not (run / "calls").exists()


def test_long_chain_prints_without_recursion(capsys):
    rows = []
    for i in range(1500):
        r = browse.Row(
            f"c_{i:06d}",
            "",
            "chat",
            "self",
            "m",
            True,
            "stop",
            "",
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
        )
        r.parent = f"c_{i - 1:06d}" if i else None
        rows.append(r)
    (root,) = browse.tree(rows)[None]
    _print_children(root, {r.id: r.id for r in rows}, "")
    assert capsys.readouterr().out.count("\n") == 1499
