"""Actually calls the DeepSeek official API (costs a few cents RMB). Not run by default.

Requires CALLKIT_LIVE_DEEPSEEK_API_KEY; optional CALLKIT_LIVE_DEEPSEEK_MODEL (default deepseek-flash). Direct connection, no proxy.
  uv run pytest -m network tests/network/test_live_deepseek.py -q
"""

import os

import pytest

from callkit import Client, Endpoint, Price, extend, pending_calls, recording_http_client
from callkit.store import RunStore

BASE = "https://api.deepseek.com"
KEY_ENV = "CALLKIT_LIVE_DEEPSEEK_API_KEY"
MODEL = os.environ.get("CALLKIT_LIVE_DEEPSEEK_MODEL", "deepseek-flash")
PRICE = {MODEL: Price(input=0.27, output=1.1, cache_read=0.07, cache_write=0.27)}  # sample price
ASK = [{"role": "user", "content": "北京今天天气怎么样？请调用工具查询，然后用一句话回答。"}]

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.environ.get(KEY_ENV), reason=f"{KEY_ENV} is not set"),
]


def test_live_chat_and_anthropic_round_trips(tmp_path):
    """Test both interfaces through a tool round trip; the Anthropic-compatible endpoint URL has an /anthropic prefix, and follow-up turns must not duplicate the prefix."""
    chat_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    ant_tool = {
        "name": "get_weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    cases = [
        (Endpoint("ds", BASE, KEY_ENV, api="openai.chat_completions"), {"tools": [chat_tool]}),
        (
            Endpoint("ds-ant", BASE + "/anthropic", KEY_ENV),
            {"tools": [ant_tool], "max_tokens": 800},
        ),
    ]
    for ep, extra in cases:
        with Client(ep, tmp_path / ep.name, prices=PRICE) as c:
            first = c.call({"model": MODEL, "messages": ASK, **extra})
            assert first.outcome.ok and pending_calls(first), first.outcome
            second = c.call(
                extend(first, tool_results={i: "晴，25°C" for i in pending_calls(first)}),
                parent=first,
            )
            assert second.outcome.ok and second.request.url == first.request.url, second.outcome
            assert second.cost_usd and second.cost_usd > 0


def test_live_sdk_prices_and_cap(tmp_path):
    openai = pytest.importorskip("openai")
    http = recording_http_client(tmp_path / "run", prices=PRICE, cap_usd=1e-9)
    client = openai.OpenAI(base_url=BASE, api_key=os.environ[KEY_ENV], http_client=http)
    msgs = [{"role": "user", "content": "1 加 1 等于几？只回答数字。"}]
    client.chat.completions.create(model=MODEL, messages=msgs)
    with pytest.raises(openai.APIStatusError) as e:
        client.chat.completions.create(model=MODEL, messages=msgs)
    assert e.value.status_code == 402
    store = RunStore(tmp_path / "run")
    (row,) = store.index()
    assert row.ok and row.cost_usd and row.cost_usd > 0


def test_live_sdk_replay(tmp_path):
    """Two official SDK runs send byte-for-byte identical request bodies; the second fully hits the replay cache and returns the same answers as the first."""
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    msgs = [{"role": "user", "content": "1 加 1 等于几？只回答数字。"}]

    def once(run, replay_from=None):
        kw = {"replay_from": replay_from}
        oc = openai.OpenAI(
            base_url=BASE,
            api_key=os.environ[KEY_ENV],
            http_client=recording_http_client(run, **kw),
        )
        ac = anthropic.Anthropic(
            base_url=BASE + "/anthropic",
            api_key=os.environ[KEY_ENV],
            http_client=recording_http_client(run, lib="httpx2", **kw),
        )
        a = oc.chat.completions.create(model=MODEL, messages=msgs).choices[0].message.content
        stream = oc.chat.completions.create(model=MODEL, messages=msgs, stream=True)
        b = "".join(
            c.choices[0].delta.content or "" for c in stream if c.choices and c.choices[0].delta
        )
        c = ac.messages.create(model=MODEL, max_tokens=200, messages=msgs).content[-1].text
        return [a, b, c]

    first = once(tmp_path / "first")
    second = once(tmp_path / "second", replay_from=tmp_path / "first")
    assert second == first
    store = RunStore(tmp_path / "second")
    tags = [store.load(r.id).tags.get("replay_of", "") for r in store.index()]
    assert len(tags) == 3 and all(t.endswith(f"c_00000{i}") for i, t in enumerate(tags, 1)), tags


def test_live_scheduler_sync_and_async(tmp_path):
    """Scheduler wraps the official SDK: records are attached to their items, and reruns skip all."""
    import asyncio

    from callkit import arun_items, recording_async_http_client, run_items

    openai = pytest.importorskip("openai")
    items = {f"q{i}": f"{i} 加 {i} 等于几？只回答数字。" for i in range(1, 4)}
    run = tmp_path / "run"
    client = openai.OpenAI(
        base_url=BASE,
        api_key=os.environ[KEY_ENV],
        http_client=recording_http_client(run, prices=PRICE),
    )

    def ask(text):
        msgs = [{"role": "user", "content": text}]
        return client.chat.completions.create(model=MODEL, messages=msgs).choices[0].message.content

    got = {r.item_id: r for r in run_items(items, ask, run=run, concurrency=3)}
    assert all(r.ok and r.value and len(r.call_ids) == 1 for r in got.values()), got
    assert list(run_items(items, ask, run=run)) == []
    store = RunStore(run)
    assert {store.load(r.call_ids[0]).item_id for r in got.values()} == set(items)

    arun = tmp_path / "arun"
    aclient = openai.AsyncOpenAI(
        base_url=BASE,
        api_key=os.environ[KEY_ENV],
        http_client=recording_async_http_client(arun),
    )

    async def aask(text):
        msgs = [{"role": "user", "content": text}]
        resp = await aclient.chat.completions.create(model=MODEL, messages=msgs)
        return resp.choices[0].message.content

    async def main():
        return [r async for r in arun_items(items, aask, run=arun, concurrency=3)]

    results = asyncio.run(main())
    assert len(results) == 3 and all(r.ok and len(r.call_ids) == 1 for r in results)
    astore = RunStore(arun)
    assert {astore.load(r.call_ids[0]).item_id for r in results} == set(items)
