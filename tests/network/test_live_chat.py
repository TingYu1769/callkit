"""Really calls OpenAI Chat Completions (costs money, about $0.02-0.1): callkit itself, and the official SDK through the recording proxy. Not run by default.

Requires CALLKIT_LIVE_OPENAI_BASE_URL (without /v1), CALLKIT_LIVE_OPENAI_API_KEY; optional
CALLKIT_LIVE_OPENAI_MODEL (default gpt-5.6-sol), CALLKIT_LIVE_PROXY.
  uv run pytest -m network tests/network/test_live_chat.py -q
"""

import os

import httpx
import pytest

from callkit import Client, Endpoint, extend, fork, pending_calls
from callkit.proxy import Proxy

BASE = os.environ.get("CALLKIT_LIVE_OPENAI_BASE_URL", "")
MODEL = os.environ.get("CALLKIT_LIVE_OPENAI_MODEL", "gpt-5.6-sol")
pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not BASE, reason="CALLKIT_LIVE_OPENAI_BASE_URL is not set"),
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市今天的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# Some relay gateways occasionally return 404 (channel issue); callkit does not retry 404, so the test retries it a few times.
TRIES_ON_404 = 6


def endpoint():
    return Endpoint(
        "live-chat",
        BASE,
        "CALLKIT_LIVE_OPENAI_API_KEY",
        api="openai.chat_completions",
        proxy=os.environ.get("CALLKIT_LIVE_PROXY") or None,
    )


def body(text, **extra):
    # From GPT-5.4 onward, reasoning must be disabled in Chat to call tools.
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "reasoning_effort": "none",
        **extra,
    }


def call(c, b, **kw):
    for _ in range(TRIES_ON_404):
        rec = c.call(b, **kw)
        if rec.outcome.ok or rec.response.status != 404:
            break
    return rec


def test_live_tool_call_extend_fork(tmp_path):
    """Call tool, extend with result to get final answer, fork to change output limit and send again."""
    with Client(endpoint(), tmp_path / "run", max_attempts=4) as c:
        first = call(c, body("北京和上海今天天气怎么样？请调用工具查询。", tools=TOOLS))
        assert first.outcome.ok and first.outcome.stop_reason == "tool_calls", first.outcome
        assert len(pending_calls(first)) >= 1 and first.usage and first.usage.output_tokens > 0
        assert (tmp_path / "run" / "calls" / first.id / "response.sse").exists()
        second = call(
            c,
            extend(first, tool_results={i: "晴，25°C，无降水" for i in pending_calls(first)}),
            parent=first,
        )
        assert second.outcome.ok and second.outcome.stop_reason == "stop", second.outcome
        assert second.parent == first.id and second.edits == [] and second.message
        assert second.message["choices"][0]["message"]["content"]
        third = call(
            c, fork(second, edit=lambda b: {**b, "max_completion_tokens": 300}), fork_of=second
        )
        assert third.outcome.ok and third.edits == ["max_completion_tokens"], third.outcome


def test_live_batch_and_resume(tmp_path):
    items = {f"q{i}": body(f"{i} 加 {i} 等于几？只回答数字。") for i in range(1, 4)}
    done: set[str] = set()
    with Client(endpoint(), tmp_path / "run", max_attempts=4) as c:
        # Resume only re-sends failed items; use this to bypass gateway 404s.
        for _ in range(TRIES_ON_404):
            done |= {r.item_id for r in c.batch(items, concurrency=3) if r.outcome.ok and r.item_id}
            if done == set(items):
                break
    assert done == set(items)
    with Client(endpoint(), tmp_path / "run") as c:
        assert list(c.batch(items, concurrency=3)) == []


def test_live_sdk_through_proxy(tmp_path):
    """Official openai SDK sends non-streaming and streaming once each through the recording proxy; both are recorded as successful Chat records."""
    openai = pytest.importorskip("openai")
    proxy = Proxy(BASE, tmp_path / "run", upstream_proxy=os.environ.get("CALLKIT_LIVE_PROXY"))
    proxy.start()
    try:
        sdk = openai.OpenAI(
            base_url=proxy.url + "/v1",
            api_key=os.environ["CALLKIT_LIVE_OPENAI_API_KEY"],
            http_client=httpx.Client(trust_env=False),
            max_retries=0,
        )
        msgs = [{"role": "user", "content": "1 加 1 等于几？只回答数字。"}]
        for stream in (False, True):
            for _ in range(TRIES_ON_404):
                try:
                    if stream:
                        chunks = sdk.chat.completions.create(
                            model=MODEL,
                            messages=msgs,
                            stream=True,
                            stream_options={"include_usage": True},
                        )
                        # Some gateways return a final chunk with only moderation results and no delta
                        text = "".join(
                            ch.choices[0].delta.content or ""
                            for ch in chunks
                            if ch.choices and ch.choices[0].delta
                        )
                    else:
                        text = sdk.chat.completions.create(model=MODEL, messages=msgs).choices[0]
                        text = text.message.content or ""
                    break
                except openai.NotFoundError:
                    continue
            assert "2" in text
    finally:
        proxy.stop()
    recs = [proxy.store.load(r.id) for r in proxy.store.index()]
    good = [r for r in recs if r.outcome.ok]
    assert len(good) == 2, [r.outcome for r in recs]
    assert all(r.api == "openai.chat_completions" and r.usage for r in good)
    assert {(proxy.store.calls / r.id / "response.sse").exists() for r in good} == {True, False}
