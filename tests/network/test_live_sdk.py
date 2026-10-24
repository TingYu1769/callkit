"""Official SDK connected to the recording client, sending real requests (costs about $0.05-0.2). Not run by default.

For OpenAI set CALLKIT_LIVE_OPENAI_BASE_URL (without /v1) and CALLKIT_LIVE_OPENAI_API_KEY (optionally CALLKIT_LIVE_OPENAI_MODEL);
for Anthropic set CALLKIT_LIVE_BASE_URL (without /v1) and CALLKIT_LIVE_API_KEY (optionally CALLKIT_LIVE_MODEL);
to use a proxy set CALLKIT_LIVE_PROXY.
  uv run pytest -m network tests/network/test_live_sdk.py -q
"""

import asyncio
import os

import pytest

from callkit import recording_async_http_client, recording_http_client
from callkit.store import RunStore

OPENAI_BASE = os.environ.get("CALLKIT_LIVE_OPENAI_BASE_URL", "")
OPENAI_MODEL = os.environ.get("CALLKIT_LIVE_OPENAI_MODEL", "gpt-5.6-sol")
CLAUDE_BASE = os.environ.get("CALLKIT_LIVE_BASE_URL", "")
CLAUDE_MODEL = os.environ.get("CALLKIT_LIVE_MODEL", "claude-opus-5-5")
PROXY = os.environ.get("CALLKIT_LIVE_PROXY") or None
MSGS = [{"role": "user", "content": "1 加 1 等于几？只回答数字。"}]
# Some relay gateways occasionally return 404 or 500 (channel issues); let the SDK retry. Each retry records one entry, and tests only check the last successful one.
SDK_RETRIES = 5

pytestmark = pytest.mark.network
needs_openai = pytest.mark.skipif(not OPENAI_BASE, reason="CALLKIT_LIVE_OPENAI_BASE_URL is not set")
needs_claude = pytest.mark.skipif(not CLAUDE_BASE, reason="CALLKIT_LIVE_BASE_URL is not set")


def records(run):
    store = RunStore(run)
    return [store.load(r.id) for r in store.index()]


def openai_client(run, lib):
    import openai

    http = recording_http_client(run, lib=lib, proxy=PROXY, session="live-sdk")
    return openai.OpenAI(
        base_url=OPENAI_BASE + "/v1",
        api_key=os.environ["CALLKIT_LIVE_OPENAI_API_KEY"],
        http_client=http,
        max_retries=SDK_RETRIES,
    )


def retry_404(fn):
    """The OpenAI SDK does not retry 404; gateway 404s are channel issues, so retry here a few times."""
    import openai

    for _ in range(SDK_RETRIES):
        try:
            return fn()
        except openai.NotFoundError:
            continue
    return fn()


@needs_openai
@pytest.mark.parametrize("lib", ["httpx", "httpx2"])
def test_live_openai_chat_and_responses(tmp_path, lib):
    client = openai_client(tmp_path / "run", lib)
    chat = retry_404(lambda: client.chat.completions.create(model=OPENAI_MODEL, messages=MSGS))
    assert "2" in (chat.choices[0].message.content or "")
    stream = retry_404(
        lambda: client.chat.completions.create(
            model=OPENAI_MODEL, messages=MSGS, stream=True, stream_options={"include_usage": True}
        )
    )
    text = "".join(
        c.choices[0].delta.content or "" for c in stream if c.choices and c.choices[0].delta
    )
    assert "2" in text
    resp = retry_404(
        lambda: client.responses.create(model=OPENAI_MODEL, input="1 加 1 等于几？只回答数字。")
    )
    assert "2" in resp.output_text
    ok = [r for r in records(tmp_path / "run") if r.outcome.ok]
    assert [r.api for r in ok] == ["openai.chat_completions"] * 2 + ["openai.responses"]
    assert all(r.source == "sdk" and r.usage and r.tags["session"] == "live-sdk" for r in ok)
    assert ok[0].message and ok[0].message["id"] == chat.id
    assert ok[2].message and ok[2].message["id"] == resp.id


@needs_openai
def test_live_async_openai_concurrent(tmp_path):
    import openai

    http = recording_async_http_client(tmp_path / "run", proxy=PROXY)

    async def main():
        client = openai.AsyncOpenAI(
            base_url=OPENAI_BASE + "/v1",
            api_key=os.environ["CALLKIT_LIVE_OPENAI_API_KEY"],
            http_client=http,
            max_retries=SDK_RETRIES,
        )

        async def ask(i):
            for _ in range(SDK_RETRIES):
                try:
                    msgs = [{"role": "user", "content": f"{i} 加 {i} 等于几？只回答数字。"}]
                    return await client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
                except openai.NotFoundError:
                    continue

        got = await asyncio.gather(*(ask(i) for i in range(1, 4)))
        await client.close()
        return got

    got = asyncio.run(main())
    assert all(g is not None for g in got)
    ok = [r for r in records(tmp_path / "run") if r.outcome.ok]
    assert len(ok) == 3 and len({r.id for r in ok}) == 3


@needs_claude
def test_live_anthropic_httpx2(tmp_path):
    import anthropic

    http = recording_http_client(tmp_path / "run", lib="httpx2", proxy=PROXY)
    client = anthropic.Anthropic(
        base_url=CLAUDE_BASE, api_key=os.environ["CALLKIT_LIVE_API_KEY"], http_client=http
    )
    msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=50, messages=MSGS)
    assert any("2" in getattr(b, "text", "") for b in msg.content)
    with client.messages.stream(model=CLAUDE_MODEL, max_tokens=50, messages=MSGS) as s:
        text = "".join(s.text_stream)
    assert "2" in text
    ok = [r for r in records(tmp_path / "run") if r.outcome.ok]
    assert len(ok) == 2 and all(r.api == "anthropic.messages" and r.usage for r in ok)
    assert ok[0].message and ok[0].message["id"] == msg.id
    assert (tmp_path / "run" / "calls" / ok[1].id / "response.sse").exists()
