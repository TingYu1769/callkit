"""SDK integration: real openai (3.37) and anthropic (2.2) SDKs over MockTransport (both httpx and httpx2 tested), no network access."""

import asyncio
import gzip
import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from callkit import recording_async_http_client, recording_http_client
from callkit.adapters.anthropic import AnthropicMessages
from callkit.errors import CallkitError
from callkit.recorder import Recorder
from callkit.sdk import _Tap
from callkit.sse import parse
from callkit.store import RunStore
from tests.fakes import DATA

CHAT_JSON = (DATA / "chat_tool_calls.json").read_bytes()
CHAT_SSE = (DATA / "chat_text.sse").read_bytes()
MSGS = [{"role": "user", "content": "北京和上海今天天气怎么样？"}]


def chunked(mod, data: bytes, size: int = 97):
    """Yield data in chunks as a response stream (without reading it fully upfront), readable by both sync and async."""

    class Chunks(mod.SyncByteStream, mod.AsyncByteStream):
        def __iter__(self):
            for i in range(0, len(data), size):
                yield data[i : i + size]

        async def __aiter__(self):
            for i in range(0, len(data), size):
                yield data[i : i + size]

    return Chunks()


class Gateway:
    """Fake gateway: replays replies in order; each item is (status, content type, bytes) or an exception. Requests received are stored in seen."""

    def __init__(self, mod, *replies, headers=None):
        self.mod, self.replies, self.seen = mod, list(replies), []
        self.headers = headers or {}

    def __call__(self, request):
        self.seen.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, ctype, data = reply
        headers = {"content-type": ctype, **self.headers}
        return self.mod.Response(status, headers=headers, stream=chunked(self.mod, data))

    def transport(self):
        return self.mod.MockTransport(self)


def oai(gw, run, **kw):
    http = recording_http_client(run, lib=gw.mod.__name__, transport=gw.transport(), **kw)
    return openai.OpenAI(
        base_url="http://gw.test/v1", api_key="sk-test", http_client=http, max_retries=0
    )


def records(run):
    store = RunStore(run)
    return [store.load(r.id) for r in store.index()]


@pytest.mark.parametrize("mod", [httpx, httpx2])
def test_openai_chat_json_recorded(tmp_path, mod):
    gw = Gateway(mod, (200, "application/json", CHAT_JSON))
    resp = oai(gw, tmp_path / "run", tags={"exp": "E1"}).chat.completions.create(
        model="gpt-5.6-sol", messages=MSGS
    )
    assert resp.choices[0].finish_reason == "tool_calls"
    assert gw.seen[0].headers["accept-encoding"] == "identity"
    (rec,) = records(tmp_path / "run")
    assert rec.source == "sdk" and rec.api == "openai.chat_completions" and rec.outcome.ok
    assert rec.message == json.loads(CHAT_JSON) and rec.usage and rec.usage.input_tokens == 139
    assert rec.tags == {"retry_count": "0", "exp": "E1"}
    assert rec.request.headers["authorization"] == "<redacted>"
    # records the headers the SDK originally sent
    assert rec.request.headers["accept-encoding"] != "identity"
    assert json.loads(rec.body or b"") == {"messages": MSGS, "model": "gpt-5.6-sol"}


def test_openai_stream_recorded_with_timing(tmp_path):
    gw = Gateway(httpx, (200, "text/event-stream", CHAT_SSE))
    stream = oai(gw, tmp_path / "run").chat.completions.create(
        model="gpt-5.6-sol", messages=MSGS, stream=True, stream_options={"include_usage": True}
    )
    text = "".join(
        c.choices[0].delta.content or "" for c in stream if c.choices and c.choices[0].delta
    )
    (rec,) = records(tmp_path / "run")
    assert (
        rec.outcome.ok and rec.message and rec.message["choices"][0]["message"]["content"] == text
    )
    assert (tmp_path / "run" / "calls" / rec.id / "response.sse").read_bytes() == CHAT_SSE
    assert rec.timing.first_byte_ms is not None


def test_early_close_is_client_gone(tmp_path):
    gw = Gateway(httpx, (200, "text/event-stream", CHAT_SSE))
    stream = oai(gw, tmp_path / "run").chat.completions.create(
        model="gpt-5.6-sol", messages=MSGS, stream=True
    )
    next(iter(stream))
    stream.close()
    (rec,) = records(tmp_path / "run")
    assert not rec.outcome.ok and rec.outcome.transport_end == "client_gone"


def test_sdk_retries_each_recorded(tmp_path):
    err = json.dumps({"error": {"code": "server_error", "message": "x"}}).encode()
    gw = Gateway(
        httpx,
        (500, "application/json", err),
        (200, "application/json", CHAT_JSON),
        headers={"x-should-retry": "true", "retry-after-ms": "1"},
    )
    oai(gw, tmp_path / "run").with_options(max_retries=1).chat.completions.create(
        model="gpt-5.6-sol", messages=MSGS
    )
    first, second = records(tmp_path / "run")
    assert (
        not first.outcome.ok and first.response.status == 500 and first.tags["retry_count"] == "0"
    )
    assert second.outcome.ok and second.tags["retry_count"] == "1"


def test_connect_error_recorded_then_raised(tmp_path):
    gw = Gateway(httpx, httpx.ConnectError("refused"))
    with pytest.raises(openai.APIConnectionError):
        oai(gw, tmp_path / "run").chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    (rec,) = records(tmp_path / "run")
    assert not rec.outcome.ok and rec.outcome.transport_end == "connect_error"
    assert rec.response.status is None and "refused" in (rec.outcome.error or "")


def test_gzip_still_parsed(tmp_path):
    gw = Gateway(
        httpx,
        (200, "application/json", gzip.compress(CHAT_JSON)),
        headers={"content-encoding": "gzip"},
    )
    resp = oai(gw, tmp_path / "run").chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    assert resp.id == json.loads(CHAT_JSON)["id"]
    (rec,) = records(tmp_path / "run")
    assert rec.outcome.ok and rec.message == json.loads(CHAT_JSON)


def test_other_paths_pass_through_untouched(tmp_path):
    models = json.dumps({"object": "list", "data": []}).encode()
    gw = Gateway(httpx, (200, "application/json", models))
    oai(gw, tmp_path / "run").models.list()
    assert gw.seen[0].headers.get("accept-encoding") != "identity"
    assert RunStore(tmp_path / "run").index() == []


def test_run_none_is_plain_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gw = Gateway(httpx, (200, "application/json", CHAT_JSON))
    http = recording_http_client(None, transport=gw.transport())
    client = openai.OpenAI(base_url="http://gw.test/v1", api_key="sk-test", http_client=http)
    client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    assert gw.seen[0].headers.get("accept-encoding") != "identity"
    assert list(tmp_path.iterdir()) == [] and http.trust_env is False


def test_parent_inferred_within_session(tmp_path):
    gw = Gateway(httpx, (200, "application/json", CHAT_JSON), (200, "application/json", CHAT_JSON))
    client = oai(gw, tmp_path / "run", session="exp-1")
    client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    raw = json.loads(CHAT_JSON)["choices"][0]["message"]
    tools = [
        {"role": "tool", "tool_call_id": tc["id"], "content": "晴"} for tc in raw["tool_calls"]
    ]
    client.chat.completions.create(model="gpt-5.6-sol", messages=[*MSGS, raw, *tools])
    first, second = records(tmp_path / "run")
    assert first.tags["session"] == "exp-1" and second.parent == first.id


def test_anthropic_needs_httpx2_and_is_recorded(tmp_path):
    msg = AnthropicMessages().assemble(
        parse(
            (0.0, line)
            for line in (DATA / "stream_tool_use.sse").read_text(encoding="utf-8").split("\n")
        )
    )
    gw = Gateway(httpx2, (200, "application/json", json.dumps(msg).encode()))
    with pytest.raises(TypeError):
        anthropic.Anthropic(api_key="sk", http_client=recording_http_client(tmp_path / "x"))
    http = recording_http_client(tmp_path / "run", lib="httpx2", transport=gw.transport())
    client = anthropic.Anthropic(
        base_url="http://gw.test", api_key="sk-test", http_client=http, max_retries=0
    )
    out = client.messages.create(model="claude-opus-5-5", max_tokens=100, messages=MSGS)
    assert out.stop_reason == msg["stop_reason"] and gw.seen[0].url.path == "/v1/messages"
    (rec,) = records(tmp_path / "run")
    assert rec.api == "anthropic.messages" and rec.outcome.ok and rec.message == msg
    # anthropic SDK 2.2 uses Authorization: Bearer for api_key and does not send x-api-key
    assert rec.request.headers["authorization"] == "<redacted>" and rec.usage


@pytest.mark.parametrize("mod", [httpx, httpx2])
def test_async_openai_stream_and_json(tmp_path, mod):
    gw = Gateway(mod, (200, "text/event-stream", CHAT_SSE), (200, "application/json", CHAT_JSON))
    http = recording_async_http_client(tmp_path / "run", lib=mod.__name__, transport=gw.transport())

    async def main():
        client = openai.AsyncOpenAI(
            base_url="http://gw.test/v1", api_key="sk-test", http_client=http, max_retries=0
        )
        stream = await client.chat.completions.create(
            model="gpt-5.6-sol", messages=MSGS, stream=True
        )
        async for _ in stream:
            pass
        await client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
        await client.close()

    asyncio.run(main())
    recs = records(tmp_path / "run")
    assert [r.outcome.ok for r in recs] == [True, True] and all(r.source == "sdk" for r in recs)
    assert (tmp_path / "run" / "calls" / recs[0].id / "response.sse").exists()


def test_shared_recorder_no_id_clash(tmp_path):
    gw = Gateway(httpx, *[(200, "application/json", CHAT_JSON)] * 2)
    a, b = oai(gw, tmp_path / "run"), oai(gw, tmp_path / "run")
    a.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    b.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    assert [r.id for r in RunStore(tmp_path / "run").index()] == ["c_000001", "c_000002"]


def test_bad_arguments(tmp_path):
    with pytest.raises(CallkitError, match="either proxy or transport"):
        recording_http_client(
            tmp_path / "run", proxy="http://p:1", transport=httpx.MockTransport(lambda r: None)
        )
    with pytest.raises(CallkitError, match="lib"):
        recording_async_http_client(tmp_path / "run", lib="requests")


def test_opaque_encoding_not_parsed(tmp_path):
    from callkit.recorder import Exchange

    ex = Exchange("POST", "http://gw/v1/chat/completions", {}, b"{}")
    tap = _Tap(Recorder(tmp_path / "run"), ex, None, {})
    tap.start(200, {"content-type": "application/json", "content-encoding": "br"})
    tap.feed(b"\x8b\x00")
    tap.finish()
    assert ex.chunks == [] and "br" in (ex.error or "")


def test_preread_response_recorded(tmp_path):
    """When the inner transport returns the full body directly (response already read and closed), it must still be recorded."""

    def reply(request):
        return httpx.Response(200, headers={"content-type": "application/json"}, content=CHAT_JSON)

    http = recording_http_client(tmp_path / "run", transport=httpx.MockTransport(reply))
    assert http.follow_redirects is True and http.trust_env is False
    client = openai.OpenAI(base_url="http://gw.test/v1", api_key="sk-test", http_client=http)
    client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    (rec,) = records(tmp_path / "run")
    assert rec.outcome.ok and rec.message == json.loads(CHAT_JSON)


def test_tap_finishes_once(tmp_path):
    from callkit.recorder import Exchange

    r = Recorder(tmp_path / "run")
    tap = _Tap(r, Exchange("POST", "http://gw/v1/chat/completions", {}, b"{}"), None, {})
    tap.start(200, {"content-type": "application/json"})
    tap.feed(CHAT_JSON)
    tap.finish()
    # repeated close (for example the caller closes it again) no longer writes to disk
    tap.finish(complete=False)
    assert len(r.store.index()) == 1


def test_recording_failure_does_not_break_sdk_call(tmp_path, monkeypatch):
    gw = Gateway(httpx, (200, "application/json", CHAT_JSON))
    client = oai(gw, tmp_path / "run")

    def boom(*a, **k):
        raise OSError("磁盘满了")

    monkeypatch.setattr(Recorder.shared(tmp_path / "run").store, "write", boom)
    with pytest.warns(RuntimeWarning, match="callkit failed to record"):
        resp = client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    assert resp.choices[0].finish_reason == "tool_calls"


def test_cancel_before_response_recorded(tmp_path):
    """When an async request is cancelled before a response is received: record a client_gone entry, and the cancellation still propagates to the caller."""

    async def slow(request):
        await asyncio.sleep(10)

    http = recording_async_http_client(tmp_path / "run", transport=httpx.MockTransport(slow))

    async def main():
        client = openai.AsyncOpenAI(
            base_url="http://gw.test/v1", api_key="sk-test", http_client=http, max_retries=0
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS), 0.2
            )

    asyncio.run(main())
    (rec,) = records(tmp_path / "run")
    assert not rec.outcome.ok and rec.outcome.transport_end == "client_gone"


# ---- regression tests ----


class TimeoutMidBody(httpx.SyncByteStream):
    """Error body times out halfway through reading."""

    def __iter__(self):
        yield b'{"error": {"code": "inval'
        raise httpx.ReadTimeout("read timed out")


def test_stream_error_body_timeout_still_recorded(tmp_path):
    def reply(request):
        return httpx.Response(
            400, headers={"content-type": "application/json"}, stream=TimeoutMidBody()
        )

    http = recording_http_client(tmp_path / "run", transport=httpx.MockTransport(reply))
    client = openai.OpenAI(
        base_url="http://gw.test/v1", api_key="sk-test", http_client=http, max_retries=0
    )
    # whatever the SDK raises does not matter, what matters is the record
    with pytest.raises(Exception):  # noqa: B017
        client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS, stream=True)
    (rec,) = records(tmp_path / "run")
    assert rec.response.status == 400 and rec.outcome.transport_end == "timeout"
    assert "ReadTimeout" in (rec.outcome.error or "")


def test_unmapped_exception_is_not_client_gone(tmp_path):
    import ssl

    gw = Gateway(httpx, ssl.SSLError("handshake failed"))
    with pytest.raises(openai.APIConnectionError):
        oai(gw, tmp_path / "run").chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    (rec,) = records(tmp_path / "run")
    assert rec.outcome.transport_end == "eof" and "SSLError" in (rec.outcome.error or "")


def test_raw_deflate_parsed(tmp_path):
    import zlib

    z = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = z.compress(CHAT_JSON) + z.flush()
    gw = Gateway(httpx, (200, "application/json", raw), headers={"content-encoding": "deflate"})
    resp = oai(gw, tmp_path / "run").chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    assert resp.id == json.loads(CHAT_JSON)["id"]
    (rec,) = records(tmp_path / "run")
    assert rec.outcome.ok and rec.message == json.loads(CHAT_JSON)


def test_mounts_and_trust_env_refused(tmp_path):
    with pytest.raises(CallkitError, match="mounts"):
        recording_http_client(tmp_path / "run", mounts={"all://": httpx.HTTPTransport()})
    with pytest.raises(CallkitError, match="trust_env"):
        recording_async_http_client(tmp_path / "run", trust_env=True)
    assert recording_http_client(tmp_path / "run", trust_env=False).trust_env is False


def test_transport_kwargs_reach_inner_transport(tmp_path):
    import ssl

    for run in (None, tmp_path / "run"):
        c = recording_http_client(run, verify=False, limits=httpx.Limits(max_connections=1))
        inner = c._transport if run is None else c._transport._inner
        assert inner._pool._max_connections == 1
        assert inner._pool._ssl_context.verify_mode == ssl.CERT_NONE
    with pytest.raises(CallkitError, match="transport"):
        recording_http_client(
            tmp_path / "run", transport=httpx.MockTransport(lambda r: None), verify=False
        )


# ---- pricing and caps ----

PRICE = {
    "gpt-5.6-sol": __import__("callkit").Price(
        input=1.25, output=10, cache_read=0.125, cache_write=1.5625
    )
}


def test_sdk_records_cost(tmp_path):
    from callkit.pricing import cost_usd

    gw = Gateway(httpx, (200, "application/json", CHAT_JSON))
    oai(gw, tmp_path / "run", prices=PRICE).chat.completions.create(
        model="gpt-5.6-sol", messages=MSGS
    )
    (rec,) = records(tmp_path / "run")
    assert (
        rec.usage and rec.cost_usd == cost_usd(rec.usage, PRICE["gpt-5.6-sol"]) and rec.cost_usd > 0
    )
    assert rec.attempts[0].cost_usd == rec.cost_usd
    assert json.loads((tmp_path / "run" / "prices.json").read_text())["gpt-5.6-sol"]["output"] == 10


@pytest.mark.parametrize("stream", [False, True])
def test_cap_reached_gives_402_without_retry(tmp_path, stream, monkeypatch):
    refusals = []
    original = Recorder.refusal

    def counting(self, body, cap):
        got = original(self, body, cap)
        refusals.append(got is not None)
        return got

    monkeypatch.setattr(Recorder, "refusal", counting)
    gw = Gateway(httpx, (200, "application/json", CHAT_JSON))
    client = oai(gw, tmp_path / "run", prices=PRICE, cap_usd=1e-9).with_options(max_retries=2)
    # this one has not hit the cap yet
    client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    with pytest.raises(openai.APIStatusError) as e:
        client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS, stream=stream)
    assert e.value.status_code == 402 and "reached the cap" in e.value.message
    assert e.value.response.headers["x-should-retry"] == "false"
    assert refusals == [False, True]  # SDK did not retry the refused request
    # not sent, no retry, not recorded
    assert len(gw.seen) == 1 and len(records(tmp_path / "run")) == 1


def test_missing_price_with_cap_refused(tmp_path):
    gw = Gateway(httpx, (200, "application/json", CHAT_JSON))
    client = oai(gw, tmp_path / "run", prices=PRICE, cap_usd=5)
    with pytest.raises(openai.APIStatusError) as e:
        client.chat.completions.create(model="other-model", messages=MSGS)
    assert e.value.status_code == 402 and "other-model" in e.value.message and gw.seen == []


def test_cap_refusal_anthropic_and_async(tmp_path):
    msg = AnthropicMessages().assemble(
        parse(
            (0.0, line)
            for line in (DATA / "stream_tool_use.sse").read_text(encoding="utf-8").split("\n")
        )
    )
    run = tmp_path / "run"
    price = {"claude-opus-5-5": __import__("callkit").Price(5, 25, 0.5, 6.25)}
    gw = Gateway(httpx2, (200, "application/json", json.dumps(msg).encode()))
    http = recording_http_client(
        run, lib="httpx2", transport=gw.transport(), prices=price, cap_usd=1e-9
    )
    claude = anthropic.Anthropic(base_url="http://gw.test", api_key="sk", http_client=http)
    claude.messages.create(model="claude-opus-5-5", max_tokens=10, messages=MSGS)
    with pytest.raises(anthropic.APIStatusError) as e:
        claude.messages.create(model="claude-opus-5-5", max_tokens=10, messages=MSGS)
    assert e.value.status_code == 402 and "reached the cap" in str(e.value)
    http2 = recording_async_http_client(run, transport=Gateway(httpx).transport(), cap_usd=1e-9)

    async def main():
        client = openai.AsyncOpenAI(base_url="http://gw.test/v1", api_key="sk", http_client=http2)
        with pytest.raises(openai.APIStatusError):
            await client.chat.completions.create(model="claude-opus-5-5", messages=MSGS)

    # the price table already exists in the same run directory, so the async client can cap even without passing prices
    asyncio.run(main())
    assert len(records(run)) == 1


def test_prices_conflict_and_run_none_warning(tmp_path):
    from callkit import Price

    recording_http_client(tmp_path / "run", prices={"m": Price(1, 2, 0.1, 1.25)})
    with pytest.raises(CallkitError, match="already has different prices"):
        recording_http_client(tmp_path / "run", prices={"m": Price(9, 9, 9, 9)})
    with pytest.warns(RuntimeWarning, match="ignored"):
        recording_http_client(None, cap_usd=5)


def test_ok_without_usage_is_unknown_cost(tmp_path):
    """Chat streaming without include_usage: success but no usage, cost unknown (not recorded as 0), flagged usage_missing with a warning.
    Failed attempts with no usage still record 0."""
    no_usage = b"\n".join(
        line for line in CHAT_SSE.split(b"\n") if b'"usage":{"completion_tokens"' not in line
    )
    gw = Gateway(httpx, (200, "text/event-stream", no_usage), (500, "application/json", b"{}"))
    client = oai(gw, tmp_path / "run", prices=PRICE)
    with pytest.warns(RuntimeWarning, match="include_usage"):
        for _ in client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS, stream=True):
            pass
    with pytest.raises(openai.InternalServerError):
        client.chat.completions.create(model="gpt-5.6-sol", messages=MSGS)
    ok, failed = records(tmp_path / "run")
    assert ok.outcome.ok and ok.cost_usd is None and ok.outcome.flags == ["usage_missing"]
    assert not failed.outcome.ok and failed.cost_usd == 0.0
