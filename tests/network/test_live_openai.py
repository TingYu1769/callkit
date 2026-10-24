"""Makes one actual OpenAI Responses call (costs money, about $0.05-0.2). Disabled by default.

Requires CALLKIT_LIVE_OPENAI_BASE_URL (without /v1), CALLKIT_LIVE_OPENAI_API_KEY; optional
CALLKIT_LIVE_OPENAI_MODEL (default gpt-5.6-sol), CALLKIT_LIVE_PROXY.
  uv run pytest -m network tests/network/test_live_openai.py -q
"""

import os

import pytest

from callkit import Client, Endpoint, pending_calls

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.environ.get("CALLKIT_LIVE_OPENAI_BASE_URL"),
        reason="CALLKIT_LIVE_OPENAI_BASE_URL is not set",
    ),
]
TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "查询某个城市今天的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


# Some relay gateways route Responses requests to a channel that does not support them (returns 500); the live test relaxes attempts to 6.
ATTEMPTS = 6


def endpoint():
    return Endpoint(
        "live-openai",
        os.environ.get("CALLKIT_LIVE_OPENAI_BASE_URL", ""),
        "CALLKIT_LIVE_OPENAI_API_KEY",
        api="openai.responses",
        proxy=os.environ.get("CALLKIT_LIVE_PROXY") or None,
    )


def body(text, **extra):
    return {
        "model": os.environ.get("CALLKIT_LIVE_OPENAI_MODEL", "gpt-5.6-sol"),
        "input": text,
        "reasoning": {"effort": "low"},
        "store": False,
        "include": ["reasoning.encrypted_content"],
        **extra,
    }


def test_live_tool_call(tmp_path):
    with Client(endpoint(), tmp_path / "run", max_attempts=ATTEMPTS) as c:
        rec = c.call(body("北京今天天气怎么样？请调用工具查询。", tools=TOOLS))
    assert rec.outcome.ok and rec.outcome.stop_reason == "completed", rec.outcome
    assert pending_calls(rec), rec.message
    assert rec.usage and rec.usage.output_tokens > 0
    assert (tmp_path / "run" / "calls" / rec.id / "response.sse").exists()


def test_live_batch_and_resume(tmp_path):
    items = {f"q{i}": body(f"{i} 加 {i} 等于几？只回答数字。") for i in range(1, 4)}
    with Client(endpoint(), tmp_path / "run", max_attempts=ATTEMPTS) as c:
        recs = list(c.batch(items, concurrency=3))
    assert len(recs) == 3 and all(r.outcome.ok for r in recs), [r.outcome for r in recs]
    with Client(endpoint(), tmp_path / "run", max_attempts=ATTEMPTS) as c:
        assert list(c.batch(items, concurrency=3)) == []


def test_live_extend_and_fork(tmp_path):
    """Call the tool -> extend fills in the tool result to get the final answer -> fork changes reasoning effort and sends again."""
    from callkit import extend, fork

    with Client(endpoint(), tmp_path / "run", max_attempts=ATTEMPTS) as c:
        first = c.call(body("北京今天天气怎么样？请调用工具查询，然后用一句话回答。", tools=TOOLS))
        assert first.outcome.ok and pending_calls(first), first.outcome
        second = c.call(
            extend(first, tool_results={i: "晴，25°C，无降水" for i in pending_calls(first)}),
            parent=first,
        )
        assert second.outcome.ok and second.parent == first.id and second.edits == [], (
            second.outcome
        )
        assert second.message and not pending_calls(second)
        assert any(i["type"] == "message" for i in second.message["output"])
        third = c.call(
            fork(second, edit=lambda b: {**b, "reasoning": {"effort": "medium"}}), fork_of=second
        )
        assert third.outcome.ok and third.fork_of == second.id and third.edits == ["reasoning"], (
            third.outcome
        )
