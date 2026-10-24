"""Makes one real model call (costs about $0.01). Not run by default: make test uses -m "not network".

Set environment variables and run manually: CALLKIT_LIVE_BASE_URL (without /v1), CALLKIT_LIVE_API_KEY,
optional CALLKIT_LIVE_MODEL (default claude-sonnet-5-5),
CALLKIT_LIVE_PROXY (proxy for reaching the endpoint, e.g. http://proxy.example:8080).
  uv run pytest -m network tests/network -q
"""

import os

import pytest

from callkit import Client, Endpoint

pytestmark = pytest.mark.network

PROXY = os.environ.get("CALLKIT_LIVE_PROXY") or None

TOOLS = [
    {
        "name": "get_weather",
        "description": "查询某个城市今天的天气",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


@pytest.mark.skipif(
    not os.environ.get("CALLKIT_LIVE_BASE_URL"), reason="CALLKIT_LIVE_BASE_URL is not set"
)
def test_live_tool_call(tmp_path):
    ep = Endpoint("live", os.environ["CALLKIT_LIVE_BASE_URL"], "CALLKIT_LIVE_API_KEY", proxy=PROXY)
    body = {
        "model": os.environ.get("CALLKIT_LIVE_MODEL", "claude-sonnet-5-5"),
        "max_tokens": 2000,
        "output_config": {"effort": "low"},
        "tools": TOOLS,
        "messages": [{"role": "user", "content": "北京今天天气怎么样？请调用工具查询。"}],
    }
    with Client(ep, tmp_path / "run") as c:
        rec = c.call(body)
    assert rec.outcome.ok, rec.outcome
    assert rec.outcome.stop_reason == "tool_use"
    assert rec.message and any(b["type"] == "tool_use" for b in rec.message["content"])
    assert (tmp_path / "run" / "calls" / rec.id / "response.sse").exists()


@pytest.mark.skipif(
    not os.environ.get("CALLKIT_LIVE_BASE_URL"), reason="CALLKIT_LIVE_BASE_URL is not set"
)
def test_live_extend_and_fork(tmp_path):
    """Call a tool, extend with the tool result to get the final answer, then fork a second call with a changed effort and send it again."""
    from callkit import extend, fork

    ep = Endpoint("live", os.environ["CALLKIT_LIVE_BASE_URL"], "CALLKIT_LIVE_API_KEY", proxy=PROXY)
    body = {
        "model": os.environ.get("CALLKIT_LIVE_MODEL", "claude-sonnet-5-5"),
        "max_tokens": 2000,
        "output_config": {"effort": "low"},
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？请调用工具查询，然后用一句话回答。"}
        ],
    }
    with Client(ep, tmp_path / "run") as c:
        first = c.call(body)
        assert first.outcome.ok and first.outcome.stop_reason == "tool_use", first.outcome
        assert first.message
        ids = [b["id"] for b in first.message["content"] if b["type"] == "tool_use"]
        second = c.call(
            extend(first, tool_results={i: "晴，25°C，无降水" for i in ids}), parent=first
        )
        assert second.outcome.ok and second.outcome.stop_reason == "end_turn", second.outcome
        assert second.parent == first.id
        third = c.call(
            fork(second, edit=lambda b: {**b, "output_config": {"effort": "medium"}}),
            fork_of=second,
        )
        assert third.outcome.ok, third.outcome
        assert third.fork_of == second.id and third.edits == ["output_config"]


@pytest.mark.skipif(
    not os.environ.get("CALLKIT_LIVE_BASE_URL"), reason="CALLKIT_LIVE_BASE_URL is not set"
)
def test_live_batch_and_resume(tmp_path):
    ep = Endpoint("live", os.environ["CALLKIT_LIVE_BASE_URL"], "CALLKIT_LIVE_API_KEY", proxy=PROXY)
    model = os.environ.get("CALLKIT_LIVE_MODEL", "claude-sonnet-5-5")
    items = {
        f"q{i}": {
            "model": model,
            "max_tokens": 1000,
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": f"{i} 加 {i} 等于几？只回答数字。"}],
        }
        for i in range(1, 4)
    }
    with Client(ep, tmp_path / "run") as c:
        recs = list(c.batch(items, concurrency=3))
    assert len(recs) == 3 and all(r.outcome.ok for r in recs), [r.outcome for r in recs]
    with Client(ep, tmp_path / "run") as c:
        assert list(c.batch(items, concurrency=3)) == []
