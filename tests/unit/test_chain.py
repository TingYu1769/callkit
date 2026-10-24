"""Concatenation, forking, normalization, parent inference, stale thinking counting."""

import copy
import json
from pathlib import Path

import pytest

from callkit import ToolResult, extend, fork
from callkit.chain import infer_parent, normalize_messages, stale_thinking
from callkit.errors import PairingError
from callkit.record import Record

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "meta_sample_v2.json"
THINK = {"type": "thinking", "thinking": "", "signature": "sig"}
CALL = {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "北京"}}
BODY = {
    "model": "claude-opus-5-5",
    "max_tokens": 100,
    "system": "你是一个简洁的助手。",
    "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
    "stream": True,
}


def rec(body=BODY, content=(THINK, CALL), stop="tool_use", cid="c_000001"):
    r = Record.from_dict(json.loads(SAMPLE.read_text(encoding="utf-8")))
    r.id = cid
    r.body = json.dumps(body, ensure_ascii=False).encode()
    r.message = {
        "id": f"msg_{cid}",
        "content": [copy.deepcopy(b) for b in content],
        "stop_reason": stop,
    }
    return r


def test_extend_tool_result():
    r = rec()
    after = extend(r, tool_results={"toolu_1": "晴，25°C"}, set_params={"max_tokens": 200})
    assert list(after) == list(BODY) and after["max_tokens"] == 200 and after["stream"] is True
    assert after["messages"][1] == {"role": "assistant", "content": [THINK, CALL]}
    assert after["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "晴，25°C"}],
    }
    assert json.loads(r.body) == BODY  # Original record unchanged


def test_extend_tool_result_first_then_text_and_is_error():
    after = extend(
        rec(), tool_results={"toolu_1": ToolResult("超时", is_error=True)}, text="换一个城市"
    )
    user = after["messages"][2]["content"]
    assert user[0]["is_error"] is True and user[1] == {"type": "text", "text": "换一个城市"}


def test_extend_pairing_errors():
    with pytest.raises(PairingError, match="missing"):
        extend(rec(), tool_results={})
    with pytest.raises(PairingError, match="extra"):
        extend(rec(), tool_results={"toolu_1": "a", "toolu_9": "b"})
    with pytest.raises(PairingError, match="prefill"):
        extend(rec(content=[{"type": "text", "text": "好"}], stop="end_turn"))
    after = extend(rec(content=[{"type": "text", "text": "好"}], stop="end_turn"), text="再说一遍")
    assert after["messages"][-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "再说一遍"}],
    }


def test_extend_pause_turn_sends_back_as_is():
    server = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}
    r = rec(content=[THINK, server], stop="pause_turn")
    after = extend(r)
    assert after["messages"][-1] == {"role": "assistant", "content": [THINK, server]}
    with pytest.raises(PairingError, match="pause_turn"):
        extend(r, text="继续")


def test_extend_mixed_turn_with_pending_server_tool():
    server = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}
    r = rec(content=[THINK, server, CALL])
    assert len(extend(r, tool_results={"toolu_1": "ok"})["messages"]) == 3
    with pytest.raises(PairingError, match="server tool calls"):
        extend(r, tool_results={"toolu_1": "ok"}, text="还有")
    with pytest.raises(PairingError, match="server tool calls"):
        extend(r, tool_results={"toolu_1": "ok"}, set_params={"tools": []})
    done = {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}
    assert extend(
        rec(content=[THINK, server, done, CALL]), tool_results={"toolu_1": "ok"}, text="还有"
    )


def test_fork_keeps_key_order_and_original():
    r = rec()
    after = fork(r, model="claude-sonnet-5-5", edit=lambda b: {**b, "max_tokens": 1})
    assert (
        list(after) == list(BODY)
        and after["model"] == "claude-sonnet-5-5"
        and after["max_tokens"] == 1
    )
    assert json.loads(r.body) == BODY


def test_normalize():
    a = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    assert normalize_messages(a) == normalize_messages([{"role": "user", "content": "x"}])


def test_infer_parent_needs_reply_to_disambiguate():
    a = rec(content=[{"type": "text", "text": "A"}], stop="end_turn", cid="c_000001")
    b = rec(content=[{"type": "text", "text": "B"}], stop="end_turn", cid="c_000002")
    nxt = extend(a, text="继续")
    assert infer_parent(nxt, [a, b]) == ("c_000001", [])
    twin = rec(content=[{"type": "text", "text": "A"}], stop="end_turn", cid="c_000003")
    assert infer_parent(nxt, [a, b, twin]) == (None, ["c_000001", "c_000003"])
    other = {**nxt, "system": "另一个系统提示"}
    assert infer_parent(other, [a, b]) == (None, [])


def test_infer_parent_claude_code_pattern():
    """Each turn moves the cache_control breakpoint to the last content block; the previous block with the breakpoint is collapsed to a string."""
    cc = {"type": "ephemeral"}
    body0 = {
        **BODY,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "任务"}]},
            {"role": "system", "content": [{"type": "text", "text": "环境", "cache_control": cc}]},
        ],
    }
    first = rec(body=body0, cid="c_000001")
    body1 = {
        **BODY,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "任务"}]},
            {"role": "system", "content": "环境"},
            {"role": "assistant", "content": [THINK, CALL]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
            {
                "role": "system",
                "content": [{"type": "text", "text": "剩余 token", "cache_control": cc}],
            },
        ],
    }
    assert infer_parent(body1, [first]) == ("c_000001", [])


def test_stale_thinking():
    r = rec()
    before = json.loads(r.body)
    assert stale_thinking(before, extend(r, tool_results={"toolu_1": "ok"}), [THINK, CALL]) == 0
    nxt = rec(body=extend(r, tool_results={"toolu_1": "ok"}), cid="c_000002")
    nb = json.loads(nxt.body)

    def edit_first_user(b):
        b["messages"][0]["content"] = "上海今天天气怎么样？"
        return b

    assert stale_thinking(nb, fork(nxt, edit=edit_first_user)) == 1
    assert stale_thinking(nb, fork(nxt, model="claude-sonnet-5-5")) == 1
    assert stale_thinking(nb, fork(nxt, edit=lambda b: {**b, "max_tokens": 9})) == 0
    assert stale_thinking(nb, fork(nxt, edit=lambda b: {**b, "tools": []})) == 1


def test_infer_parent_ignores_default_caller():
    """The endpoint returns caller: {"type": "direct"} on tool_use; Claude Code drops it when replaying."""
    with_caller = {**CALL, "caller": {"type": "direct"}}
    first = rec(content=[THINK, with_caller], cid="c_000001")
    replay = {
        **BODY,
        "messages": [
            *BODY["messages"],
            {"role": "assistant", "content": [THINK, CALL]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
        ],
    }
    assert infer_parent(replay, [first]) == ("c_000001", [])
    other = {**CALL, "caller": {"type": "code_execution_20260521", "tool_id": "srvtoolu_9"}}
    assert infer_parent(replay, [rec(content=[THINK, other], cid="c_000002")]) == (None, [])


def test_pending_calls_anthropic():
    from callkit import pending_calls

    assert pending_calls(rec()) == ["toolu_1"]
    assert pending_calls(rec(content=[{"type": "text", "text": "好"}], stop="end_turn")) == []
