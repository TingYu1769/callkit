"""Chat Completions concatenation, parent inference, and converting non-streaming requests to streaming. The sample uses a real request and a real JSON response."""

import copy
import json
from pathlib import Path

import pytest

from callkit import ToolResult, extend, fork, pending_calls
from callkit.chain import infer_parent, stale_thinking
from callkit.chains.openai_chat import replay_message
from callkit.errors import CallkitError, PairingError
from callkit.record import Record

DATA = Path(__file__).resolve().parents[1] / "data"
BODY = json.loads((DATA / "chat_tool_calls.request.json").read_text(encoding="utf-8"))
REPLY = json.loads((DATA / "chat_tool_calls.json").read_text(encoding="utf-8"))
CHAT = "openai.chat_completions"
TEXT = {"role": "assistant", "content": "好", "refusal": None, "annotations": []}


def rec(body=None, message=None, cid="c_000001"):
    r = Record.from_dict(json.loads((DATA / "meta_sample_v2.json").read_text(encoding="utf-8")))
    r.id, r.api = cid, CHAT
    r.body = json.dumps(body if body is not None else BODY, ensure_ascii=False).encode()
    reply = copy.deepcopy(REPLY)
    if message is not None:
        reply["choices"][0]["message"] = copy.deepcopy(message)
    r.message = reply
    return r


def results(r):
    return {cid: f"晴（{i}）" for i, cid in enumerate(pending_calls(r))}


def test_extend_replays_cleaned_message_and_appends_tool_messages():
    r = rec()
    after = extend(r, tool_results=results(r), set_params={"temperature": 0})
    n = len(BODY["messages"])
    assert after["messages"][:n] == BODY["messages"]
    replayed = after["messages"][n]
    raw = REPLY["choices"][0]["message"]
    assert replayed == {
        "role": "assistant",
        "tool_calls": raw["tool_calls"],
    }  # drop null and annotations
    tail = after["messages"][n + 1 :]
    assert tail == [
        {"role": "tool", "tool_call_id": cid, "content": f"晴（{i}）"}
        for i, cid in enumerate(pending_calls(r))
    ]
    assert after["temperature"] == 0 and list(after)[: len(BODY)] == list(BODY)
    # do not add stream if the original request omitted it
    assert json.loads(r.body) == BODY and "stream" not in after


def test_replay_keeps_vendor_fields_and_trims_audio():
    msg = {
        "role": "assistant",
        "content": "答",
        "reasoning_content": "想了想",  # DeepSeek requires it to be sent back unchanged
        "function_call": {"name": "f", "arguments": "{}"},
        "audio": {"id": "audio_1", "data": "AAAA", "transcript": "答", "expires_at": 1},
        "annotations": [{"type": "url_citation"}],
        "refusal": None,
    }
    assert replay_message(msg) == {
        "role": "assistant",
        "content": "答",
        "reasoning_content": "想了想",
        "function_call": {"name": "f", "arguments": "{}"},
        "audio": {"id": "audio_1"},
    }
    after = extend(rec(message=msg), text="继续", set_params={"stream": True})
    assert after["messages"][-2] == replay_message(msg) and after["stream"] is True
    assert after["messages"][-1] == {"role": "user", "content": "继续"}


def test_extend_errors():
    r = rec()
    with pytest.raises(PairingError, match="missing"):
        extend(r, tool_results={})
    with pytest.raises(PairingError, match="extra"):
        extend(r, tool_results={**results(r), "call_zzz": "x"})
    with pytest.raises(PairingError, match="user text"):
        extend(rec(message=TEXT))
    r.message = None
    with pytest.raises(CallkitError, match="no assembled response"):
        extend(r, text="x")


def test_tool_result_object_uses_content():
    r = rec()
    ids = pending_calls(r)
    after = extend(r, tool_results={ids[0]: ToolResult("坏了", is_error=True), ids[1]: "晴"})
    assert after["messages"][-2]["content"] == "坏了"


def test_infer_parent_matches_raw_replay():
    """SDK users often put the whole message back unchanged (with refusal: null, annotations: []); it must still match."""
    first = rec()
    raw = REPLY["choices"][0]["message"]
    tools = [{"role": "tool", "tool_call_id": cid, "content": "晴"} for cid in pending_calls(first)]
    nxt = {**BODY, "messages": [*BODY["messages"], copy.deepcopy(raw), *tools]}
    assert infer_parent(nxt, [first], api=CHAT) == ("c_000001", [])
    assert infer_parent(extend(first, tool_results=results(first)), [first], api=CHAT) == (
        "c_000001",
        [],
    )


def test_infer_parent_string_and_text_parts_are_same():
    first = rec(body={**BODY, "messages": [{"role": "user", "content": "你好"}]}, message=TEXT)
    nxt = extend(first, text="继续")
    nxt["messages"][0]["content"] = [{"type": "text", "text": "你好"}]
    assert infer_parent(nxt, [first], api=CHAT) == ("c_000001", [])


def test_infer_parent_disambiguates_by_reply_and_tools():
    a = rec(message={**TEXT, "content": "A"}, cid="c_000001")
    b = rec(message={**TEXT, "content": "B"}, cid="c_000002")
    nxt = extend(a, text="继续")
    assert infer_parent(nxt, [a, b], api=CHAT) == ("c_000001", [])
    twin = rec(message={**TEXT, "content": "A"}, cid="c_000003")
    assert infer_parent(nxt, [a, twin], api=CHAT) == (None, ["c_000001", "c_000003"])
    assert infer_parent({**nxt, "tools": []}, [a], api=CHAT) == (None, [])
    longer = rec(body=nxt, message={**TEXT, "content": "C"}, cid="c_000004")
    deeper = extend(longer, text="再继续")
    assert infer_parent(deeper, [a, longer], api=CHAT) == ("c_000004", [])


def test_stream_false_becomes_stream_on_fork_and_extend():
    first = rec(body={**BODY, "stream": False})
    body = fork(first, model="gpt-5.6-pro")
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["model"] == "gpt-5.6-pro"
    after = extend(first, tool_results=results(first))
    assert after["stream"] is True and after["stream_options"] == {"include_usage": True}
    kept = rec(body={**BODY, "stream": False, "stream_options": {"include_usage": False}})
    assert fork(kept)["stream_options"] == {"include_usage": False}
    assert "stream" not in fork(rec())  # if stream is absent, Client fills it in


def test_no_stale_reasoning_for_chat():
    r = rec()
    assert stale_thinking(json.loads(r.body), fork(r, model="gpt-5.6-pro"), api=CHAT) == 0
