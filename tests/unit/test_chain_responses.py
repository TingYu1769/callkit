"""Responses concatenation, parent inference, and stale reasoning counting. Samples are real requests and real streams."""

import copy
import json
from pathlib import Path

import pytest

from callkit import ToolResult, extend, fork, pending_calls
from callkit.adapters.openai_responses import OpenAIResponses
from callkit.chain import infer_parent, stale_thinking
from callkit.errors import CallkitError, PairingError
from callkit.record import Record
from callkit.sse import parse

DATA = Path(__file__).resolve().parents[1] / "data"
BODY = json.loads((DATA / "responses_tool_calls.request.json").read_text(encoding="utf-8"))
MESSAGE = OpenAIResponses().assemble(
    parse(
        (0.0, line)
        for line in (DATA / "responses_tool_calls.sse").read_text(encoding="utf-8").split("\n")
    )
)
TEXT = {
    "type": "message",
    "id": "msg_x",
    "role": "assistant",
    "status": "completed",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": "好", "annotations": [], "logprobs": []}],
}


def rec(body=None, output=None, cid="c_000001"):
    r = Record.from_dict(json.loads((DATA / "meta_sample_v2.json").read_text(encoding="utf-8")))
    r.id, r.api = cid, "openai.responses"
    r.body = json.dumps(body if body is not None else BODY, ensure_ascii=False).encode()
    assert MESSAGE is not None
    r.message = {
        **MESSAGE,
        "output": copy.deepcopy(output if output is not None else MESSAGE["output"]),
    }
    return r


def results(r):
    return {cid: f"晴（{i}）" for i, cid in enumerate(pending_calls(r))}


def test_extend_replays_output_and_appends_results():
    r = rec()
    after = extend(r, tool_results=results(r), set_params={"reasoning": {"effort": "low"}})
    out = MESSAGE["output"]
    assert after["input"][: len(BODY["input"])] == BODY["input"]
    assert after["input"][len(BODY["input"]) : len(BODY["input"]) + len(out)] == out
    tail = after["input"][len(BODY["input"]) + len(out) :]
    assert [t["type"] for t in tail] == ["function_call_output"] * 4
    assert [t["call_id"] for t in tail] == pending_calls(r) and tail[0]["output"] == "晴（0）"
    assert after["reasoning"] == {"effort": "low"} and list(after) == list(BODY)
    assert json.loads(r.body) == BODY


def test_extend_text_string_input_custom_calls_and_echoed_fields():
    custom = {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "status": "completed",
        "call_id": "call_c",
        "name": "exec",
        "input": "ls",
        "metadata": {"turn_id": "t"},
        "internal_chat_message_metadata_passthrough": {"turn_id": "t"},
    }
    r = rec(body={**BODY, "input": "列一下目录"}, output=[TEXT, custom])
    after = extend(r, tool_results={"call_c": ToolResult("a.txt", is_error=True)}, text="继续")
    assert after["input"][0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "列一下目录"}],
    }
    replayed = after["input"][2]
    assert (
        "metadata" not in replayed and "internal_chat_message_metadata_passthrough" not in replayed
    )
    assert replayed["status"] == "completed" and replayed["id"] == "ctc_1"
    assert after["input"][3] == {
        "type": "custom_tool_call_output",
        "call_id": "call_c",
        "output": "a.txt",
    }
    assert after["input"][4]["content"] == [{"type": "input_text", "text": "继续"}]


def test_extend_errors():
    r = rec()
    with pytest.raises(PairingError, match="missing"):
        extend(r, tool_results={})
    with pytest.raises(PairingError, match="extra"):
        extend(r, tool_results={**results(r), "call_zzz": "x"})
    with pytest.raises(PairingError, match="user text"):
        extend(rec(output=[TEXT]))
    assert extend(rec(output=[TEXT]), text="再说一遍")["input"][-1]["role"] == "user"
    search = {
        "type": "tool_search_call",
        "call_id": "call_s",
        "execution": "client",
        "arguments": {},
    }
    with pytest.raises(CallkitError, match="tool_search_call"):
        extend(rec(output=[search]), tool_results={"call_s": "x"})
    server_search = {**search, "execution": "server"}
    assert pending_calls(rec(output=[server_search, TEXT])) == []
    with pytest.raises(CallkitError, match="previous_response_id"):
        extend(rec(body={**BODY, "previous_response_id": "resp_1"}), tool_results=results(r))


def test_infer_parent_matches_codex_style_replay():
    """Codex replay strips echoed fields such as metadata while id and status stay the same; comparing only key fields allows matching."""
    echoed = [{**item, "metadata": {"turn_id": "t"}} for item in MESSAGE["output"]]
    first = rec(output=echoed)
    nxt = extend(first, tool_results=results(first))
    assert infer_parent(nxt, [first], api="openai.responses") == ("c_000001", [])
    # defaults to Anthropic lookup, does not cross APIs
    assert infer_parent(nxt, [first]) == (None, [])


def test_infer_parent_disambiguates_by_reply_and_context():
    a = rec(output=[{**TEXT, "content": [{"type": "output_text", "text": "A"}]}], cid="c_000001")
    b = rec(output=[{**TEXT, "content": [{"type": "output_text", "text": "B"}]}], cid="c_000002")
    nxt = extend(a, text="继续")
    assert infer_parent(nxt, [a, b], api="openai.responses") == ("c_000001", [])
    twin = rec(
        output=[{**TEXT, "id": "msg_other", "content": [{"type": "output_text", "text": "A"}]}],
        cid="c_000003",
    )
    assert infer_parent(nxt, [a, twin], api="openai.responses") == (None, ["c_000001", "c_000003"])
    assert infer_parent({**nxt, "tools": []}, [a], api="openai.responses") == (None, [])


def test_stale_reasoning_only_on_model_change():
    r = rec()
    nxt = extend(r, tool_results=results(r))
    second = rec(body=nxt, output=[TEXT], cid="c_000002")
    nb = json.loads(second.body)
    assert stale_thinking(nb, fork(second, model="gpt-5.6-pro"), api="openai.responses") == 1
    assert (
        stale_thinking(
            nb, fork(second, edit=lambda b: {**b, "store": True}), api="openai.responses"
        )
        == 0
    )


def test_infer_parent_ignores_volatile_fields_on_other_items():
    """Server tool entries (e.g. web_search_call) are replayed without echoed metadata and with a changed status; they should still match."""
    search = {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"query": "北京天气"},
        "metadata": {"turn_id": "t"},
    }
    first = rec(output=[search, TEXT])
    nxt = extend(first, text="继续")
    nxt["input"][len(BODY["input"])]["status"] = "in_progress"
    assert infer_parent(nxt, [first], api="openai.responses") == ("c_000001", [])


def test_infer_parent_tells_apart_images_namespaces_and_search_results():
    """Different images, namespaces, or search results must not match as parent."""
    img = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "看图"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
        ],
    }
    first = rec(body={**BODY, "input": [img]}, output=[TEXT])
    nxt = extend(first, text="继续")
    assert infer_parent(nxt, [first], api="openai.responses") == ("c_000001", [])
    other = copy.deepcopy(nxt)
    other["input"][0]["content"][1]["image_url"] = "data:image/png;base64,BBB"
    assert infer_parent(other, [first], api="openai.responses") == (None, [])

    call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "run",
        "namespace": "functions",
        "arguments": "{}",
    }
    first = rec(output=[call, TEXT])
    nxt = extend(first, tool_results={"call_1": "ok"}, text="继续")
    assert infer_parent(nxt, [first], api="openai.responses") == ("c_000001", [])
    moved = copy.deepcopy(nxt)
    moved["input"][len(BODY["input"])]["namespace"] = "mcp__other"
    assert infer_parent(moved, [first], api="openai.responses") == (None, [])

    found = {
        "type": "tool_search_output",
        "call_id": "call_s",
        "status": "completed",
        "execution": "client",
        "tools": [{"name": "a"}],
    }
    first = rec(body={**BODY, "input": [*BODY["input"], found]}, output=[TEXT])
    nxt = extend(first, text="继续")
    assert infer_parent(nxt, [first], api="openai.responses") == ("c_000001", [])
    swapped = copy.deepcopy(nxt)
    swapped["input"][len(BODY["input"])]["tools"] = [{"name": "b"}]
    assert infer_parent(swapped, [first], api="openai.responses") == (None, [])
