"""OpenAI Chat Completions adapter: assembly, judgment, usage, non-streaming. Samples from a relay gateway's real stream and JSON."""

import json
from pathlib import Path

import pytest

from callkit.adapters import api_for_path, get_adapter
from callkit.adapters.openai_chat import OpenAIChat
from callkit.config import Endpoint
from callkit.sse import SseEvent, parse

DATA = Path(__file__).resolve().parents[1] / "data"
A = OpenAIChat()


def load(name="chat_tool_calls.sse"):
    text = (DATA / name).read_text(encoding="utf-8")
    return parse((0.0, line) for line in text.split("\n"))


def edit(events, fn):
    """Edit each chunk: fn receives the parsed chunk ([DONE] unchanged as a string), return None to delete, return a list to expand into multiple chunks."""
    out = []
    for ev in events:
        d = ev.data if ev.data == "[DONE]" else json.loads(ev.data)
        got = fn(d)
        for x in [] if got is None else got if isinstance(got, list) else [got]:
            out.append(SseEvent(None, x if isinstance(x, str) else json.dumps(x), ev.t))
    return out


def judge(events, status=200, error_body=None):
    return A.judge(status, "done", events, A.assemble(events), error_body)


def no_done(d):
    return None if d == "[DONE]" else d


def test_assemble_text():
    c = A.assemble(load("chat_text.sse"))
    assert c is not None and c["object"] == "chat.completion"
    # First chunk is an empty shell; use first non-empty value
    assert c["id"].startswith("chatcmpl-") and c["created"] > 0
    (ch,) = c["choices"]
    assert ch["finish_reason"] == "stop" and ch["message"]["role"] == "assistant"
    assert ch["message"]["content"].startswith("前缀缓存") and ch["message"]["refusal"] is None
    assert ch["content_filter_results"] == {} and c["usage"]["prompt_tokens"] == 17


def test_assemble_tool_calls_matches_json_shape():
    c = A.assemble(load())
    assert c is not None
    msg = c["choices"][0]["message"]
    assert msg["content"] is None and [tc["function"]["name"] for tc in msg["tool_calls"]] == [
        "get_weather",
        "get_weather",
    ]
    assert [json.loads(tc["function"]["arguments"]) for tc in msg["tool_calls"]] == [
        {"city": "北京"},
        {"city": "上海"},
    ]
    assert all(tc["id"].startswith("call_") and "index" not in tc for tc in msg["tool_calls"])
    shape = json.loads((DATA / "chat_tool_calls.json").read_text(encoding="utf-8"))
    assert set(msg) <= set(shape["choices"][0]["message"]) and set(msg["tool_calls"][0]) == set(
        shape["choices"][0]["message"]["tool_calls"][0]
    )


def test_good_streams_ok():
    j = judge(load())
    assert j.outcome.ok and j.outcome.protocol == "completed" and j.reason == "completed"
    assert j.outcome.stop_reason == "tool_calls" and not j.retryable
    assert judge(load("chat_text.sse")).outcome.stop_reason == "stop"


def test_finish_reason_without_done_counts_as_finished():
    assert judge(edit(load(), no_done)).outcome.ok


def test_cut_stream_is_retryable():
    def cut(d):
        if d == "[DONE]" or (d.get("choices") and d["choices"][0].get("finish_reason")):
            return None
        return d

    j = judge(edit(load(), cut))
    assert not j.outcome.ok and j.outcome.protocol == "incomplete" and j.retryable
    assert j.reason == "no_done"


def test_n2_one_choice_unfinished_is_cut():
    def twin(d):
        if d == "[DONE]" or not d.get("choices"):
            return d
        other = json.loads(json.dumps(d))
        for ch in other["choices"]:
            ch["index"] = 1
            ch["finish_reason"] = None  # Second response is unfinished
        return [d, other]

    events = edit(edit(load(), twin), no_done)
    c = A.assemble(events)
    assert c is not None and len(c["choices"]) == 2
    assert not judge(events).outcome.ok


def test_error_chunk():
    def boom(code):
        def fn(d):
            if d == "[DONE]":
                return [{"error": {"code": code, "message": "x"}}, d]
            return d

        return fn

    j = judge(edit(load(), boom("server_error")))
    assert not j.outcome.ok and j.retryable and j.reason == "server_error"
    j = judge(edit(load(), boom("invalid_request_error")))
    assert not j.retryable and j.reason == "invalid_request_error"


def cut_last_call(finish):
    def fn(d):
        if d == "[DONE]" or not d.get("choices"):
            return d
        ch = d["choices"][0]
        for tc in (ch.get("delta") or {}).get("tool_calls") or []:
            if tc.get("index") == 1 and tc["function"].get("arguments") == '"上海"}':
                tc["function"]["arguments"] = '"上'
        if ch.get("finish_reason"):
            ch["finish_reason"] = finish
        return d

    return fn


def test_truncated_arguments_ok_only_when_cut_by_length():
    j = judge(edit(load(), cut_last_call("length")))
    assert j.outcome.ok and j.outcome.flags == ["truncated_tool_input"]
    assert j.outcome.stop_reason == "length"
    j = judge(edit(load(), cut_last_call("tool_calls")))
    assert (
        not j.outcome.ok
        and j.reason == "bad_assembly"
        and "not valid JSON" in (j.outcome.error or "")
    )


@pytest.mark.parametrize(("status", "retryable"), [(404, False), (429, True), (502, True)])
def test_http_errors(status, retryable):
    body = json.dumps({"error": {"code": "model_not_found", "message": "x"}})
    j = judge([], status=status, error_body=body)
    assert not j.outcome.ok and j.retryable is retryable and j.reason == "model_not_found"
    assert judge([], status=status, error_body="<html>").reason == f"http_{status}"


def test_transport_failure_without_status():
    j = A.judge(None, "connect_error", [], None, "ConnectError: x")
    assert not j.outcome.ok and j.retryable and j.reason == "connect_error"


def test_usage_maps_cache_and_reasoning():
    raw = {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "prompt_tokens_details": {
            "cached_tokens": 600,
            "cache_write_tokens": 300,
            "cache_write_tokens_details": {"ttl_30m_tokens": 100, "ttl_12h_tokens": 200},
        },
        "completion_tokens_details": {"reasoning_tokens": 20},
    }
    u = A.usage(
        edit(
            load(),
            lambda d: {**d, "usage": raw} if d != "[DONE]" and "usage" in d and d["usage"] else d,
        )
    )
    assert u is not None and u.input_tokens == 100 and u.cache_read_tokens == 600
    assert u.cache_write_tokens == 300 and u.cache_write_by_ttl == {"30m": 100, "12h": 200}
    assert u.output_tokens == 50 and u.thinking_tokens == 20
    no_usage = edit(load(), lambda d: d if d == "[DONE]" or d.get("choices") else None)
    assert A.usage(no_usage) is None


def test_tool_calls_without_index_kept_apart():
    def strip_index(d):
        if d == "[DONE]" or not d.get("choices"):
            return d
        for tc in (d["choices"][0].get("delta") or {}).get("tool_calls") or []:
            tc.pop("index", None)
        return d

    c = A.assemble(edit(load(), strip_index))
    assert c is not None
    calls = c["choices"][0]["message"]["tool_calls"]
    assert [json.loads(tc["function"]["arguments"]) for tc in calls] == [
        {"city": "北京"},
        {"city": "上海"},
    ]


def test_logprobs_lists_concatenated():
    def add_lp(d):
        if d == "[DONE]" or not d.get("choices"):
            return d
        ch = d["choices"][0]
        text = (ch.get("delta") or {}).get("content")
        if text:
            ch["logprobs"] = {"content": [{"token": text, "logprob": -0.1}], "refusal": None}
        return d

    c = A.assemble(edit(load("chat_text.sse"), add_lp))
    assert c is not None
    lp = c["choices"][0]["logprobs"]
    assert "".join(t["token"] for t in lp["content"]) == c["choices"][0]["message"]["content"]
    assert lp["refusal"] is None


def test_from_json():
    text = (DATA / "chat_tool_calls.json").read_text(encoding="utf-8")
    body, j, u = A.from_json(200, "done", text)
    assert body == json.loads(text) and j.outcome.ok and j.outcome.stop_reason == "tool_calls"
    assert u is not None and u.input_tokens == 139 and u.output_tokens == 47
    body, j, u = A.from_json(200, "eof", text[:100])
    assert body is None and u is None
    assert j.outcome.protocol == "incomplete" and j.retryable and j.reason == "bad_json"
    assert j.outcome.transport_end == "eof"
    err = json.dumps({"error": {"code": "server_error", "message": "x"}})
    body, j, _ = A.from_json(200, "done", err)
    assert body is None and j.retryable and j.reason == "server_error"
    body, j, _ = A.from_json(404, "done", err)
    assert body is None and not j.retryable


def test_stream_defaults():
    assert A.stream_defaults({}) == {"stream": True, "stream_options": {"include_usage": True}}
    assert A.stream_defaults({"stream_options": {"include_usage": False}}) == {"stream": True}


def test_http_request_and_registry():
    ep = Endpoint("v", "https://gw.example/", "K", api="openai.chat_completions")
    req = A.http_request(b"{}", ep, "sk-1")
    assert req.url == "https://gw.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-1"
    assert get_adapter("openai.chat_completions").api == A.api


@pytest.mark.parametrize(
    ("path", "api"),
    [
        ("/v1/chat/completions", "openai.chat_completions"),
        ("/compatible-mode/v1/chat/completions", "openai.chat_completions"),
        ("/v1/responses", "openai.responses"),
        ("/openai/v1/responses", "openai.responses"),
        ("/v1/messages", "anthropic.messages"),
        ("/v1/threads/t1/messages", None),
        ("/v1/threads/t1/runs/r1/responses", None),
        ("/v1/responses/compact", None),
        ("/v1/embeddings", None),
        ("/messages", None),
    ],
)
def test_api_for_path(path, api):
    assert api_for_path(path) == api


def test_no_choices_is_not_success():
    shell_only = edit(load(), lambda d: d if d == "[DONE]" or not d.get("choices") else None)
    j = judge(shell_only)
    assert not j.outcome.ok and j.outcome.protocol == "incomplete" and j.retryable
    assert j.reason == "no_choices"
    _, j, _ = A.from_json(200, "done", json.dumps({"id": "x", "choices": []}))
    assert not j.outcome.ok and j.reason == "no_choices"
    _, j, _ = A.from_json(200, "done", json.dumps({"detail": "网关的别的回复"}))
    assert not j.outcome.ok and not j.retryable and j.reason == "unexpected_body"


def test_dict_deltas_merged_not_overwritten():
    def audio(d):
        if d == "[DONE]" or not d.get("choices"):
            return d
        delta = d["choices"][0].get("delta") or {}
        if delta.get("role"):
            delta["audio"] = {"id": "audio_1", "transcript": ""}
        elif delta.get("content"):
            delta["audio"] = {"data": "QUFB"}
        return d

    c = A.assemble(edit(load("chat_text.sse"), audio))
    assert c is not None and c["choices"][0]["message"]["audio"]["id"] == "audio_1"


# ---- regression tests ----


def legacy_function_call(pieces, finish="function_call"):
    """Legacy function_call stream: name in the first chunk, arguments arrive in pieces."""
    first = {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "function_call": {"name": "f", "arguments": ""}},
            }
        ]
    }
    rest = [
        {"choices": [{"index": 0, "delta": {"function_call": {"arguments": p}}}]} for p in pieces
    ]
    end = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    return [SseEvent(None, json.dumps(d), 0.0) for d in [first, *rest, end]] + [
        SseEvent(None, "[DONE]", 0.0)
    ]


def test_nested_string_deltas_concatenated():
    events = legacy_function_call(['{"x":', "1}"])
    c = A.assemble(events)
    assert c is not None
    assert c["choices"][0]["message"]["function_call"] == {"name": "f", "arguments": '{"x":1}'}
    assert judge(events).outcome.ok
    j = judge(legacy_function_call(['{"x":']))
    assert not j.outcome.ok and "function_call" in (j.outcome.error or "")
    assert judge(legacy_function_call(['{"x":'], finish="length")).outcome.flags == [
        "truncated_tool_input"
    ]


def test_audio_pieces_concatenated_and_id_kept():
    def audio(d):
        if d == "[DONE]" or not d.get("choices"):
            return d
        delta = d["choices"][0].get("delta") or {}
        if delta.get("role"):
            delta["audio"] = {"id": "audio_1", "expires_at": 1}
        elif delta.get("content"):
            delta["audio"] = {"data": "QQ", "transcript": delta["content"]}
        return d

    c = A.assemble(edit(load("chat_text.sse"), audio))
    assert c is not None
    msg = c["choices"][0]["message"]
    assert msg["audio"]["id"] == "audio_1" and msg["audio"]["transcript"] == msg["content"]
    assert set(msg["audio"]["data"]) == {"Q"} and len(msg["audio"]["data"]) > 2


def test_interleaved_calls_without_index_by_id():
    def chunk(tc):
        return SseEvent(
            None, json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [tc]}}]}), 0.0
        )

    events = [
        chunk({"id": "A", "type": "function", "function": {"name": "f", "arguments": '{"a":'}}),
        chunk({"id": "B", "type": "function", "function": {"name": "g", "arguments": '{"b":2}'}}),
        chunk({"id": "A", "function": {"arguments": "1}"}}),
        SseEvent(
            None,
            json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            0.0,
        ),
        SseEvent(None, "[DONE]", 0.0),
    ]
    c = A.assemble(events)
    assert c is not None
    calls = c["choices"][0]["message"]["tool_calls"]
    assert [(tc["id"], tc["function"]["arguments"]) for tc in calls] == [
        ("A", '{"a":1}'),
        ("B", '{"b":2}'),
    ]
    assert judge(events).outcome.ok


def test_non_chat_json_is_unexpected():
    legacy = {
        "object": "text_completion",
        "choices": [{"index": 0, "text": "2", "finish_reason": "stop"}],
    }
    _, j, _ = A.from_json(200, "done", json.dumps(legacy))
    assert not j.outcome.ok and j.reason == "unexpected_body"
    one_chunk = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "2"}}],
    }
    _, j, _ = A.from_json(200, "done", json.dumps(one_chunk))
    assert not j.outcome.ok and j.reason == "unexpected_body"
    no_object = json.loads((DATA / "chat_tool_calls.json").read_text(encoding="utf-8"))
    no_object.pop("object")
    # Some compatibility services omit object
    assert A.from_json(200, "done", json.dumps(no_object))[1].outcome.ok


def test_usage_reads_deepseek_cache_field():
    """DeepSeek usage includes prompt_cache_hit_tokens; honor it even without prompt_tokens_details."""
    raw = {
        "prompt_tokens": 412,
        "completion_tokens": 19,
        "prompt_cache_hit_tokens": 256,
        "prompt_cache_miss_tokens": 156,
    }
    _, _, u = A.from_json(
        200,
        "done",
        json.dumps(
            {
                **json.loads((DATA / "chat_tool_calls.json").read_text(encoding="utf-8")),
                "usage": raw,
            }
        ),
    )
    assert u is not None and u.cache_read_tokens == 256 and u.input_tokens == 156
