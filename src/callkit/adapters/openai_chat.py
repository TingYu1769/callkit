"""OpenAI Chat Completions adapter. Only performs offline conversion.

Stream assembly: only ``data:`` lines are used, ending with ``data: [DONE]``. Top-level fields take the first non-empty value from each chunk (some gateways start with an empty shell whose id is empty and created is 0); chunks are merged by ``choices[].index``: string deltas (content, refusal, reasoning_content from compatible services, etc.) are concatenated chunk by chunk, list deltas are appended, and ``tool_calls`` are merged by their own index (the first chunk carries id, type, and function name; later chunks only carry argument fragments). Other choice-level fields (``content_filter_results``, etc.) also take the first non-empty value, and lists inside ``logprobs`` are concatenated. A chunk with empty ``choices`` and a ``usage`` field is the usage chunk (requires ``stream_options.include_usage``).

Completion: ``[DONE]`` counts as finished; otherwise every choice that appeared must have a ``finish_reason``. Checking only whether any ``finish_reason`` appeared is not enough: with n>1, a stream where one choice finished and the rest stopped halfway would be treated as complete. The stop reason is copied from ``choices[0].finish_reason``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from callkit.adapters.base import (
    RETRYABLE_CODES,
    HttpRequest,
    Judgement,
    bad_json,
    build_request,
    failed,
    parse_json,
)
from callkit.config import Endpoint
from callkit.record import Outcome, Usage
from callkit.sse import SseEvent

DONE = "[DONE]"
# the two sub-objects in tool_calls that hold names and arguments
CALL_PARTS = ("function", "custom")
APPENDED = ("arguments", "input")  # these fields are concatenated chunk by chunk
IDENTITY = ("role", "id", "name", "type")  # fields given only once in deltas and not concatenated


class OpenAIChat:
    api = "openai.chat_completions"
    path = "/v1/chat/completions"  # default path, appended after the endpoint base_url

    def http_request(
        self,
        body: bytes,
        endpoint: Endpoint,
        key: str,
        *,
        path: str | None = None,
        inherit_headers: Mapping[str, str] | None = None,
    ) -> HttpRequest:
        return build_request(
            body,
            endpoint,
            key,
            path=path or self.path,
            default_auth="authorization",
            fixed_headers={},
            inherit_headers=inherit_headers,
        )

    def stream_defaults(self, body: Mapping[str, Any]) -> dict[str, Any]:
        extra: dict[str, Any] = {"stream": True}
        if "stream_options" not in body:
            # without this the stream has no usage
            extra["stream_options"] = {"include_usage": True}
        return extra

    def assemble(self, events: Sequence[SseEvent]) -> dict[str, Any] | None:
        return _assemble(events).completion()

    def judge(
        self,
        status: int | None,
        transport_end: str,
        events: Sequence[SseEvent],
        message: dict[str, Any] | None,
        error_body: str | None,
        request: Mapping[str, Any] | None = None,
    ) -> Judgement:
        if status is None:
            return failed(transport_end, "failed", transport_end, True, error_body)
        if status != 200:
            return _http_error(transport_end, status, error_body)
        a = _assemble(events)
        if a.errors:
            return _error_judgement(transport_end, a.errors[0])
        finished = a.done or (
            bool(a.choices) and all(c["finish_reason"] for c in a.choices.values())
        )
        if not finished:
            return failed(
                transport_end,
                "incomplete",
                "no_done",
                True,
                "stream ended without [DONE] and not every choice has a finish_reason",
            )
        return _judge_choices(transport_end, a.completion() or {})

    def from_json(
        self,
        status: int | None,
        transport_end: str,
        text: str,
        request: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Judgement, Usage | None]:
        if status != 200:
            return None, self.judge(status, transport_end, [], None, text, request), None
        body = parse_json(text)
        if body is None:
            return None, bad_json(transport_end), None
        if isinstance(body.get("error"), dict):
            return None, _error_judgement(transport_end, body["error"]), None
        # e.g. text_completion or a single chunk: not a chat.completion response
        if not _is_completion(body):
            detail = f"not a chat.completion: object={body.get('object')!r}"
            return None, failed(transport_end, "failed", "unexpected_body", False, detail), None
        usage = _to_usage(body["usage"]) if isinstance(body.get("usage"), dict) else None
        return body, _judge_choices(transport_end, body), usage

    def usage(self, events: Sequence[SseEvent]) -> Usage | None:
        raw = _assemble(events).usage
        return _to_usage(raw) if isinstance(raw, dict) else None


@dataclass
class _Assembly:
    top: dict[str, Any] = field(default_factory=dict)
    choices: dict[int, dict[str, Any]] = field(default_factory=dict)
    # choice -> call index -> call
    calls: dict[int, dict[int, dict[str, Any]]] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def completion(self) -> dict[str, Any] | None:
        if not self.top and not self.choices:
            return None
        choices = []
        for i in sorted(self.choices):
            c = self.choices[i]
            msg = {"role": "assistant", "content": None, **c["message"]}
            if self.calls.get(i):
                msg["tool_calls"] = [self.calls[i][k] for k in sorted(self.calls[i])]
            choices.append({**c, "message": msg})
        out = {**self.top, "object": "chat.completion", "choices": choices}
        if self.usage is not None:
            out["usage"] = self.usage
        return out


def _assemble(events: Sequence[SseEvent]) -> _Assembly:
    a = _Assembly()
    for ev in events:
        if ev.data.strip() == DONE:
            a.done = True
            continue
        d = parse_json(ev.data)
        if d is None:
            continue
        if isinstance(d.get("error"), dict):
            a.errors.append(d["error"])
            continue
        _merge_top(a.top, d)
        if isinstance(d.get("usage"), dict):
            a.usage = d["usage"]
        for ch in d.get("choices") or []:
            _merge_choice(a, ch)
    return a


def _merge_top(top: dict[str, Any], chunk: dict[str, Any]) -> None:
    for k, v in chunk.items():
        if k in ("choices", "usage", "object"):
            continue
        if k not in top or (_empty(top[k]) and not _empty(v)):
            top[k] = v


def _merge_choice(a: _Assembly, ch: dict[str, Any]) -> None:
    idx = int(ch.get("index", 0))
    c = a.choices.setdefault(idx, {"index": idx, "message": {}, "finish_reason": None})
    calls = a.calls.setdefault(idx, {})
    for k, v in (ch.get("delta") or {}).items():
        if k == "tool_calls":
            for tc in v or []:
                _merge_call(calls, tc)
        else:
            _merge_value(c["message"], k, v)
    if ch.get("finish_reason") is not None:
        c["finish_reason"] = ch["finish_reason"]
    for k, v in ch.items():
        if k in ("index", "delta", "finish_reason"):
            continue
        if k == "logprobs" and isinstance(v, dict):
            lp = c["logprobs"] if isinstance(c.get("logprobs"), dict) else {}
            for kk, vv in v.items():
                lp[kk] = [*(lp.get(kk) or []), *vv] if isinstance(vv, list) else lp.get(kk, vv)
            c["logprobs"] = lp
        elif k not in c or (_empty(c[k]) and not _empty(v)):
            c[k] = v


def _call_slot(calls: dict[int, dict[str, Any]], tc: dict[str, Any]) -> int:
    """Which call this fragment belongs to. If it has an id, match by id first (some compatible
    services omit index, and fragments may arrive interleaved); a fragment with neither index nor
    id is appended to the last call."""
    idx, cid = tc.get("index"), tc.get("id")
    if cid:
        same = [i for i, c in calls.items() if c.get("id") == cid]
        if same:
            return same[0]
        if not isinstance(idx, int) or calls.get(idx, {}).get("id") not in (None, cid):
            return max(calls, default=-1) + 1  # new call
        return idx
    return idx if isinstance(idx, int) else max(calls, default=0)


def _merge_call(calls: dict[int, dict[str, Any]], tc: dict[str, Any]) -> None:
    cur = calls.setdefault(_call_slot(calls, tc), {})
    for k, v in tc.items():
        if k == "index" or v is None:
            continue
        if k in CALL_PARTS and isinstance(v, dict):
            part = cur.setdefault(k, {})
            for kk, vv in v.items():
                if kk in APPENDED and isinstance(vv, str):
                    part[kk] = part.get(kk, "") + vv
                elif vv is not None:
                    part[kk] = vv
        else:
            cur[k] = v


def _merge_value(msg: dict[str, Any], key: str, value: Any) -> None:
    """Concatenate strings chunk by chunk (except identity fields), append lists, and merge dicts
    recursively (e.g. audio data and transcript arrive in chunks, id only in the first chunk;
    legacy function_call arguments arrive in chunks)."""
    if key in IDENTITY:
        if value:
            msg[key] = value
    elif isinstance(value, str):
        msg[key] = (msg.get(key) or "") + value
    elif isinstance(value, list):
        msg.setdefault(key, []).extend(value)
    elif isinstance(value, dict):
        cur = msg[key] if isinstance(msg.get(key), dict) else {}
        for k, v in value.items():
            _merge_value(cur, k, v)
        msg[key] = cur
    elif value is None:
        msg.setdefault(key, None)
    else:
        msg[key] = value


def _judge_choices(transport_end: str, completion: Mapping[str, Any]) -> Judgement:
    choices = completion.get("choices")
    if not isinstance(choices, list):
        return failed(
            transport_end, "failed", "unexpected_body", False, "response has no choices list"
        )
    if not choices:  # only empty shell chunks and [DONE], or empty choices: no answer was produced
        return failed(transport_end, "incomplete", "no_choices", True, "response has no choices")
    stop = choices[0].get("finish_reason")
    flags: list[str] = []
    problems: list[str] = []
    for ch in choices:
        bad, truncated = _bad_arguments(ch)
        if truncated:
            # when max tokens is hit the last call's arguments are incomplete; this is normal
            flags.append("truncated_tool_input")
        if bad:
            problems.append(f"arguments of {bad} in choice {ch.get('index')} are not valid JSON")
    ok = not problems
    outcome = Outcome(
        ok, transport_end, "completed", stop, None if ok else "; ".join(problems), flags
    )
    return Judgement(outcome, retryable=False, reason="completed" if ok else "bad_assembly")


def _bad_arguments(ch: Mapping[str, Any]) -> tuple[list[str], bool]:
    """Calls whose arguments are not valid JSON (including legacy function_call); when
    finish_reason=length, the last incomplete one is not treated as an error."""
    msg = ch.get("message") or {}
    pieces = [
        (f"tool call {i}", tc["function"].get("arguments"))
        for i, tc in enumerate(msg.get("tool_calls") or [])
        if isinstance(tc.get("function"), dict)
    ]
    if isinstance(msg.get("function_call"), dict):
        pieces.append(("function_call", msg["function_call"].get("arguments")))
    bad = [i for i, (_, args) in enumerate(pieces) if not _is_json(args)]
    truncated = bool(bad) and ch.get("finish_reason") == "length" and bad == [len(pieces) - 1]
    return ([] if truncated else [pieces[i][0] for i in bad]), truncated


def _is_completion(body: Mapping[str, Any]) -> bool:
    choices = body.get("choices")
    return body.get("object", "chat.completion") == "chat.completion" and (
        not isinstance(choices, list)
        or all(isinstance(c, dict) and isinstance(c.get("message"), dict) for c in choices)
    )


def _http_error(transport_end: str, status: int, body: str | None) -> Judgement:
    err = (parse_json(body or "") or {}).get("error")
    reason = _code(err) if isinstance(err, dict) else None
    retryable = status in (408, 429) or status >= 500
    return failed(
        transport_end, "failed", reason or f"http_{status}", retryable, (body or "")[:500]
    )


def _error_judgement(transport_end: str, err: Mapping[str, Any]) -> Judgement:
    code = _code(err) or "error"
    return failed(
        transport_end,
        "failed",
        code,
        code in RETRYABLE_CODES,
        json.dumps(err, ensure_ascii=False)[:500],
    )


def _to_usage(raw: dict[str, Any]) -> Usage:
    """prompt_tokens includes cache reads and writes; subtract them to get tokens not served from
    cache. completion_tokens includes thinking tokens."""
    details = raw.get("prompt_tokens_details") or {}
    # prompt_cache_hit_tokens is DeepSeek's field name
    cached = details.get("cached_tokens") or raw.get("prompt_cache_hit_tokens") or 0
    write = details.get("cache_write_tokens") or 0
    ttl = details.get("cache_write_tokens_details") or {}
    by_ttl = {
        label: n
        for label, n in (("30m", ttl.get("ttl_30m_tokens")), ("12h", ttl.get("ttl_12h_tokens")))
        if n
    }
    return Usage(
        input_tokens=max((raw.get("prompt_tokens") or 0) - cached - write, 0),
        cache_read_tokens=cached,
        cache_write_tokens=write,
        cache_write_by_ttl=by_ttl,
        output_tokens=raw.get("completion_tokens") or 0,
        thinking_tokens=(raw.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        raw=raw,
    )


def _code(err: Mapping[str, Any]) -> str | None:
    value = err.get("code") or err.get("type")
    return str(value) if value else None


def _empty(value: Any) -> bool:
    return value in (None, "", 0, [], {})


def _is_json(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True
