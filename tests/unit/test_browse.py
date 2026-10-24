"""Inspect a run directory: browse pure functions and callkit ls / show / tree / export. Create a few records for each of the three APIs."""

import json

import pytest

from callkit import Client, Endpoint, Price, browse, extend, fork, pending_calls, transport
from callkit.cli import main
from callkit.store import RunStore
from tests.fakes import DATA, Server, error_response, good_stream, sse_response

PRICE = Price(input=3, output=15, cache_read=0.3, cache_write=3.75)
TOOL = {"name": "get_weather", "input_schema": {"type": "object"}}
ASK = [{"role": "user", "content": "北京今天天气怎么样？"}]


@pytest.fixture
def run(monkeypatch, tmp_path):
    """A run directory: three Anthropic calls (first turn, continuation, fork) plus one failure, one Responses call, and one Chat call."""
    monkeypatch.setenv("CALLKIT_TEST_KEY", "sk-test")
    monkeypatch.setattr(transport.time, "sleep", lambda s: None)
    s = Server(
        sse_response(good_stream()),
        sse_response(good_stream()),
        sse_response(good_stream()),
        error_response(400, "invalid_request_error"),
        sse_response((DATA / "responses_tool_calls.sse").read_bytes()),
        sse_response((DATA / "chat_tool_calls.sse").read_bytes()),
    )
    monkeypatch.setattr(transport, "open_client", lambda endpoint: s.client())
    path = tmp_path / "run"
    body = {"model": "claude-sonnet-5-5", "max_tokens": 100, "tools": [TOOL], "messages": ASK}
    with Client(
        Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY"), path, prices={"claude-sonnet-5-5": PRICE}
    ) as c:
        first = c.call(body, tags={"session": "s1"})
        second = c.call(
            extend(first, tool_results={i: "晴" for i in pending_calls(first)}), parent=first
        )
        c.call(fork(second, model="claude-opus-5-5"), fork_of=second)
        c.call({**body, "max_tokens": 1})
    oai = {"api": "openai.responses"}
    with Client(Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY", **oai), path) as c:
        c.call(
            {
                "model": "gpt-5.6-sol",
                "input": "查天气",
                "tools": [{"type": "function", "name": "get_weather"}],
            }
        )
    chat = json.loads((DATA / "chat_tool_calls.request.json").read_text(encoding="utf-8"))
    with Client(
        Endpoint("gw", "http://gw", "CALLKIT_TEST_KEY", api="openai.chat_completions"), path
    ) as c:
        c.call(chat)
    return path


def recs(path):
    return {r.id: r for r in browse.records(RunStore(path))}


def test_rows_and_totals(run):
    rows = [browse.row_of(r) for r in recs(run).values()]
    assert [r.api for r in rows] == ["anthropic.messages"] * 4 + [
        "openai.responses",
        "openai.chat_completions",
    ]
    a1, a2, a3, bad = rows[:4]
    assert a1.ok and a1.stop == "tool_use" and a1.session == "s1" and a1.cost_usd
    assert a2.parent == a1.id and a3.fork_of == a2.id and a3.model == "claude-opus-5-5"
    assert not bad.ok and bad.reason == "invalid_request_error"
    t = browse.totals(rows)
    assert (
        t["calls"] == 6 and t["ok"] == 5 and t["unpriced"] == 3
    )  # Only claude-sonnet-5-5 has a price; its 400 response records 0 cost.
    assert t["by_model"]["claude-sonnet-5-5"]["calls"] == 3
    assert (rows[5].input_tokens, rows[5].output_tokens) == (
        139,
        47,
    )  # Chat sample's prompt / completion


def test_answers_and_requests(run):
    r = list(recs(run).values())
    first, second, responses, chat = r[0], r[1], r[4], r[5]
    assert browse.answer(first).tool_calls and browse.last_user(first) == "北京今天天气怎么样？"
    assert browse.last_user(second).startswith("[tool results x") and browse.tool_names(first) == [
        "get_weather"
    ]
    assert (
        len(browse.answer(responses).tool_calls) == 4
        and browse.answer(responses).thinking_chars >= 0
    )
    assert browse.last_user(responses) == "查天气" and browse.tool_names(responses) == [
        "get_weather"
    ]
    a = browse.answer(chat)
    assert [c.split("(")[0] for c in a.tool_calls] == ["get_weather", "get_weather"]
    assert browse.tool_names(chat) == ["get_weather"] and "上海" in browse.last_user(chat)


def test_last_user_counts_chat_tool_messages(run):
    chat = recs(run)["c_000006"]
    raw = json.loads(chat.body)
    reply = chat.message["choices"][0]["message"]
    tools = [
        {"role": "tool", "tool_call_id": tc["id"], "content": "晴"} for tc in reply["tool_calls"]
    ]
    chat.body = json.dumps({**raw, "messages": [*raw["messages"], reply, *tools]}).encode()
    assert browse.last_user(chat) == "[tool results x2]"


def test_tree(run):
    forest = browse.tree([browse.row_of(r) for r in recs(run).values()])
    (root,) = forest["s1"]
    (child,) = root.children
    (grand,) = child.children
    assert child.how == "→" and grand.how == "⑂" and grand.row.fork_of == child.row.id
    # The forked call has no session tag but is attached to the s1 tree; the failed call and the other two API calls are each a root.
    assert len(forest[None]) == 3


def test_cli_ls_show_tree(run, capsys):
    assert main(["ls", str(run)]) == 0
    out = capsys.readouterr().out
    assert (
        "c_000006" in out
        and "6 calls, 5 ok, 1 failed" in out
        and "(not counted: 3 without a price)" in out
    )
    by_model = next(line for line in out.splitlines() if line.startswith("by model: "))
    assert (
        "claude-opus-5-5: 1 call, input " in by_model
        and "tokens, cost not tracked (no prices)" in by_model
    )
    assert "claude-sonnet-5-5: 3 calls, input 29,928, output 92 tokens, cost $" in by_model
    # Lines without a parent call have no trailing spaces.
    assert all(line == line.rstrip() for line in out.splitlines())
    main(["ls", str(run), "--failed"])
    out = capsys.readouterr().out
    assert [line.split()[0] for line in out.splitlines() if line.startswith("c_")] == ["c_000004"]
    main(["show", str(run), "c_000001"])
    out = capsys.readouterr().out
    assert "tool call Write(" in out and "last user input: 北京今天天气怎么样？" in out
    main(["show", str(run), "c_000006", "--raw"])
    assert json.loads(capsys.readouterr().out)["object"] == "chat.completion"
    main(["tree", str(run), "--session", "s1"])
    out = capsys.readouterr().out
    assert out.startswith("session s1") and "└→ c_000002" in out and "└⑂ c_000003" in out


def test_cli_export(run, tmp_path, capsys):
    out = tmp_path / "all.jsonl"
    assert main(["export", str(run), "-o", str(out), "--ok-only"]) == 0
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 5 and all(line["outcome"]["ok"] for line in lines)
    assert lines[0]["request"]["messages"] == ASK and lines[0]["response"]["content"]
    assert lines[1]["parent"] == lines[0]["id"] and lines[0]["usage"]["raw"] is None
    assert "exported 5 calls" in capsys.readouterr().err


def test_cli_rejects_non_run_dir(tmp_path):
    with pytest.raises(SystemExit, match="is not a callkit run dir"):
        main(["ls", str(tmp_path)])
