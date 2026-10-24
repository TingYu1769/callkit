"""Command-line entry: ``callkit proxy`` starts the recording proxy; ``ls / show / tree / export`` inspect the run directory (read only)."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from callkit.browse import Node, Row
    from callkit.record import Record
    from callkit.store import RunStore

API_SHORT = {
    "anthropic.messages": "messages",
    "openai.responses": "responses",
    "openai.chat_completions": "chat",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="callkit", description="callkit command line")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "proxy",
        help="run a local recording proxy: relay byte for byte to the upstream, one record per model call",
    )
    p.add_argument(
        "--upstream", required=True, help="upstream base URL without /v1, e.g. http://host:port"
    )
    p.add_argument("--run", required=True, help="run directory to write records into")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="port to listen on; 0 picks a free port")
    p.add_argument(
        "--upstream-proxy",
        default=None,
        help="proxy for reaching the upstream, e.g. http://proxy.example:8080",
    )
    p.add_argument(
        "--prices", default=None, help="price file (JSON: model -> USD per million tokens)"
    )
    p.add_argument(
        "--cap",
        type=float,
        default=None,
        help="spending cap in USD; once reached, requests get HTTP 402 and are not forwarded",
    )
    p = sub.add_parser("ls", help="list the calls in a run: outcome, tokens, cost, and totals")
    p.add_argument("run", help="run directory")
    p.add_argument("--failed", action="store_true", help="only list failed calls")
    p.add_argument(
        "--limit", type=int, default=None, help="only list the last N calls (totals cover all)"
    )
    p = sub.add_parser("show", help="show one call: outcome, usage, request and answer summary")
    p.add_argument("run", help="run directory")
    p.add_argument("call_id", help="call id, e.g. c_000007")
    p.add_argument("--raw", action="store_true", help="print the assembled answer as JSON")
    p = sub.add_parser(
        "tree", help="print parent / fork_of links as a tree per session, to pick fork points"
    )
    p.add_argument("run", help="run directory")
    p.add_argument("--session", default=None, help="only show this session")
    p = sub.add_parser(
        "export", help="export JSONL: one call per line with the full request body and answer"
    )
    p.add_argument("run", help="run directory")
    p.add_argument("-o", "--output", default=None, help="output file; defaults to stdout")
    p.add_argument("--ok-only", action="store_true", help="only export successful calls")
    args = parser.parse_args(argv)
    commands = {"proxy": _proxy, "ls": _ls, "show": _show, "tree": _tree, "export": _export}
    return commands[args.command](args)


def _proxy(args: argparse.Namespace) -> int:
    from callkit.pricing import load_prices
    from callkit.proxy import Proxy

    proxy = Proxy(
        args.upstream,
        args.run,
        host=args.host,
        port=args.port,
        upstream_proxy=args.upstream_proxy,
        prices=load_prices(args.prices) if args.prices else None,
        cap_usd=args.cap,
    )
    print(
        f"callkit proxy listening on {proxy.url} -> {proxy.upstream}, recording to {proxy.store.dir}",
        file=sys.stderr,
    )
    print(f"Claude Code: ANTHROPIC_BASE_URL={proxy.url} claude ...", file=sys.stderr)
    provider = f'model_providers.callkit={{name="callkit",base_url="{proxy.url}/v1",wire_api="responses",env_key="<KEY_ENV_VAR>"}}'
    print(
        f"Codex (with a fresh CODEX_HOME): codex -c '{provider}' -c model_provider=callkit ...",
        file=sys.stderr,
    )
    try:
        proxy.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proxy.stop()
    return 0


def _open(run: str) -> RunStore:
    from callkit.store import RunStore

    path = Path(run)
    if not (path / "run.json").exists():
        sys.exit(f"{run} is not a callkit run dir (no run.json)")
    # Read only: do not create calls/ or write run.json
    return RunStore(path, create=False)


def _ls(args: argparse.Namespace) -> int:
    from callkit import browse

    rows = [browse.row_of(rec) for rec in browse.records(_open(args.run))]
    shown = [r for r in rows if not (args.failed and r.ok)]
    if args.limit:
        shown = shown[-args.limit :]
    left = [("call", 9), ("started (UTC)", 16), ("api", 10), ("model", 22), ("result", 22)]
    right = [("input", 8), ("output", 7), ("cost", 10)]
    print(
        "".join(_pad(h, w) for h, w in left)
        + "".join(_pad(h, w, right=True) for h, w in right)
        + "  links"
    )
    for r in shown:
        print(_ls_line(r))
    t = browse.totals(rows)
    print(
        f"{_calls(t['calls'])}, {t['ok']} ok, {t['calls'] - t['ok']} failed; "
        f"input {t['input_tokens']:,} tokens (cache read {t['cache_read_tokens']:,}), "
        f"output {t['output_tokens']:,} tokens; {_cost_text(t)}"
    )
    if len(t["by_model"]) > 1:
        parts = [
            f"{m or '?'}: {_calls(v['calls'])}, input {v['input_tokens']:,}, "
            f"output {v['output_tokens']:,} tokens, {_cost_text(v)}"
            for m, v in t["by_model"].items()
        ]
        print("by model: " + "; ".join(parts))
    return 0


def _calls(n: int) -> str:
    return f"{n} call" if n == 1 else f"{n} calls"


def _cost_text(t: dict[str, Any]) -> str:
    """Report unpriced calls and successful calls without usage separately."""
    if t["unpriced"] == t["calls"]:
        return "cost not tracked (no prices)"
    gaps = [
        f"{t['unpriced']} without a price" if t["unpriced"] else "",
        f"{t['usage_unknown']} without usage" if t["usage_unknown"] else "",
    ]
    extra = ", ".join(g for g in gaps if g)
    return f"cost ${t['cost_usd']:.4f}" + (f" (not counted: {extra})" if extra else "")


def _pad(text: str, width: int, *, right: bool = False) -> str:
    """Pad spaces by terminal display width: Chinese and fullwidth characters occupy two columns."""
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    space = " " * max(width - shown, 0)
    return space + text if right else text + space


def _ls_line(r: Row) -> str:
    result = ("ok " + (r.stop or "")) if r.ok else ("✗ " + r.reason)
    links = " ".join(
        x for x in (r.parent and f"←{r.parent}", r.fork_of and f"⑂{r.fork_of}", r.item) if x
    )
    cost = "-" if r.cost_usd is None else f"${r.cost_usd:.4f}"
    when = r.started_at[5:19].replace("T", " ")
    return (
        f"{r.id:<9}{when:<16}{API_SHORT.get(r.api, r.api):<10}{r.model[:21]:<22}{result[:21]:<22}"
        f"{r.input_tokens:>8,}{r.output_tokens:>7,}{cost:>10}  {links}"
    ).rstrip()


def _show(args: argparse.Namespace) -> int:
    from callkit import browse

    try:
        rec = _open(args.run).load(args.call_id)
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f"cannot read {args.call_id}: {type(e).__name__}: {e}")
    if args.raw:
        print(json.dumps(rec.message, ensure_ascii=False, indent=2))
        return 0
    r, a, o = browse.row_of(rec), browse.answer(rec), rec.outcome
    u = rec.usage
    print(f"{rec.id}  {rec.api}  source {rec.source}  {rec.timing.started_at}")
    print(f"url       {rec.request.url}  status {rec.response.status}")
    print(
        f"outcome   {'ok' if o.ok else 'failed'}  stop={o.stop_reason}  transport={o.transport_end}  protocol={o.protocol}  flags={o.flags}"
    )
    if o.error:
        print(f"error     {browse.clip(o.error, 200)}")
    print(
        f"timing    first byte {rec.timing.first_byte_ms} ms, total {rec.timing.total_ms} ms, {len(rec.attempts)} attempt(s)"
    )
    if u:
        print(
            f"usage     input {u.input_tokens}, cache read {u.cache_read_tokens}, cache write {u.cache_write_tokens}, output {u.output_tokens} (thinking {u.thinking_tokens})"
        )
    print(f"cost      {'-' if rec.cost_usd is None else f'${rec.cost_usd:.6f}'}")
    print(
        f"links     parent={r.parent or '-'}  fork_of={r.fork_of or '-'}  item={r.item or '-'}  tags={rec.tags}"
    )
    print(
        f"request   model {r.model}; tools {', '.join(browse.tool_names(rec)) or 'none'}; last user input: {browse.last_user(rec)}"
    )
    print(f"answer    {browse.clip(a.text, 400) or '(no text)'}")
    if a.thinking_chars:
        print(f"thinking  {a.thinking_chars} chars")
    for call in a.tool_calls:
        print(f"tool call {call}")
    return 0


def _tree(args: argparse.Namespace) -> int:
    from callkit import browse

    recs = list(browse.records(_open(args.run)))
    labels = {rec.id: _label(rec) for rec in recs}
    forest = browse.tree([browse.row_of(rec) for rec in recs])
    for session, roots in forest.items():
        if args.session is not None and session != args.session:
            continue
        print(f"session {session or '(none)'}")
        for root in roots:
            print(labels[root.row.id])
            _print_children(root, labels, "")
    return 0


def _label(rec: Record) -> str:
    from callkit import browse

    r, a = browse.row_of(rec), browse.answer(rec)
    said = "; ".join(a.tool_calls) if a.tool_calls else browse.clip(a.text, 50)
    state = (r.stop or "") if r.ok else f"✗ {r.reason}"
    return f"{r.id}  {r.model}  {state}  {said}"


def _print_children(node: Node, labels: dict[str, str], prefix: str) -> None:
    """Print with an explicit stack so long continuation chains do not overflow recursion."""
    stack = [(child, prefix, i == len(node.children) - 1) for i, child in enumerate(node.children)]
    stack.reverse()
    while stack:
        child, pre, last = stack.pop()
        print(f"{pre}{'└' if last else '├'}{child.how} {labels[child.row.id]}")
        deeper = pre + ("   " if last else "│  ")
        kids = child.children
        stack.extend((k, deeper, i == len(kids) - 1) for i, k in reversed(list(enumerate(kids))))


def _export(args: argparse.Namespace) -> int:
    from callkit import browse

    store = _open(args.run)
    if args.output is None:
        n = browse.export(store, sys.stdout, ok_only=args.ok_only)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            n = browse.export(store, f, ok_only=args.ok_only)
    print(f"exported {_calls(n)}", file=sys.stderr)
    return 0
