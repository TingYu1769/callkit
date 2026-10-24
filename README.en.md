# callkit

[![CI](https://github.com/TingYu1769/callkit/actions/workflows/ci.yml/badge.svg)](https://github.com/TingYu1769/callkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/TingYu1769/callkit/blob/main/LICENSE)

[中文](https://github.com/TingYu1769/callkit/blob/main/README.md) | English

> The Chinese README is the primary one; this translation may lag behind it.

callkit is a Python library that records LLM calls one API call at a time, for researchers running LLM and agent experiments. Every call keeps the raw request body and the raw event stream, plus the assembled answer, usage and cost. Records can be spliced into multi-turn conversations, and any call can be forked and sent again.

Unlike observability platforms and HTTP cassette tools, callkit works without changing your existing calling code. It mounts under the official openai and anthropic SDKs, or records real Claude Code and Codex traffic through a local proxy, and it keeps everything in local files with no database or server. It can also cap spending from a price table and, when you rerun an experiment, answer identical requests from earlier recordings. It speaks Anthropic Messages, OpenAI Responses and OpenAI Chat Completions, and services compatible with them (such as DeepSeek).

## Quickstart

```bash
pip install git+https://github.com/TingYu1769/callkit
```

Use the official SDK as usual and pass one extra `http_client`:

```python
import callkit
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="...",
    http_client=callkit.recording_http_client("runs/demo"),
)
client.chat.completions.create(model="deepseek-flash", messages=[{"role": "user", "content": "Explain SSE in one sentence."}])
```

Each call gets its own folder in the run directory:

```
runs/demo/
├── run.json          format name, creation time (plus endpoint config and callkit version with callkit's Client)
├── index.jsonl       one line per call: id, outcome, cost, parent, fork_of
└── calls/c_000001/
    ├── meta.json     the record: outcome, usage, timing, tags
    ├── request.json  the request body, byte for byte
    ├── response.sse  the raw event stream (non-streaming responses go to response.body)
    └── message.json  the assembled answer
```

Look at it from the command line:

```
$ callkit ls runs/demo
call     started (UTC)   api       model                 result                   input output      cost  links
c_000001 10-24 09:30:02  chat      deepseek-flash        ok stop                     14     18         -
1 call, 1 ok, 0 failed; input 14 tokens (cache read 0), output 18 tokens; cost not tracked (no prices)
```

No price table was given, so the cost column shows `-`; see [Pricing and caps](#pricing-and-caps) for how to add prices.

## Highlights

- **Verbatim records.** Request bodies are stored as bytes and event streams as their raw lines; streams recorded by the SDK mount and the proxy can be restored byte for byte. The assembled answer is stored next to them for easy reading.
- **Two-layer outcome.** The transport layer records how the connection ended (done, eof, timeout, connect_error, client_gone); the protocol layer records whether the exchange completed (completed, failed, incomplete). Stop reasons are kept in each API's own words.
- **Splice and fork.** `extend` builds the next request body from the last request and answer, carrying thinking signatures, encrypted reasoning and `call_id`s back verbatim; `fork` resends any call with a different model or parameters.
- **Three ways in.** SDK mounting (sync and async, httpx and httpx2), a recording proxy, and callkit's own `Client`. All three produce the same record format.
- **Pricing and caps.** Give a price table and every call gets a cost. With a cap, requests stop once the run directory's total reaches it. Several processes writing one directory share one ledger.
- **Replay cache.** The k-th occurrence of a request gets the k-th successful record from an earlier run, so a rerun of a sampling experiment gets back the same distinct samples.
- **Scheduler.** `run_items` / `arun_items` wrap any "process one item" function with concurrency, resume and a spending cap, and tag each record with its item.
- **CLI.** `ls`, `show`, `tree` and `export` list calls, show one call, draw conversation trees and export JSONL.

## Installation

Python 3.12 or newer. Tested on Linux only. Cross-process bookkeeping uses `fcntl`, so Windows is not supported.

```bash
pip install git+https://github.com/TingYu1769/callkit
# or
uv add git+https://github.com/TingYu1769/callkit
```

The only runtime dependency is httpx. Mounting under the anthropic SDK 2.x needs httpx2, which comes with the anthropic SDK.

## Usage

### Mount under the official SDKs

```python
import callkit
from anthropic import Anthropic
from openai import AsyncOpenAI, OpenAI

RUN = "runs/E1"     # set to None to disable recording; the SDK then works as usual

client = OpenAI(..., http_client=callkit.recording_http_client(RUN))
aclient = AsyncOpenAI(..., http_client=callkit.recording_async_http_client(RUN))
claude = Anthropic(..., http_client=callkit.recording_http_client(RUN, lib="httpx2"))
```

- Only POST requests whose path ends with `/v1/messages`, `/responses` or `/chat/completions` are recorded; other requests (embeddings, models, …) pass through. Streaming and non-streaming responses are both recorded.
- Each SDK retry gets its own record, with the retry number in the `retry_count` tag. Failures without a response (connection errors, timeouts, cancellation) are recorded too, then raised as usual.
- If recording itself fails (say, the disk is full), callkit only emits a RuntimeWarning; the SDK call is not affected.
- Proxy environment variables are ignored; pass `proxy="http://..."` to use a proxy. Options such as `verify`, `limits` and `http2` work as usual and are handed to the inner transport.
- `session="..."` names a session (parents are inferred within a session only); `tags={...}` are merged into every record.

### Use callkit's own client

```python
from callkit import Client, Endpoint, Price

ep = Endpoint("anthropic", "https://api.anthropic.com", key_env="ANTHROPIC_API_KEY")
prices = {"claude-sonnet-5-5": Price(input=3, output=15, cache_read=0.3, cache_write=3.75)}  # USD per million tokens, illustrative

with Client(ep, "runs/E2", prices=prices, cap_usd=5.0) as c:
    rec = c.call({"model": "claude-sonnet-5-5", "max_tokens": 1000,
                  "messages": [{"role": "user", "content": "Hello"}]})

rec.outcome.ok, rec.outcome.stop_reason, rec.cost_usd
```

- Keys are read from the environment variable named by `key_env` and never written to records. Auth headers are redacted in records, including a custom `auth_header`.
- `Endpoint.api` picks the API: `"anthropic.messages"` (default), `"openai.responses"` or `"openai.chat_completions"`. `base_url` has no `/v1`. If a service lives under a path prefix, put the prefix in `base_url`, e.g. `https://api.deepseek.com/anthropic` for DeepSeek's Anthropic-compatible API.
- `Client` only sends streaming requests. If the body has no `stream`, callkit adds it (and `stream_options.include_usage` for Chat).
- Failed calls do not raise; check `rec.outcome`. Dropped streams, 429 and 5xx are retried (3 attempts in total by default), and every attempt is recorded and billed. The spending cap is the exception; see [Pricing and caps](#pricing-and-caps).
- `c.batch(items, concurrency=8)` runs a batch concurrently and yields in completion order; `(item_id, repeat)` pairs that already succeeded are skipped next time.

### Splice into conversations

```python
from callkit import extend, pending_calls

first = c.call(body)                                   # the model calls tools
results = {i: "sunny, 25°C" for i in pending_calls(first)}
second = c.call(extend(first, tool_results=results), parent=first)
third = c.call(extend(second, text="And Shanghai?"), parent=second)
```

`extend` makes no network calls; it only builds the next request body from the last request, this answer and the tool results, carrying the answer back verbatim, including thinking signatures, encrypted reasoning and `call_id`s. Mismatched tool results, or a request that would end with an assistant message, raise `PairingError` before anything is sent. Anthropic's `pause_turn` is the exception: such an answer is sent back as is, without tool results or text, so the model can continue.

### Fork and replay

```python
from callkit import fork

alt = c.call(fork(second, model="claude-opus-5-5"), fork_of=second)
alt = c.call(fork(second, edit=lambda b: {**b, "max_tokens": 500}), fork_of=second)
```

- `fork` keeps key order when editing the body; the new record's `edits` lists the changed top-level fields. Forks reuse the original call's query string and headers (such as Claude Code's `?beta=true` and beta headers); auth comes from the new endpoint.
- After a model change or edited history, earlier thinking blocks may no longer be valid. callkit keeps them and adds a `thinking_may_be_invalid:N` flag to `rec.outcome.flags`.
- If the original request had `stream: false` (common for calls recorded from an SDK), `fork` and `extend` switch it to streaming.

### Record Claude Code and Codex

```bash
callkit proxy --upstream https://api.anthropic.com --run runs/cc-001   # prints the address, e.g. http://127.0.0.1:41235
ANTHROPIC_BASE_URL=http://127.0.0.1:41235 claude -p "..."
```

The proxy forwards requests to `--upstream` as they are and does not convert between APIs, so Codex needs its own proxy whose upstream serves the Responses API. Codex also needs a fresh `CODEX_HOME` and a custom provider pointing at the proxy (the provider must not be named `OpenAI`):

```bash
callkit proxy --upstream https://api.openai.com --run runs/codex-001   # e.g. http://127.0.0.1:41236
CODEX_HOME=/tmp/codex-home MY_KEY=... codex exec \
  -c 'model_providers.callkit={name="callkit",base_url="http://127.0.0.1:41236/v1",wire_api="responses",env_key="MY_KEY"}' \
  -c model_provider=callkit "..."
```

- The proxy relays chunks unchanged; it only asks the upstream not to compress.
- WebSocket upgrade requests get 426; Codex then falls back to HTTP.
- Records are grouped by session, taken from `X-Claude-Code-Session-Id`, then `session-id`, then the body's `prompt_cache_key`. Parents are inferred when the context matches and "the candidate's request plus its answer" is a prefix of this request; ambiguous matches go to `parent_candidates`.
- Any recorded call can be forked and resent.

### Pricing and caps

```python
http = callkit.recording_http_client("runs/E3", prices={"deepseek-flash": Price(...)}, cap_usd=5.0)
```

```bash
callkit proxy --upstream ... --run runs/E3 --prices prices.json --cap 5
```

- With `cap_usd`, once the run directory's total spending reaches the cap, requests are not sent. The SDK mount and the proxy answer HTTP 402 with `x-should-retry: false`, which the openai and anthropic SDKs do not retry; requests for a model with no price are stopped the same way. callkit's own `Client` raises instead: `CapReached` at the cap, `MissingPrice` for a model with no price.
- Prices are merged into the run directory's `prices.json` per model; passing a different price for the same model later is an error.
- A successful call without usage (e.g. streaming Chat without `stream_options={"include_usage": True}`) gets an unknown cost, a `usage_missing` flag and a one-time warning.
- Codex retries a 402 from the proxy a few times; each retry is stopped too and costs nothing.

### Replay cache

```python
with Client(ep, "runs/E1-rerun", replay_from="runs/E1") as c: ...
client = OpenAI(..., http_client=callkit.recording_http_client("runs/E1-rerun", replay_from="runs/E1"))
```

- Two requests are the same when the URL, the body, the `anthropic-version` and `anthropic-beta` headers and three scoping headers (`openai-organization`, `openai-project`, `anthropic-workspace-id`) all match. The official SDKs send byte-identical bodies within one SDK version, so they hit; after an SDK upgrade the key order may change and they won't.
- The k-th occurrence of a request gets the k-th successful record from the old run; once those run out, calls go to the network. Within one process, clients replaying from the same old run into the same new run share the count; a new process starts counting from the beginning.
- A hit is still recorded in the new run, with a `replay_of` tag pointing at the old record and a cost of 0.

### Scheduler

```python
def ask(question):                     # process one item; call the SDK as often as you like
    return client.chat.completions.create(model=..., messages=[{"role": "user", "content": question}])

for r in callkit.run_items(questions, ask, run="runs/E4", concurrency=8, cap_usd=5):
    r.item_id, r.ok, r.value, r.error, r.call_ids
```

- `questions` maps item ids to arguments; `repeats=N` runs each item N times. `arun_items` takes an async function.
- Each finished item gets a line in the run directory's `items.jsonl`; reruns skip items that already succeeded.
- Calls made inside the function through a recording client or `Client` are tagged with the item id; `r.call_ids` lists them.
- An exception fails only that item. To stop early and clean up at once, wrap the generator in `contextlib.closing(run_items(...))` (`contextlib.aclosing` for the async version).

## Command line

```bash
callkit ls RUN [--failed] [--limit N]     # list calls, with token and cost totals
callkit show RUN CALL_ID [--raw]          # one call: outcome, timing, usage, request and answer summary
callkit tree RUN [--session S]            # continuations (→) and forks (⑂) as a tree per session
callkit export RUN [-o FILE] [--ok-only]  # JSONL, one call per line with full request body and answer
callkit proxy --upstream URL --run RUN    # run the recording proxy
```

```
$ callkit tree runs/weather
session demo
c_000001  deepseek-flash  tool_calls  get_weather({"city": "Beijing"})
└→ c_000002  deepseek-flash  stop  Sunny in Beijing today, about 25°C.
```

## Storage format

One folder per call; `meta.json` is the record, in format `callkit.call/2`. Main fields:

| Field | Content |
|---|---|
| `api`, `source` | API name; where the record came from: `self` (callkit's Client), `proxy`, `sdk` |
| `request` | URL, headers (secrets redacted), body file name, request hash |
| `response` | status, headers, event stream file, answer file |
| `outcome` | `ok`, `transport_end`, `protocol`, `stop_reason`, `error`, `flags` |
| `usage` | uncached input, cache reads, cache writes (by TTL), output, thinking, and the raw usage |
| `attempts` | status, ending, usage and cost of every attempt |
| `parent`, `fork_of`, `edits` | splice and fork links; links into another run dir are written `dir:call_id` |
| `item_id`, `repeat`, `tags` | batch and scheduler keys, plus custom tags |

Load a record with `RunStore("runs/E1").load("c_000001")`; it comes with its request body and answer and can go straight into `extend` or `fork`. Older formats stay readable: `/1` records are upgraded to `/2` on load.

## How it works

```
official SDK ──http_client──▶ recording transport ─┐
                                                   ├──▶ Recorder ──┐
Claude Code / Codex ──▶ recording proxy ───────────┘               ├──▶ adapters (assemble, judge, usage) ──▶ RunStore (disk)
callkit Client ──▶ transport (retries) ────────────────────────────┘
```

The SDK mount and the proxy share one Recorder; callkit's Client builds its own records. All three paths use the same adapters and the same RunStore, so records look the same. Adapters are pure transformations; only the transport, the proxy and the SDK mount touch the network.

## Compared with similar tools

| Tool | Unit of record | How it hooks in | Raw bytes | Local files only | Splice / fork | Spending cap |
|---|---|---|---|---|---|---|
| callkit | one API call | SDK mount, proxy, own client | yes | yes | yes | yes |
| [llm](https://github.com/simonw/llm) | one conversation turn | own client and plugins | response JSON | yes (SQLite) | continue and fork threads | no |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | one evaluation | own model layer | first few calls per model and all failed calls by default; full logging optional | yes | `fork()` copies state to run several solvers | yes (per sample, needs model prices) |
| [LiteLLM](https://github.com/BerriAI/litellm) | gateway logs | gateway, SDK | no | gateway can run without a database; spend tracking and budgets need one | cache | yes (budgets) |
| [Langfuse](https://github.com/langfuse/langfuse), [Phoenix](https://github.com/Arize-ai/phoenix) | traces | OpenTelemetry | no | needs a server | replay in playground | no |
| [vcrpy](https://github.com/kevin1024/vcrpy) | HTTP exchange | patches HTTP libraries | yes | yes | replay cassettes | no |
| [llm-cassette](https://github.com/aozmekik/llm-cassette), [retake](https://github.com/Zecrari/retake) | HTTP exchange | httpx transport, proxy | yes | yes | replay cassettes | no |
| [claude-trace](https://github.com/badlogic/lemmy/tree/main/apps/claude-trace) | Claude Code session | injected interceptor | yes | yes | view only | no |

Based on each project's documentation as of October 2026. Cassette tools such as vcrpy target tests and make excellent test doubles, but do not splice or fork and do not cap spending (llm-cassette can report estimated cost from a price table). Observability platforms suit production services; callkit suits offline experiments that need the raw text of every call.

## Limitations

- WebSocket traffic is not recorded (e.g. the Responses WebSocket mode); the proxy answers WebSocket upgrades with 426.
- A streaming response that the caller reads halfway and never closes may not be recorded.
- With several choices (`n>1`), `extend` continues choice 0 only.
- Streams recorded through a relay gateway may already have been rewritten by it, so they are not the upstream's original bytes.
- Event streams do not keep per-line arrival times, only time to first byte and total time. callkit's `Client` reads streams line by line and stores line endings as `\n`.
- Windows is not supported (`fcntl`).

## Contributing

See [CONTRIBUTING.md](https://github.com/TingYu1769/callkit/blob/main/CONTRIBUTING.md). Offline tests cost nothing and `make check` runs them all; tests that call real APIs are marked `network` and `make check` skips them.

## Citation

If callkit helps your research, please cite the repository; GitHub's "Cite this repository" button generates the format from [CITATION.cff](https://github.com/TingYu1769/callkit/blob/main/CITATION.cff).

## License

[MIT](https://github.com/TingYu1769/callkit/blob/main/LICENSE)
