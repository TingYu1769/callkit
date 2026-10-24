# callkit

[![CI](https://github.com/TingYu1769/callkit/actions/workflows/ci.yml/badge.svg)](https://github.com/TingYu1769/callkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/TingYu1769/callkit/blob/main/LICENSE)

中文 | [English](https://github.com/TingYu1769/callkit/blob/main/README.en.md)

callkit 是一个 Python 库，以“一次 API 调用”为单位记录大模型调用，给做 LLM 和 agent 实验的研究者用。每次调用都原样存下请求体和事件流原文，另存拼好的回答、用量和花费。记录能拼成多轮对话，也能从任意一次调用分叉重发。

和可观测性平台、HTTP 录像带工具不同，callkit 不用改原有的调用代码。它可以挂在官方 openai、anthropic SDK 底下，也可以用本地代理录 Claude Code 和 Codex 的真实流量，记录全在本地文件里，不需要数据库或服务端。它还能按价格表封顶花费，重跑实验时让相同的请求直接拿旧回答。支持 Anthropic Messages、OpenAI Responses、OpenAI Chat Completions 三种接口，以及兼容它们的服务（如 DeepSeek）。

## 30 秒上手

```bash
pip install git+https://github.com/TingYu1769/callkit
```

照常用官方 SDK，只多传一个 `http_client`：

```python
import callkit
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="...",
    http_client=callkit.recording_http_client("runs/demo"),
)
client.chat.completions.create(model="deepseek-flash", messages=[{"role": "user", "content": "用一句话介绍 SSE。"}])
```

每次调用在运行目录里存一个文件夹：

```
runs/demo/
├── run.json          格式名、建立时间（用 callkit 的 Client 时还有端点配置和 callkit 版本）
├── index.jsonl       每次调用一行：调用号、成败、花费、parent、fork_of
└── calls/c_000001/
    ├── meta.json     记录本体：结果、用量、耗时、标签
    ├── request.json  原样的请求体
    ├── response.sse  事件流原文（非流式响应存 response.body）
    └── message.json  拼好的回答
```

用命令行查看：

```
$ callkit ls runs/demo
call     started (UTC)   api       model                 result                   input output      cost  links
c_000001 10-24 09:30:02  chat      deepseek-flash        ok stop                     14     18         -
1 call, 1 ok, 0 failed; input 14 tokens (cache read 0), output 18 tokens; cost not tracked (no prices)
```

没给价格表，所以花费一栏是 `-`；给价格的写法见[计价与封顶](#计价与封顶)。

## 亮点

- **原样保存**：请求体按字节存，事件流按行存原文，SDK 挂载和代理录到的流能逐字节还原。拼好的回答另存一份，方便直接读。
- **结果分两层**：传输层记连接怎么结束（done、eof、timeout、connect_error、client_gone），协议层记有没有走完（completed、failed、incomplete）。停止原因照录接口原词。
- **拼多轮与分叉**：`extend` 用上一次的请求和回答生成下一次的请求体，原样搬回 thinking 签名、加密推理、`call_id`；`fork` 改模型或参数后重发任意一次调用。
- **三种接入方式**：SDK 挂载（同步、异步，httpx 与 httpx2）、录制代理、callkit 自己的 `Client`。三种方式录出的记录格式一样。
- **计价与封顶**：给价格表就算出每次调用的花费。设了上限，运行目录的累计花费到顶后不再发请求。多个进程写同一个目录时共用一本账。
- **回放缓存**：同一请求第 k 次出现，就取旧运行里的第 k 条成功记录。重跑采样实验，拿回的还是原来那几个不同的样本。
- **调度器**：`run_items` / `arun_items` 包住任意一个“处理一个 item”的函数，负责并发、续跑和封顶，并把记录挂到对应的 item 上。
- **命令行**：`ls`、`show`、`tree`、`export` 用来列调用、看单次调用、看对话树、导出 JSONL。

## 安装

需要 Python 3.12 或更高版本。只在 Linux 上测过。跨进程记账用到了 `fcntl`，所以不支持 Windows。

```bash
pip install git+https://github.com/TingYu1769/callkit
# 或者
uv add git+https://github.com/TingYu1769/callkit
```

运行时只依赖 httpx。挂在 anthropic SDK 2.x 底下要用 httpx2，它随 anthropic SDK 一起安装。

## 用法

### 挂在官方 SDK 底下

```python
import callkit
from anthropic import Anthropic
from openai import AsyncOpenAI, OpenAI

RUN = "runs/E1"     # 设成 None 就不录，SDK 照常发请求

client = OpenAI(..., http_client=callkit.recording_http_client(RUN))
aclient = AsyncOpenAI(..., http_client=callkit.recording_async_http_client(RUN))
claude = Anthropic(..., http_client=callkit.recording_http_client(RUN, lib="httpx2"))
```

- 只录路径以 `/v1/messages`、`/responses`、`/chat/completions` 结尾的 POST 请求，其他请求（embeddings、models 等）照发不录。流式和非流式都录。
- SDK 自己的每次重试各记一条，重试序号写在标签 `retry_count` 里。连不上、超时、被取消这类没有响应的失败也会记一条，然后照常抛出异常。
- 录制出错时（比如磁盘满）只发 RuntimeWarning，不影响 SDK 调用。
- 不读环境变量里的代理设置。要走代理就传 `proxy="http://..."`。`verify`、`limits`、`http2` 这类参数照常传，callkit 会交给内层的传输层。
- `session="..."` 用来给会话命名，parent 只在同一个会话里推断。`tags={...}` 会合进每条记录的标签。

### 用 callkit 自己的客户端

```python
from callkit import Client, Endpoint, Price

ep = Endpoint("anthropic", "https://api.anthropic.com", key_env="ANTHROPIC_API_KEY")
prices = {"claude-sonnet-5-5": Price(input=3, output=15, cache_read=0.3, cache_write=3.75)}  # 美元 / 百万 token，示意值

with Client(ep, "runs/E2", prices=prices, cap_usd=5.0) as c:
    rec = c.call({"model": "claude-sonnet-5-5", "max_tokens": 1000,
                  "messages": [{"role": "user", "content": "你好"}]})

rec.outcome.ok, rec.outcome.stop_reason, rec.cost_usd
```

- 密钥只从环境变量 `key_env` 读，不写进记录。记录里的鉴权头一律打码，`auth_header` 自定义的头名也一样。
- `Endpoint` 的 `api` 选接口：默认 `"anthropic.messages"`，另有 `"openai.responses"`、`"openai.chat_completions"`。`base_url` 不带 `/v1`。服务地址带路径前缀的，前缀写进 `base_url`，比如 DeepSeek 的 Anthropic 兼容接口写 `https://api.deepseek.com/anthropic`。
- `Client` 只发流式请求。请求体里没写 `stream`，会自动补上（Chat 接口还会补 `stream_options.include_usage`）。
- 调用失败不抛异常，结果看 `rec.outcome`。断流、429、5xx 会自动重试，默认共尝试 3 次，每次尝试都记一条并计费。封顶是例外，见[计价与封顶](#计价与封顶)。
- `c.batch(items, concurrency=8)` 并发跑一批请求，按完成顺序产出。已经成功的 `(item_id, 重复序号)` 下次跳过。

### 拼成多轮

```python
from callkit import extend, pending_calls

first = c.call(body)                                   # 模型要调用工具
results = {i: "晴，25°C" for i in pending_calls(first)}
second = c.call(extend(first, tool_results=results), parent=first)
third = c.call(extend(second, text="那上海呢？"), parent=second)
```

`extend` 不联网，只生成下一次的请求体。它把上一次的请求、这次的回答和工具结果拼起来，回答原样搬回，包括 thinking 签名、加密推理、`call_id`。工具结果和工具调用对不上，或者请求会以 assistant 消息结尾，都在拼接时报 `PairingError`，不会发出去。Anthropic 的 `pause_turn` 是例外：这种回答不加工具结果和文本，原样接回去续写。

### 分叉重放

```python
from callkit import fork

alt = c.call(fork(second, model="claude-opus-5-5"), fork_of=second)
alt = c.call(fork(second, edit=lambda b: {**b, "max_tokens": 500}), fork_of=second)
```

- `fork` 改动请求体时保持键的顺序，新记录的 `edits` 列出改了哪些顶层字段。分叉沿用原调用的查询串和请求头（比如 Claude Code 的 `?beta=true` 和 beta 头），鉴权由新端点重新给。
- 换了模型或改了历史以后，旧的 thinking 块可能失效。callkit 照样保留它们，并在 `rec.outcome.flags` 里记一条 `thinking_may_be_invalid:N`。
- 原请求写了 `stream: false` 的（比如 SDK 录到的非流式调用），`fork` 和 `extend` 会改成流式。

### 录下 Claude Code 和 Codex

```bash
callkit proxy --upstream https://api.anthropic.com --run runs/cc-001   # 打印监听地址，如 http://127.0.0.1:41235
ANTHROPIC_BASE_URL=http://127.0.0.1:41235 claude -p "..."
```

代理只把请求原样转给 `--upstream`，不转换接口，所以 Codex 要另起一个上游是 Responses 接口服务的代理。Codex 还需要一个全新的 `CODEX_HOME`，再加一个指向代理的自定义 provider（名字不能叫 `OpenAI`）：

```bash
callkit proxy --upstream https://api.openai.com --run runs/codex-001   # 如 http://127.0.0.1:41236
CODEX_HOME=/tmp/codex-home MY_KEY=... codex exec \
  -c 'model_providers.callkit={name="callkit",base_url="http://127.0.0.1:41236/v1",wire_api="responses",env_key="MY_KEY"}' \
  -c model_provider=callkit "..."
```

- 代理按块原样转发，不改内容，只要求上游不压缩。
- WebSocket 升级请求一律回 426，Codex 收到后会改用 HTTP。
- 记录按会话分组，会话号依次取 `X-Claude-Code-Session-Id`、`session-id`、请求体里的 `prompt_cache_key`。parent 自动推断：上下文相同，并且“候选的请求 + 它的回答”是本次请求的前缀。推断不出唯一的 parent 时，候选写进 `parent_candidates`。
- 录下来的任意一次调用都能拿去 `fork` 重发。

### 计价与封顶

```python
http = callkit.recording_http_client("runs/E3", prices={"deepseek-flash": Price(...)}, cap_usd=5.0)
```

```bash
callkit proxy --upstream ... --run runs/E3 --prices prices.json --cap 5
```

- 设了 `cap_usd` 后，运行目录的累计花费到顶，请求就不再发出。SDK 挂载和代理直接回 HTTP 402，响应头带 `x-should-retry: false`，openai、anthropic SDK 都不会重试；缺所用模型的价格时也这样拦下。callkit 自己的 `Client` 改为抛异常：到顶抛 `CapReached`，缺价格抛 `MissingPrice`。
- 价格按模型合并进运行目录的 `prices.json`。同一个模型后来传了不同的价格，会报错。
- 调用成功却没有用量时（比如 Chat 流式请求没带 `stream_options={"include_usage": True}`），花费记为未知，结果里标 `usage_missing`，并提醒一次。
- 代理回给 Codex 的 402 会被它再试几次，每次也都会被拦下，不会产生费用。

### 回放缓存

```python
with Client(ep, "runs/E1-rerun", replay_from="runs/E1") as c: ...
client = OpenAI(..., http_client=callkit.recording_http_client("runs/E1-rerun", replay_from="runs/E1"))
```

- 地址、请求体、`anthropic-version` 和 `anthropic-beta` 头，以及三个选组织的请求头（`openai-organization`、`openai-project`、`anthropic-workspace-id`）都相同，才算同一个请求。官方 SDK 在同一版本内发出的请求体逐字节相同，可以命中；升级 SDK 后键的顺序可能变，就命中不了。
- 同一个请求第 k 次出现时，取旧运行里的第 k 条成功记录；旧记录用完了就照常联网。同一个进程里，从同一个旧运行回放到同一个新运行的客户端共用计数；换了进程，计数从头算起。
- 命中时新运行里照样记一条，标签 `replay_of` 指向旧记录，花费记 0。

### 调度器

```python
def ask(question):                     # 处理一个 item，里面调几次 SDK 都行
    return client.chat.completions.create(model=..., messages=[{"role": "user", "content": question}])

for r in callkit.run_items(questions, ask, run="runs/E4", concurrency=8, cap_usd=5):
    r.item_id, r.ok, r.value, r.error, r.call_ids
```

- `questions` 是从 item_id 到参数的映射，`repeats=N` 让每个 item 跑 N 次。异步版 `arun_items` 接 async 函数。
- 每个 item 跑完，在运行目录的 `items.jsonl` 记一行。重跑时跳过已经成功的。
- 函数里经录制客户端或 `Client` 发出的调用会记上 item_id，`r.call_ids` 列出这个 item 的调用号。
- 函数抛异常时只算这个 item 失败，其他照跑。要中途停下并马上清理，用 `contextlib.closing(run_items(...))` 包一层（异步版用 `contextlib.aclosing`）。

## 命令行

```bash
callkit ls RUN [--failed] [--limit N]     # 列出调用，末尾汇总 token 和花费
callkit show RUN CALL_ID [--raw]          # 看一次调用：结果、耗时、用量、请求与回答摘要
callkit tree RUN [--session S]            # 按会话把续轮（→）和分叉（⑂）串成树
callkit export RUN [-o FILE] [--ok-only]  # 导出 JSONL，一次调用一行，含完整请求体和回答
callkit proxy --upstream URL --run RUN    # 起录制代理
```

```
$ callkit tree runs/weather
session demo
c_000001  deepseek-flash  tool_calls  get_weather({"city": "北京"})
└→ c_000002  deepseek-flash  stop  北京今天晴，气温约 25°C。
```

## 存储格式

一次调用一个文件夹，`meta.json` 是记录本体，格式名 `callkit.call/2`。主要字段：

| 字段 | 内容 |
|---|---|
| `api`、`source` | 接口名；记录来源：`self`（callkit 的 Client）、`proxy`、`sdk` |
| `request` | 地址、请求头（密钥已打码）、请求体文件名、请求哈希 |
| `response` | 状态码、响应头、事件流文件名、回答文件名 |
| `outcome` | `ok`、`transport_end`、`protocol`、`stop_reason`、`error`、`flags` |
| `usage` | 未参与缓存的输入、缓存读、缓存写（按 TTL 分档）、输出、思考，以及原样的用量 |
| `attempts` | 每次尝试的状态、结局、用量、花费 |
| `parent`、`fork_of`、`edits` | 拼接与分叉的关系；指向别的运行目录时写成 `目录:调用号` |
| `item_id`、`repeat`、`tags` | 批量与调度器的键，以及自定义标签 |

读回一条记录用 `RunStore("runs/E1").load("c_000001")`，记录里带着请求体和回答，可以直接拿去 `extend` 或 `fork`。格式升级时，旧格式照样能读：读到 `/1` 会就地升成 `/2`。

## 工作原理

```
官方 SDK ──http_client──▶ 录制传输层 ─┐
                                      ├──▶ Recorder ──┐
Claude Code / Codex ──▶ 录制代理 ─────┘               ├──▶ 适配器（拼装、判定、用量）──▶ RunStore（磁盘）
callkit Client ──▶ 传输层（重试） ────────────────────┘
```

SDK 挂载和代理共用一个 Recorder，callkit 的 Client 自己建记录。三条路用的是同一组适配器和同一个 RunStore，所以记录格式一样。适配器只做不联网的转换。只有传输层、代理和 SDK 挂载会发网络请求。

## 和同类工具比较

| 工具 | 记录单位 | 接入方式 | 原样字节 | 只用本地文件 | 拼接与分叉 | 花费封顶 |
|---|---|---|---|---|---|---|
| callkit | 一次 API 调用 | SDK 挂载、代理、自带客户端 | 有 | 是 | 有 | 有 |
| [llm](https://github.com/simonw/llm) | 一次对话回合 | 自带客户端与插件 | 存回答 JSON | 是（SQLite） | 可续写、分叉线程 | 无 |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | 一次评测 | 自带模型层 | 默认存每个模型的前几次调用和出错的调用，可开全量 | 是 | `fork()` 复制状态跑多个 solver | 有（按 sample，需配价格） |
| [LiteLLM](https://github.com/BerriAI/litellm) | 网关日志 | 网关、SDK | 无 | 网关可不用数据库，花费跟踪和预算要数据库 | 缓存 | 有（预算） |
| [Langfuse](https://github.com/langfuse/langfuse)、[Phoenix](https://github.com/Arize-ai/phoenix) | 链路追踪 | OpenTelemetry | 无 | 需要服务端 | 在 playground 里重放 | 无 |
| [vcrpy](https://github.com/kevin1024/vcrpy) | HTTP 往返 | 替换 HTTP 库 | 有 | 是 | 回放录像带 | 无 |
| [llm-cassette](https://github.com/aozmekik/llm-cassette)、[retake](https://github.com/Zecrari/retake) | HTTP 往返 | httpx 传输层、代理 | 有 | 是 | 回放录像带 | 无 |
| [claude-trace](https://github.com/badlogic/lemmy/tree/main/apps/claude-trace) | Claude Code 会话 | 注入拦截器 | 有 | 是 | 只能查看 | 无 |

表中信息据各项目 2026 年 10 月的文档整理。vcrpy 这类录像带工具面向测试，能把测试替身做得很好，但不管拼接和分叉，也不封顶花费（llm-cassette 能按价格表报告估算的花费）。可观测性平台适合线上服务，callkit 适合要留存每次调用原文的离线实验。

## 局限

- 不录 WebSocket（比如 Responses 的 WebSocket 模式）；代理遇到 WebSocket 升级回 426。
- 调用方读到一半、又没关掉的流式响应，可能没有记录。
- 多个回答（`n>1`）时，`extend` 只接第 0 个。
- 经过中转网关录到的流，可能已被网关改写过，不是上游的原始字节。
- 事件流不存每行的到达时间，只记首字节和总耗时。callkit 的 `Client` 按行读流，行尾统一存成 `\n`。
- 不支持 Windows（用到 `fcntl`）。

## 参与开发

见 [CONTRIBUTING.md](https://github.com/TingYu1769/callkit/blob/main/CONTRIBUTING.md)。离线测试不花钱，`make check` 一条命令跑完。真实调用的测试标了 `network`，`make check` 不跑。

## 引用

在研究里用到 callkit，可以引用仓库，GitHub 页面右侧的 “Cite this repository” 会按 [CITATION.cff](https://github.com/TingYu1769/callkit/blob/main/CITATION.cff) 生成格式。

## 许可证

[MIT](https://github.com/TingYu1769/callkit/blob/main/LICENSE)
