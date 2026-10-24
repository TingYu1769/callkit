# 更新记录

版本号按[语义化版本](https://semver.org/lang/zh-CN/)。1.0 之前，次版本可能带不兼容的改动，都会在这里写明；记录格式（`callkit.call/N`）升级时，旧格式照样能读。

## 未发布

## 0.2.1（2026-10-24）

- 回放缓存打开旧运行时只读 `index.jsonl`，不再逐条读 `meta.json`；某个请求命中同一请求哈希时，才按调用号顺序读候选、核对地址和请求头，读到第一条对上的就停。10 万条记录、放在机械盘上的旧运行，打开时间从几十分钟降到几秒。回放语义不变。
- 旧运行里有读不了的 `meta.json` 时，回放跳过这一条并发 RuntimeWarning，不再在打开时报错。

## 0.2.0（2026-10-24）

首次公开发布。

- 三种接口：Anthropic Messages、OpenAI Responses、OpenAI Chat Completions，以及兼容它们的服务（如 DeepSeek 的两种接口）。流式、非流式响应都能拼装、判定、计算用量。
- 三种接入方式：挂在官方 openai、anthropic SDK 底下（`recording_http_client`、`recording_async_http_client`，支持 httpx 与 httpx2）；录制代理 `callkit proxy`，能录 Claude Code 和 Codex；callkit 自己的 `Client`，带重试、批量和续跑。
- 每次调用原样存下请求体和事件流原文，另存拼好的回答、两层结果（传输层、协议层）、用量和花费，记录格式 `callkit.call/2`。
- `extend` 拼多轮，`fork` 分叉重发，录到的调用也能分叉；代理和 SDK 挂载在同一会话内自动推断 parent。
- 计价与封顶：按价格表算花费；到顶后 SDK 挂载和代理回 HTTP 402（SDK 不重试），`Client` 抛 `CapReached`；多个进程写同一个运行目录时共用一本账。
- 回放缓存：同一请求第 k 次出现时，取旧运行里的第 k 条成功记录。
- 调度器 `run_items` / `arun_items`：管并发、续跑和封顶，并把记录挂到对应的 item 上。
- 命令行：`callkit ls`、`show`、`tree`、`export`。
