# 更新记录

版本号按[语义化版本](https://semver.org/lang/zh-CN/)。1.0 之前，次版本可能带不兼容的改动，都会在这里写明；记录格式（`callkit.call/N`）升级时，旧格式照样能读。

## 未发布

## 0.2.5（2026-11-09）

- SQLite 运行目录的并发写入改进：压缩和摘要在写锁外算好；写事务先拿运行目录的跨进程文件锁再开始，等锁的进程按顺序被叫醒；检查点改由后台线程定时做（每个库文件一个线程），写者提交时不再顺带做。8 个进程各 16 个线程写同一目录，实测吞吐 434 条/秒，p99 延迟 0.23 秒。
- 新建的库用 64 KB 页：摘要表的页不再散落在大字段之间，机械盘上 `ls` 扫 1.1 万条从 9 秒降到 0.8 秒。页大小只在建库时生效，已建的库不变。
- README 补上 SQLite 存储、新命令、迁移与备份、NFS 和 fork 的局限。

## 0.2.4（2026-11-09）

- `callkit migrate OLD NEW`：把旧布局目录离线搬进新的 SQLite 目录。迁移期间旧目录不能有别的进程在写（拿不到旧目录的锁就报错）。按账本顺序逐条搬，调用号原样保留；记录的 `run` 改成新目录，`parent`、`fork_of`、`replay_of` 里指向旧目录的引用一并改写；`request.json.gz` 解开后存。每批提交后读回逐字节比对，对不上就停。中断后再跑同一条命令接着搬，没搬完时别的进程打开新目录会报错。
- `callkit cat RUN CALL_ID [request|response|message|meta|文件名]`：把一条调用的一个字段原样写到标准输出。
- `callkit unpack RUN [CALL_ID ...] -o DIR`：按旧布局写出文件（run.json、index.jsonl、calls/<id>/），旧布局的 `RunStore` 能直接读。
- `callkit backup RUN DEST`：用 SQLite 的在线备份接口拷一份一致的快照。运行中的库不要直接拷 `run.db`：还没检查点的数据可能只在 `run.db-wal` 里。
- `RunStore` 新增 `file_names`。

## 0.2.3（2026-11-09）

- `callkit ls`、`tree` 在 SQLite 运行目录上只读摘要表，不再逐条读整条记录；`tree` 每行的回答摘要也在写入时就算好。旧布局目录照旧逐条读。
- 回放缓存的旧运行是 SQLite 布局时，按回放键直接查索引取候选，不读 `meta.json`；“第 k 次取第 k 条”、按“旧运行 → 新运行”共享计数、打开时快照这些规则不变。两种布局的候选都按调用号的数字排序（调用号超过 6 位时和以前的字符串排序不同）。
- 调度器的 item 账本在 SQLite 布局里存进 `items` 表；旧布局仍是 `items.jsonl`。新增 `RunStore.append_item`、`items`、`items_done`。

## 0.2.2（2026-11-09）

- 新的运行目录改用 SQLite 单文件 `run.db` 存储：一条调用一个事务，不再一条调用建一个目录和四五个文件。调用摘要（模型、结果、用量、花费、parent、回答摘要）和大字段（请求体、事件流、回答、记录本体）分表存，超过 4 KB 的字段用 zlib 压缩，读回时逐字节还原。记录格式 `callkit.call/2` 不变。
- 已有的旧布局目录（`run.json`、`index.jsonl`、`calls/`）照常读写，`RunStore(path)` 按目录里有什么自动选后端。同一目录里同时有 `run.db` 和旧布局的文件时报错。
- 调用号改在写入时分配：Client 不再在联网前占号，同一批并发调用的编号按写完的先后排。`RunStore.new_call_id()` 取消；建记录时 `id` 留空，`write()` 后读 `rec.id`。
- `RunStore` 新增 `read_file`、`raw_response`、`load_message`、`run_meta`、`summaries`。`browse.Row` 等摘要函数挪到 `callkit.summary`（原处照旧可导入），`Row` 新增 `said` 字段。
- 多进程、多线程写同一个 SQLite 运行目录时共用一本账；不支持 fork 后在子进程里接着用父进程打开的运行目录（会报错，请在子进程里重新打开）。

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
