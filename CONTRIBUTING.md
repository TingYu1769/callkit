# 参与开发

欢迎提 issue 和 PR。中文、英文都可以。

## 开发环境

需要 [uv](https://docs.astral.sh/uv/) 和 Python 3.12 或更高版本。

```bash
git clone https://github.com/TingYu1769/callkit
cd callkit
make setup   # uv sync，按 uv.lock 装依赖
make check   # 格式、ruff、basedpyright、离线测试
```

`make check` 不联网、不花钱，提 PR 前请确认它全过。

## 真实调用的测试

`tests/network/` 下的测试会真的调用模型、产生费用。它们标了 `network`，`make check` 和 `make test` 传了 `-m "not network"`，不跑它们；直接运行 `uv run pytest` 会收进来，没设下面环境变量的会跳过。要跑的话，先设环境变量：

| 接口 | 环境变量 |
|---|---|
| Anthropic Messages | `CALLKIT_LIVE_BASE_URL`（不带 `/v1`）、`CALLKIT_LIVE_API_KEY`，可选 `CALLKIT_LIVE_MODEL` |
| OpenAI Responses、Chat Completions | `CALLKIT_LIVE_OPENAI_BASE_URL`（不带 `/v1`）、`CALLKIT_LIVE_OPENAI_API_KEY`，可选 `CALLKIT_LIVE_OPENAI_MODEL` |
| DeepSeek 官方接口 | `CALLKIT_LIVE_DEEPSEEK_API_KEY`，可选 `CALLKIT_LIVE_DEEPSEEK_MODEL` |
| 访问端点要走代理时 | `CALLKIT_LIVE_PROXY` |

然后运行 `uv run pytest -m network tests/network -q`。录制 Claude Code、Codex 的两个测试还需要本机装有 `claude` 和 `codex`。

## 提交与 PR

- 一个 PR 只做一件事；提交信息的类型前缀用英文（`feat`、`fix`、`refactor`、`docs`、`test`、`chore`），描述用中文或英文都行。
- 改了行为的，请在 `CHANGELOG.md` 的“未发布”里写一条。
- 新功能和修复请带测试；修 bug 时先写一个能复现它的测试。

## 这个仓库是导出的

callkit 在维护者的私有仓库里开发，每发一个版本导出到这里，所以这里的提交历史是一个版本一个提交。你的 PR 会被搬进私有仓库（保留你的作者署名），随下一个版本导出回来，届时这里的 PR 会关闭并注明对应的提交。
