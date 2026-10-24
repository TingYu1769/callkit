"""Actually run Codex through a recording proxy to do a small task, then fork one recorded call (costs about $0.2-0.5). Skipped by default.

Requires CALLKIT_LIVE_OPENAI_BASE_URL (without /v1), CALLKIT_LIVE_OPENAI_API_KEY, and codex on PATH;
optional CALLKIT_LIVE_OPENAI_MODEL (default gpt-5.6-sol), CALLKIT_LIVE_PROXY.
  uv run pytest -m network tests/network/test_live_codex_proxy.py -q
"""

import os
import shutil
import subprocess

import pytest

from callkit import Client, Endpoint, fork
from callkit.proxy import Proxy

BASE = os.environ.get("CALLKIT_LIVE_OPENAI_BASE_URL", "")
PROXY = os.environ.get("CALLKIT_LIVE_PROXY") or None
MODEL = os.environ.get("CALLKIT_LIVE_OPENAI_MODEL", "gpt-5.6-sol")
TASK = "在当前目录创建 hello.txt，内容是 hi；然后读出来确认内容。完成后用一句话汇报。"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not BASE or shutil.which("codex") is None,
        reason="no OpenAI endpoint configured or codex not installed",
    ),
]


def run_codex(proxy_url: str, home, work) -> subprocess.CompletedProcess[str]:
    """A fresh CODEX_HOME plus a custom provider pointing to the proxy: no login, no WebSocket."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CODEX_", "OPENAI_"))
        and k.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    env.update(
        CODEX_HOME=str(home),
        CALLKIT_CODEX_KEY=os.environ["CALLKIT_LIVE_OPENAI_API_KEY"],
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    provider = (
        f'model_providers.callkit={{name="callkit",base_url="{proxy_url}/v1",'
        'wire_api="responses",env_key="CALLKIT_CODEX_KEY"}'
    )
    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "workspace-write", "-C", str(work)]
    for opt in (
        provider,
        "model_provider=callkit",
        f"model={MODEL}",
        'model_reasoning_effort="low"',
        'approval_policy="never"',
    ):
        cmd += ["-c", opt]
    cmd.append(TASK)
    return subprocess.run(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def test_live_codex_via_proxy_then_fork(tmp_path):
    proxy = Proxy(BASE, tmp_path / "run", upstream_proxy=PROXY)
    proxy.start()
    home, work = tmp_path / "codex_home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    try:
        done = run_codex(proxy.url, home, work)
    finally:
        proxy.stop()
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert (work / "hello.txt").read_text().strip() == "hi"
    recs = [proxy.store.load(r.id) for r in proxy.store.index() if r.ok]
    assert len(recs) >= 2, proxy.store.index()
    assert all(r.api == "openai.responses" and r.tags.get("request_kind") == "turn" for r in recs)
    assert any(r.parent for r in recs), [(r.id, r.parent, r.parent_candidates) for r in recs]

    last = recs[-1]
    ep = Endpoint(
        "live-openai", BASE, "CALLKIT_LIVE_OPENAI_API_KEY", api="openai.responses", proxy=PROXY
    )
    with Client(ep, tmp_path / "fork", max_attempts=6) as c:
        alt = c.call(
            fork(
                last,
                edit=lambda b: {
                    **b,
                    "reasoning": {**(b.get("reasoning") or {}), "effort": "medium"},
                },
            ),
            fork_of=last,
        )
    assert alt.outcome.ok and alt.edits == ["reasoning"], alt.outcome
    assert "session-id" in {k.lower() for k in alt.request.headers}  # reused Codex request headers
