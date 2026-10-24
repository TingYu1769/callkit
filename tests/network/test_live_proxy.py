"""Actually runs a small Claude Code task through the recording proxy (costs money, about $0.05-0.2). Not run by default.

Requires CALLKIT_LIVE_BASE_URL, CALLKIT_LIVE_API_KEY, and claude in PATH; optional CALLKIT_LIVE_MODEL,
CALLKIT_LIVE_PROXY (proxy used by the proxy to reach upstream).
  uv run pytest -m network tests/network/test_live_proxy.py -q
"""

import json
import os
import shutil
import subprocess

import pytest

from callkit.proxy import Proxy

pytestmark = pytest.mark.network

BASE = os.environ.get("CALLKIT_LIVE_BASE_URL")


@pytest.mark.skipif(
    not BASE or shutil.which("claude") is None,
    reason="no endpoint configured or claude not installed",
)
def test_live_claude_code_via_proxy(tmp_path):
    assert BASE
    proxy = Proxy(
        BASE, tmp_path / "run", upstream_proxy=os.environ.get("CALLKIT_LIVE_PROXY") or None
    )
    proxy.start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        # When launching Claude Code from within a Claude Code session, clear inherited CLAUDE* and ANTHROPIC* variables.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("CLAUDE", "ANTHROPIC"))
            and k.lower() not in ("http_proxy", "https_proxy", "all_proxy")
        }
        env.update(
            CLAUDE_CONFIG_DIR=str(tmp_path / "claude_home"),
            ANTHROPIC_BASE_URL=proxy.url,
            ANTHROPIC_API_KEY=os.environ["CALLKIT_LIVE_API_KEY"],
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
            NO_PROXY="127.0.0.1,localhost",
            no_proxy="127.0.0.1,localhost",
        )
        cmd = [
            "claude",
            "-p",
            "在当前目录创建 hello.txt，内容是 hi；然后读出来确认内容。完成后用一句话汇报。",
            "--model",
            os.environ.get("CALLKIT_LIVE_MODEL", "claude-sonnet-5-5"),
            "--allowedTools",
            "Read,Write",
            "--output-format",
            "json",
        ]
        done = subprocess.run(
            cmd,
            check=False,
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=900,
        )
    finally:
        proxy.stop()
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert json.loads(done.stdout).get("is_error") is False
    assert (work / "hello.txt").read_text().strip() == "hi"
    rows = proxy.store.index()
    assert len(rows) >= 2 and all(r.ok for r in rows), rows
    main = [r for r in rows if r.parent is not None]
    assert main, "at least one call should have an inferred parent"
    first = proxy.store.load(rows[0].id)
    assert first.request.url.endswith("/v1/messages?beta=true")
    assert "anthropic-beta" in {k.lower() for k in first.request.headers}
