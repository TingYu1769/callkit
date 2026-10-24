"""callkit version and commit, written into each run's run.json."""

from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path


def version() -> str:
    return metadata.version("callkit")


def commit() -> str | None:
    """When installed editable, get the commit of the repository containing callkit; add -dirty if the package directory has uncommitted changes."""
    pkg = str(Path(__file__).resolve().parent)
    try:
        head = _git(pkg, "rev-parse", "HEAD")
        dirty = _git(pkg, "status", "--porcelain", "--", ".")
    except (OSError, subprocess.SubprocessError):
        return None  # not a git repo (non-editable install) or git is not installed
    return head + ("-dirty" if dirty else "")


def _git(cwd: str, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", cwd, *args], capture_output=True, text=True, timeout=5, check=True
    )
    return out.stdout.strip()
