"""A single source file must not exceed 800 lines (hard limit). The 400-line soft limit is enforced during review."""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "callkit"
HARD_LIMIT = 800


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_source_file_under_hard_limit(path):
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= HARD_LIMIT, (
        f"{path.name} has {lines} lines, over the {HARD_LIMIT}-line limit; split it"
    )
