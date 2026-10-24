"""Calculate cost by usage and unit price."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from callkit.config import Price, PriceTable
from callkit.record import Outcome, Usage

USAGE_MISSING = "usage_missing"  # marker for a successful call with no usage


def cost_usd(usage: Usage, price: Price) -> float:
    """Multiply each usage component by its unit price. Used only for capping and estimation; not reconciled with the gateway. Reasoning is already included in output_tokens."""
    by_ttl = usage.cache_write_by_ttl or {}
    split = sum(by_ttl.values())
    write = sum(
        n * price.cache_write_by_ttl.get(ttl, price.cache_write) for ttl, n in by_ttl.items()
    )
    write += max(usage.cache_write_tokens - split, 0) * price.cache_write
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + write
    )
    return total / 1_000_000


def load_prices(path: str | Path) -> PriceTable:
    """Read a price file: ``{"model name": {"input": ..., "output": ..., "cache_read": ..., "cache_write": ...,
    "cache_write_by_ttl": {...}}}``, USD per million tokens."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {model: Price(**fields) for model, fields in raw.items()}


def dump_prices(table: PriceTable) -> dict[str, Any]:
    return {model: asdict(price) for model, price in sorted(table.items())}


def mark_usage_missing(outcome: Outcome) -> None:
    """A successful call without usage (e.g. Chat streaming without stream_options.include_usage): cost unknown, do not record 0, mark it on the result and
    warn once (recording 0 would prevent the spending cap from ever being reached). Returns None, and callers treat it as the cost."""
    outcome.flags.append(USAGE_MISSING)
    warnings.warn(
        "callkit: a call succeeded without usage, so its cost is unknown and the spending cap "
        'will undercount; for streaming Chat requests pass stream_options={"include_usage": True}',
        RuntimeWarning,
        stacklevel=4,
    )
