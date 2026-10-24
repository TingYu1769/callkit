"""Tiered pricing by TTL. Prices are USD per million tokens, rounded to simple numbers."""

import pytest

from callkit import Price
from callkit.pricing import cost_usd
from callkit.record import Usage

ANTHROPIC = Price(
    input=3, output=15, cache_read=0.3, cache_write=3.75, cache_write_by_ttl={"1h": 6}
)
OPENAI = Price(input=2, output=8, cache_read=0.2, cache_write=2.5, cache_write_by_ttl={"12h": 5})


def test_plain_tokens():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000)
    assert cost_usd(u, ANTHROPIC) == pytest.approx(3 + 15 + 0.3)


def test_cache_write_by_ttl():
    u = Usage(cache_write_tokens=3_000_000, cache_write_by_ttl={"5m": 1_000_000, "1h": 2_000_000})
    assert cost_usd(u, ANTHROPIC) == pytest.approx(3.75 + 2 * 6)
    u = Usage(cache_write_tokens=2_000_000, cache_write_by_ttl={"30m": 1_000_000, "12h": 1_000_000})
    assert cost_usd(u, OPENAI) == pytest.approx(2.5 + 5)


def test_unsplit_and_partially_split_writes_use_default_price():
    assert cost_usd(Usage(cache_write_tokens=1_000_000), OPENAI) == pytest.approx(2.5)
    u = Usage(cache_write_tokens=3_000_000, cache_write_by_ttl={"12h": 1_000_000})
    assert cost_usd(u, OPENAI) == pytest.approx(5 + 2 * 2.5)


def test_load_and_dump_prices_round_trip(tmp_path):
    import json

    from callkit import Price
    from callkit.pricing import dump_prices, load_prices

    table = {"m": Price(1, 2, 0.1, 1.25, {"1h": 2})}
    (tmp_path / "p.json").write_text(json.dumps(dump_prices(table)))
    assert load_prices(tmp_path / "p.json") == table
