"""Record format contract: /2 sample round-trips unchanged; /1 sample upgrades in place to /2 on read."""

import json
from pathlib import Path

import pytest

from callkit.record import FORMAT, Attempt, Record

DATA = Path(__file__).resolve().parents[1] / "data"
V1, V2 = DATA / "meta_sample_v1.json", DATA / "meta_sample_v2.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_round_trip():
    data = load(V2)
    assert data["format"] == FORMAT == "callkit.call/2"
    assert Record.from_dict(data).to_dict() == data


def test_nested_types_restored():
    rec = Record.from_dict(load(V2))
    assert isinstance(rec.attempts[0], Attempt)
    assert rec.outcome.protocol == "completed"
    assert rec.usage is not None and rec.usage.cache_write_by_ttl == {"5m": 1000, "1h": 2000}


def test_v1_upgraded_on_read():
    rec = Record.from_dict(load(V1))
    assert rec.format == FORMAT
    assert rec.usage is not None
    assert (rec.usage.cache_write_tokens, rec.usage.cache_write_by_ttl) == (
        3000,
        {"5m": 1000, "1h": 2000},
    )
    assert rec.attempts[0].usage is None
    u = rec.attempts[1].usage
    assert u is not None and (u.cache_write_tokens, u.cache_write_by_ttl) == (7, {"5m": 7})
    assert "cache_write_5m_tokens" not in json.dumps(rec.to_dict())


def test_transient_fields_not_written():
    rec = Record.from_dict(load(V2))
    rec.body = b'{"model": "x"}'
    rec.message = {"id": "msg_x"}
    assert "body" not in rec.to_dict() and "message" not in rec.to_dict()


def test_unknown_format_rejected():
    data = load(V2)
    data["format"] = "callkit.call/0"
    with pytest.raises(ValueError, match="callkit.call/2"):
        Record.from_dict(data)
