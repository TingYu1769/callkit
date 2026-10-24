"""SSE framing: split received text lines into events. Parsing only; no network access."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SseEvent:
    event: str | None  # value from the ``event:`` line; None if no event line
    data: str  # value from the ``data:`` line (multiple data lines joined with newlines per spec)
    t: float  # time when the last line of this event arrived, in seconds relative to request start


def parse(lines: Iterable[tuple[float, str]]) -> list[SseEvent]:
    """Split events by blank lines. Skip comment lines starting with ``:`` (gateway ``: PING``); raw text is stored elsewhere, not discarded here.

    Per SSE spec: strip one leading space after the colon in field values; other fields like ``id``, ``retry`` are ignored;
    a trailing half-event not terminated by a blank line at stream end is discarded (the adapter will treat it as incomplete due to missing message_stop).
    """
    events: list[SseEvent] = []
    name: str | None = None
    data: list[str] = []
    last_t = 0.0
    for t, raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data:
                events.append(SseEvent(name, "\n".join(data), last_t))
            name, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
        last_t = t
    return events
