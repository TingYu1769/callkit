"""SSE framing."""

from callkit.sse import SseEvent, parse


def lines(text: str, t: float = 0.0):
    return [(t, line) for line in text.split("\n")]


def test_events_split_on_blank_lines():
    evs = parse(
        [
            (0.1, "event: ping"),
            (0.1, 'data: {"type": "ping"}'),
            (0.1, ""),
            (0.5, "event: message_stop"),
            (0.6, 'data: {"type": "message_stop"}'),
            (0.6, ""),
        ]
    )
    assert evs == [
        SseEvent("ping", '{"type": "ping"}', 0.1),
        SseEvent("message_stop", '{"type": "message_stop"}', 0.6),
    ]


def test_comment_lines_skipped():
    assert parse(lines(": PING\n\nevent: a\n: PING\ndata: x\n\n")) == [SseEvent("a", "x", 0.0)]


def test_multiline_data_and_single_space_removed():
    assert parse(lines("data: a\ndata:  b\ndata:c\n\n")) == [SseEvent(None, "a\n b\nc", 0.0)]


def test_trailing_event_without_blank_line_discarded():
    assert parse(lines("event: a\ndata: x\n\nevent: b\ndata: y")) == [SseEvent("a", "x", 0.0)]


def test_crlf_and_other_fields():
    assert parse(lines("id: 7\r\nretry: 10\r\ndata: x\r\n\r\n")) == [SseEvent(None, "x", 0.0)]
