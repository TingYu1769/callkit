"""Adapter protocol and shared request assembly. One implementation per interface: anthropic.messages, openai.responses, openai.chat_completions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from callkit.config import Endpoint
from callkit.record import Outcome, Usage
from callkit.sse import SseEvent


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str]  # contains keys, only in memory, masked before persisting
    body: bytes  # verbatim, not one byte changed


# Request headers not inherited from the original call: auth is supplied by the endpoint, the rest are managed by the HTTP client.
NOT_INHERITED = {
    "x-api-key",
    "authorization",
    "host",
    "content-length",
    "connection",
    "accept-encoding",
    "transfer-encoding",
    "keep-alive",
}


def build_request(
    body: bytes,
    endpoint: Endpoint,
    key: str,
    *,
    path: str,
    default_auth: str,
    fixed_headers: Mapping[str, str],
    inherit_headers: Mapping[str, str] | None,
) -> HttpRequest:
    """Shared by all adapters: content-type + interface fixed headers + endpoint extra headers + inherited headers, then auth last.
    The auth header name comes from ``endpoint.auth_header``; if unset, use the interface default ``default_auth``."""
    headers = {"content-type": "application/json", **fixed_headers}
    headers.update({k.lower(): v for k, v in endpoint.extra_headers.items()})
    for k, v in (inherit_headers or {}).items():
        if k.lower() not in NOT_INHERITED:
            headers[k.lower()] = v
    auth = (endpoint.auth_header or default_auth).lower()
    headers[auth] = key if auth == "x-api-key" else f"Bearer {key}"
    return HttpRequest("POST", endpoint.base_url.rstrip("/") + path, headers, body)


RETRYABLE_CODES = {
    "server_error",
    "rate_limit_exceeded",
    "server_is_overloaded",
    "slow_down",
}  # shared by both OpenAI interfaces


def parse_json(data: str) -> dict[str, Any] | None:
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def failed(
    transport_end: str, protocol: str, reason: str, retryable: bool, error: str | None
) -> Judgement:
    return Judgement(Outcome(False, transport_end, protocol, None, error), retryable, reason)


def bad_json(transport_end: str) -> Judgement:
    """Non-streaming response body is not complete JSON (likely truncated mid-transfer): no answer obtained, retryable."""
    return failed(
        transport_end, "incomplete", "bad_json", True, "response body is not complete JSON"
    )


@dataclass(frozen=True)
class Judgement:
    outcome: Outcome
    retryable: bool  # only retryable when no complete answer was obtained
    reason: str  # written into Attempt.reason


class Adapter(Protocol):
    api: str
    path: str  # interface default path, e.g. /v1/messages; appended to the endpoint base_url

    def http_request(
        self,
        body: bytes,
        endpoint: Endpoint,
        key: str,
        *,
        path: str | None = None,
        inherit_headers: Mapping[str, str] | None = None,
    ) -> HttpRequest:
        """Build the URL and request headers (version headers, beta headers, auth). ``key`` is read by the caller from environment variables and passed in.

        ``path`` is the interface default path plus the original call's query string (e.g. Claude Code's ``/v1/messages?beta=true``).
        ``inherit_headers`` are request headers from the original call (e.g. beta headers), passed when forking or continuing a recorded call. Auth headers are always supplied by the endpoint.
        """
        ...

    def assemble(self, events: Sequence[SseEvent]) -> dict[str, Any] | None:
        """Convert the event stream to an assembled Message; return None if it cannot be assembled."""
        ...

    def judge(
        self,
        status: int | None,
        transport_end: str,
        events: Sequence[SseEvent],
        message: dict[str, Any] | None,
        error_body: str | None,
        request: Mapping[str, Any] | None = None,
    ) -> Judgement:
        """Judge whether an attempt succeeded and whether it can be retried. ``request`` is the parsed request body, used by some rules (e.g. whether encrypted reasoning was requested)."""
        ...

    def stream_defaults(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Fields to add when the request body omits ``stream`` or sets it to false (callkit itself only sends streaming). Chat also needs
        ``stream_options.include_usage``, otherwise the stream has no usage."""
        ...

    def from_json(
        self,
        status: int | None,
        transport_end: str,
        text: str,
        request: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Judgement, Usage | None]:
        """Non-streaming response: read the answer, judgement, and usage from the full JSON. If JSON cannot be parsed (e.g. truncated), record incomplete and retryable."""
        ...

    def usage(self, events: Sequence[SseEvent]) -> Usage | None:
        """Read usage from the event stream; on interruption take the last cumulative value received, and if only the initial event is present compute from input tokens."""
        ...
