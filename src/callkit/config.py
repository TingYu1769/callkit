"""Endpoints and prices: configuration needed before calling. Pass Python objects instead of using a config file."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Endpoint:
    """A model service endpoint. Store only the environment variable name for the key; the key itself never appears in code, config, or logs."""

    name: str
    base_url: str  # Without /v1, e.g. http://host:port
    key_env: str  # Environment variable name holding the key
    # Or "openai.responses", "openai.chat_completions"; determines which adapter to use
    api: str = "anthropic.messages"
    # None means use the API default (x-api-key for Anthropic, authorization for OpenAI); custom header names are still masked in call records (meta.json)
    auth_header: str | None = None
    # Only used for anthropic.messages; self-built requests default to 2023-06-01
    anthropic_version: str = "2023-06-01"
    extra_headers: dict[str, str] = field(default_factory=dict)  # e.g. anthropic-workspace-id
    # Set this to use a proxy, e.g. http://proxy.example:8080; proxy environment variables are not read
    proxy: str | None = None


@dataclass(frozen=True)
class Price:
    """Unit price per model, in USD per million tokens. Used only for capping and estimation, not reconciled with the gateway.

    Cache writes are priced by TTL tier: differentiated writes use the matching ``cache_write_by_ttl`` tier (missing tiers fall back to ``cache_write``),
    undifferentiated writes use ``cache_write``.
    """

    input: float
    output: float
    cache_read: float
    # Write price for the default TTL (Anthropic 5m, OpenAI 30m); upstream writes that are not TTL-differentiated also use this
    cache_write: float
    cache_write_by_ttl: dict[str, float] = field(
        default_factory=dict
    )  # Write price for specific TTLs, e.g. {"1h": ...}


PriceTable = dict[str, Price]  # model name -> unit price
