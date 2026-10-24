"""Interface adapters: everything related to specific API formats lives here; only offline conversion."""

from __future__ import annotations

from callkit.adapters.anthropic import AnthropicMessages
from callkit.adapters.base import Adapter
from callkit.adapters.openai_chat import OpenAIChat
from callkit.adapters.openai_responses import OpenAIResponses

ADAPTERS: dict[str, type[Adapter]] = {
    AnthropicMessages.api: AnthropicMessages,
    OpenAIResponses.api: OpenAIResponses,
    OpenAIChat.api: OpenAIChat,
}


def get_adapter(api: str) -> Adapter:
    """Get the adapter by ``Endpoint.api``; unknown APIs raise an error listing supported ones."""
    try:
        return ADAPTERS[api]()
    except KeyError:
        raise ValueError(f"unsupported api {api!r}; supported: {sorted(ADAPTERS)}") from None


def api_for_path(path: str) -> str | None:
    """Proxy and SDK mounts identify APIs by path suffix; unrecognized paths (embeddings, models, count_tokens, ...) return None and are passed through without recording.

    Anthropic requires paths ending with ``/v1/messages``: matching only ``/messages`` would incorrectly record OpenAI Assistants'
    ``/threads/{id}/messages``; paths containing ``/threads/`` are never recorded.
    """
    if "/threads/" in path:
        return None
    if path.endswith("/chat/completions"):
        return OpenAIChat.api
    if path.endswith("/responses"):
        return OpenAIResponses.api
    if path.endswith("/v1/messages"):
        return AnthropicMessages.api
    return None
