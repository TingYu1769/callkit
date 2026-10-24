"""Errors raised intentionally by callkit. Call failures do not raise; they are written to the outcome. This module holds only usage errors and aborts."""


class CallkitError(Exception):
    """Base class for all callkit errors."""


class PairingError(CallkitError):
    """Raised when tool results cannot be paired with the previous round's tool_use, or when text is appended or tools are changed in a mixed tool turn."""


class StreamRequired(CallkitError):
    """Self-initiated calls must use streaming; the request body explicitly set stream: false."""


class ItemChanged(CallkitError):
    """Raised when the request hash for the same item_id changes during resumption, to avoid mixing modified inputs with old results."""


class CapReached(CallkitError):
    """Raised when the accumulated cost of completed calls exceeds the cap, so no new requests are sent."""


class MissingPrice(CallkitError):
    """A spending cap is set, but the price list is missing this model."""
