"""The rate-limit signal a provider sends to whoever sent its request."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

type RateLimitListener = Callable[[], None]
_LISTENER: ContextVar[RateLimitListener | None] = ContextVar(
    "rate_limit_listener", default=None
)


def report_rate_limited() -> None:
    """Tell the sender of the current request that the provider refused it
    for rate (a 429 or an exhausted token budget). The provider's retry layer
    still retries; the sender only learns to send less."""
    listener = _LISTENER.get()
    if listener is not None:
        listener()


@contextmanager
def listen_for_rate_limits(listener: RateLimitListener) -> Generator[None]:
    """Hear every rate limit reported while the current request runs."""
    token = _LISTENER.set(listener)
    try:
        yield
    finally:
        _LISTENER.reset(token)
