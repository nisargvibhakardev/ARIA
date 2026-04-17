from __future__ import annotations
import time


class IdleTimer:
    def __init__(self, timeout_seconds: float) -> None:
        self._timeout = timeout_seconds
        self._last_reset = time.monotonic()

    def reset(self) -> None:
        self._last_reset = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self._last_reset) >= self._timeout
