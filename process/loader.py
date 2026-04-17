from __future__ import annotations
import threading
import time
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class IdleTimer:
    def __init__(self, timeout_seconds: float) -> None:
        self._timeout = timeout_seconds
        self._last_reset: float = time.monotonic()

    def reset(self) -> None:
        self._last_reset = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self._last_reset) >= self._timeout


class ComponentLoader(Generic[T]):
    def __init__(
        self,
        load_fn: Callable[[], T],
        unload_fn: Callable[[T], None],
        timeout_seconds: float,
    ) -> None:
        self._load_fn = load_fn
        self._unload_fn = unload_fn
        self._timer = IdleTimer(timeout_seconds)
        self._component: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        with self._lock:
            if self._component is None:
                self._component = self._load_fn()
            self._timer.reset()
            return self._component

    def check_idle(self) -> None:
        with self._lock:
            if self._component is not None and self._timer.is_expired():
                self._unload_fn(self._component)
                self._component = None
