# core/event_queue.py
from __future__ import annotations
import asyncio
import queue
from .events import Event


class EventQueue:
    """Thread-safe event queue. put_nowait() can be called from any thread."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[Event] = queue.SimpleQueue()

    async def put(self, event: Event) -> None:
        self._queue.put_nowait(event)

    def put_nowait(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def get(self) -> Event:
        while True:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)

    def task_done(self) -> None:
        pass
