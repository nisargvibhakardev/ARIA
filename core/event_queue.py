# core/event_queue.py
from __future__ import annotations
import asyncio
from .events import Event


class EventQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    async def put(self, event: Event) -> None:
        await self._queue.put(event)

    def put_nowait(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def get(self) -> Event:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()
