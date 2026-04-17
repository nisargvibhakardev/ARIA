# tests/test_event_queue.py
import asyncio
import pytest
from core.event_queue import EventQueue
from core.events import Event, EventType


@pytest.fixture
def queue():
    return EventQueue()


@pytest.mark.asyncio
async def test_put_and_get(queue):
    e = Event(type=EventType.HOTKEY_PRESSED, data={})
    await queue.put(e)
    result = await queue.get()
    assert result is e


@pytest.mark.asyncio
async def test_put_nowait_and_get(queue):
    e = Event(type=EventType.SCREEN_CHANGED, data={"screenshot": b""})
    queue.put_nowait(e)
    result = await queue.get()
    assert result.type == EventType.SCREEN_CHANGED


@pytest.mark.asyncio
async def test_fifo_order(queue):
    types = [EventType.SCREEN_CHANGED, EventType.SPEECH_DETECTED, EventType.HOTKEY_PRESSED]
    for t in types:
        queue.put_nowait(Event(type=t, data={}))
    results = [await queue.get() for _ in types]
    assert [r.type for r in results] == types


@pytest.mark.asyncio
async def test_task_done(queue):
    queue.put_nowait(Event(type=EventType.HOTKEY_PRESSED, data={}))
    await queue.get()
    queue.task_done()  # must not raise


@pytest.mark.asyncio
async def test_empty_queue_blocks(queue):
    async def getter():
        return await asyncio.wait_for(queue.get(), timeout=0.05)

    with pytest.raises(asyncio.TimeoutError):
        await getter()
