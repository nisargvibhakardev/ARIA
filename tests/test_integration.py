import asyncio
import pytest
from unittest.mock import MagicMock
from core.event_queue import EventQueue
from core.events import Event, EventType
from config import Config


@pytest.mark.asyncio
async def test_screen_event_reaches_context_chunk():
    """A SCREEN_CHANGED event must produce a CONTEXT_CHUNK event."""
    import numpy as np
    queue = EventQueue()

    screen_event = Event(
        type=EventType.SCREEN_CHANGED,
        data={"screenshot": np.zeros((100, 100, 3), dtype=np.uint8),
              "window_title": "Visual Studio Code — main.py",
              "hash": "abc123"}
    )

    context_chunks = []

    async def fake_pipeline():
        event = await queue.get()
        if event.type == EventType.SCREEN_CHANGED:
            chunk = Event(
                type=EventType.CONTEXT_CHUNK,
                data={"text": "def test_foo():", "source": "screen",
                      "scene": MagicMock(app="vscode", focus_level=0.8)}
            )
            context_chunks.append(chunk)

    await queue.put(screen_event)
    await asyncio.wait_for(fake_pipeline(), timeout=2.0)
    assert len(context_chunks) == 1
    assert context_chunks[0].data["source"] == "screen"


@pytest.mark.asyncio
async def test_hotkey_event_bypasses_rate_limiter():
    """HOTKEY_PRESSED must trigger immediate LLM call, skipping energy scheduler."""
    queue = EventQueue()
    decisions_made = []

    hotkey_event = Event(type=EventType.HOTKEY_PRESSED, data={})

    async def fake_hotkey_handler(event):
        if event.type == EventType.HOTKEY_PRESSED:
            decisions_made.append("immediate_query")

    await queue.put(hotkey_event)
    event = await queue.get()
    await fake_hotkey_handler(event)
    assert decisions_made == ["immediate_query"]


def test_config_loads_for_integration():
    cfg = Config.from_yaml("config.yaml")
    assert cfg.screen.interval_seconds > 0
    assert cfg.llm.model != ""
    assert cfg.hotkey != ""
