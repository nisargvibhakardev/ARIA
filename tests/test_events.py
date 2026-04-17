# tests/test_events.py
import time
import pytest
from core.events import Event, EventType


def test_event_type_values():
    assert EventType.SCREEN_CHANGED.value == "screen_changed"
    assert EventType.SPEECH_DETECTED.value == "speech_detected"
    assert EventType.CONTEXT_CHUNK.value == "context_chunk"
    assert EventType.HOTKEY_PRESSED.value == "hotkey_pressed"


def test_event_defaults_timestamp():
    before = time.time()
    e = Event(type=EventType.SCREEN_CHANGED, data={"screenshot": b"abc"})
    after = time.time()
    assert before <= e.timestamp <= after


def test_event_explicit_timestamp():
    e = Event(type=EventType.SPEECH_DETECTED, data={}, timestamp=1234567890.0)
    assert e.timestamp == 1234567890.0


def test_event_data_preserved():
    data = {"audio_bytes": b"\x00\x01", "source": "mic"}
    e = Event(type=EventType.SPEECH_DETECTED, data=data)
    assert e.data["source"] == "mic"
    assert e.data["audio_bytes"] == b"\x00\x01"


def test_event_type_is_enum():
    e = Event(type=EventType.HOTKEY_PRESSED, data={})
    assert isinstance(e.type, EventType)
