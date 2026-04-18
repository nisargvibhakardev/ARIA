# core/events.py
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    SCREEN_CHANGED = "screen_changed"
    SPEECH_DETECTED = "speech_detected"
    CONTEXT_CHUNK = "context_chunk"
    HOTKEY_PRESSED = "hotkey_pressed"
    ROLLING_TRANSCRIPT = "rolling_transcript"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
