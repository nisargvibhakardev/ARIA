from __future__ import annotations
from typing import TYPE_CHECKING
from pynput import keyboard
from core.events import Event, EventType

if TYPE_CHECKING:
    from core.event_queue import EventQueue


def _parse_hotkey(hotkey_str: str) -> frozenset:
    parts = hotkey_str.lower().split("+")
    keys = set()
    for p in parts:
        p = p.strip()
        if p == "ctrl":
            keys.add(keyboard.Key.ctrl)
        elif p == "shift":
            keys.add(keyboard.Key.shift)
        elif p == "alt":
            keys.add(keyboard.Key.alt)
        elif len(p) == 1:
            keys.add(keyboard.KeyCode.from_char(p))
        else:
            keys.add(getattr(keyboard.Key, p, keyboard.KeyCode.from_char(p)))
    return frozenset(keys)


class HotkeyListener:
    def __init__(self, queue: "EventQueue", hotkey_str: str) -> None:
        self._queue = queue
        self._hotkey = keyboard.HotKey(
            _parse_hotkey(hotkey_str),
            self._on_activate,
        )
        self._listener: keyboard.Listener | None = None

    def _on_activate(self) -> None:
        self._queue.put_nowait(Event(type=EventType.HOTKEY_PRESSED, data={}))

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._hotkey.press,
            on_release=self._hotkey.release,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
