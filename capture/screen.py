from __future__ import annotations
import hashlib
import threading
import time
import subprocess
from typing import TYPE_CHECKING
import numpy as np
import mss

from core.events import Event, EventType
if TYPE_CHECKING:
    from core.event_queue import EventQueue
    from config import ScreenConfig


def pixel_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def diff_ratio(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    if frame_a.shape != frame_b.shape:
        return 1.0
    diff = np.abs(frame_a.astype(int) - frame_b.astype(int))
    changed = np.any(diff > 10, axis=2)
    return float(changed.mean())


def _get_window_title() -> str:
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=0.5
        )
        return result.stdout.strip()
    except Exception:
        return ""


class ScreenWatcher:
    def __init__(self, queue: EventQueue, config: ScreenConfig) -> None:
        self._queue = queue
        self._config = config
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_hash: str = ""
        self._last_frame: np.ndarray | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        with mss.mss() as sct:
            monitor = sct.monitors[self._config.monitor_index]
            while self._running:
                frame = np.array(sct.grab(monitor))[:, :, :3]
                current_hash = pixel_hash(frame)
                if current_hash != self._last_hash:
                    ratio = diff_ratio(self._last_frame, frame) if self._last_frame is not None else 1.0
                    if ratio >= self._config.diff_threshold or self._last_frame is None:
                        self._last_hash = current_hash
                        self._last_frame = frame
                        self._queue.put_nowait(Event(
                            type=EventType.SCREEN_CHANGED,
                            data={
                                "screenshot": frame,
                                "window_title": _get_window_title(),
                                "hash": current_hash,
                            }
                        ))
                time.sleep(self._config.interval_seconds)
