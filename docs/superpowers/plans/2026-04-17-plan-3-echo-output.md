# ECHO — Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the output layer — a PyQt6 frameless floating HUD that shows ARIA's messages with explainability reasons, Piper TTS for voice output, and a global hotkey listener.

**Architecture:** `Overlay` is a PyQt6 `QWidget` (frameless, always-on-top, bottom-right) that shows messages with a reason field and engagement buttons. `TTS` wraps Piper, speaks through headphones via `aplay`/`paplay` fallback. `hotkey.py` listens globally via `pynput` and fires `HOTKEY_PRESSED` events into the queue. All three are independent — they share only `EventQueue` and `Config`.

**Tech Stack:** PyQt6, Piper TTS, `aplay`/`paplay` (subprocess), `pynput`, `config.py`, `core/event_queue.py`

**Worktree:** Work in `../aria-output/`. Do not touch files outside your ownership list.

---

## File Map

| File | Responsibility |
|------|---------------|
| `output/overlay.py` | `Overlay` — PyQt6 frameless HUD, show/hide, engagement callback |
| `output/tts.py` | `TTS` — Piper wrapper, aplay→paplay fallback |
| `hotkey.py` | `HotkeyListener` — pynput global hotkey, fires `HOTKEY_PRESSED` |
| `tests/test_output.py` | All output tests (uses pytest-qt for overlay) |

---

## Task 1: output/tts.py — Piper TTS with fallback

**Files:**
- Create: `output/tts.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_output.py
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call
from output.tts import TTS
from config import IdleUnloadConfig


@pytest.fixture
def tts():
    return TTS(IdleUnloadConfig(tts_minutes=2.0))


def test_tts_uses_aplay_by_default(tts):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch("output.tts.TTS._generate_wav", return_value=b"\x00" * 100):
            tts.speak("hello world")
    calls = [str(c) for c in mock_run.call_args_list]
    assert any("aplay" in c for c in calls)


def test_tts_falls_back_to_paplay(tts):
    def run_side_effect(cmd, **kwargs):
        if "aplay" in str(cmd):
            raise FileNotFoundError("aplay not found")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=run_side_effect):
        with patch("output.tts.TTS._generate_wav", return_value=b"\x00" * 100):
            tts.speak("hello world")  # must not raise


def test_tts_idle_unloads_after_timeout():
    import time
    tts = TTS(IdleUnloadConfig(tts_minutes=0.0005))  # ~0.03 seconds
    with patch("output.tts.TTS._generate_wav", return_value=b"\x00" * 100):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            tts.speak("test")
    time.sleep(0.05)
    tts.check_idle()
    assert tts._model is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_output.py -v
```

Expected: `ImportError: cannot import name 'TTS'`

- [ ] **Step 3: Write output/tts.py**

```python
# output/tts.py
from __future__ import annotations
import subprocess
import tempfile
import os
from config import IdleUnloadConfig
from process.loader import IdleTimer


class TTS:
    def __init__(self, config: IdleUnloadConfig) -> None:
        self._config = config
        self._model = None
        self._timer = IdleTimer(config.tts_minutes * 60)

    def _load(self):
        from piper import PiperVoice
        model_path = os.path.expanduser("~/.aria/piper/en_US-lessac-medium.onnx")
        self._model = PiperVoice.load(model_path)
        self._timer.reset()

    def _generate_wav(self, text: str) -> bytes:
        if self._model is None:
            self._load()
        self._timer.reset()
        import io
        buf = io.BytesIO()
        with self._model.output_to_wav_file(buf):
            self._model.synthesize(text)
        return buf.getvalue()

    def speak(self, text: str) -> None:
        wav_data = self._generate_wav(text)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            tmp_path = f.name
        try:
            self._play(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _play(self, wav_path: str) -> None:
        try:
            subprocess.run(["aplay", "-q", wav_path], check=True, timeout=30)
        except (FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(["paplay", wav_path], check=True, timeout=30)

    def check_idle(self) -> None:
        if self._model is not None and self._timer.is_expired():
            del self._model
            self._model = None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_output.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add output/tts.py tests/test_output.py
git commit -m "feat(output): TTS added — now ARIA can interrupt you verbally too, not just visually. Sorry in advance."
```

---

## Task 2: hotkey.py — Global pynput listener

**Files:**
- Create: `hotkey.py`
- Modify: `tests/test_output.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_output.py`:

```python
from unittest.mock import MagicMock, patch
from hotkey import HotkeyListener
from core.events import EventType


def test_hotkey_listener_fires_event():
    from core.event_queue import EventQueue
    import asyncio
    q = EventQueue()
    listener = HotkeyListener(queue=q, hotkey_str="ctrl+shift+space")
    fired_events = []

    original_put = q.put_nowait
    def capture(event):
        fired_events.append(event)
    q.put_nowait = capture

    listener._on_activate()
    assert len(fired_events) == 1
    assert fired_events[0].type == EventType.HOTKEY_PRESSED


def test_hotkey_listener_stop_does_not_raise():
    from core.event_queue import EventQueue
    q = EventQueue()
    listener = HotkeyListener(queue=q, hotkey_str="ctrl+shift+space")
    listener.stop()  # must not raise even if never started
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_output.py -k "hotkey" -v
```

Expected: `ImportError: cannot import name 'HotkeyListener'`

- [ ] **Step 3: Write hotkey.py**

```python
# hotkey.py
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
    def __init__(self, queue: EventQueue, hotkey_str: str) -> None:
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_output.py -k "hotkey" -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add hotkey.py tests/test_output.py
git commit -m "feat(output): HotkeyListener wired — Ctrl+Shift+Space now summons ARIA like a genie, minus the smoke and wishes"
```

---

## Task 3: output/overlay.py — PyQt6 HUD

**Files:**
- Create: `output/overlay.py`
- Modify: `tests/test_output.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_output.py`:

```python
import pytest
from pytestqt.qt_compat import qt_api


def test_overlay_show_and_hide(qtbot):
    from output.overlay import Overlay
    from config import OverlayConfig
    cfg = OverlayConfig(position="bottom-right", auto_dismiss_seconds=8)
    overlay = Overlay(cfg)
    qtbot.addWidget(overlay)

    overlay.show_message("Meeting in 30 min", importance="high", reason="Deadline detected in screen text")
    assert overlay.isVisible()

    overlay.hide_message()
    assert not overlay.isVisible()


def test_overlay_low_importance_auto_dismiss(qtbot):
    from output.overlay import Overlay
    from config import OverlayConfig
    cfg = OverlayConfig(position="bottom-right", auto_dismiss_seconds=1)
    overlay = Overlay(cfg)
    qtbot.addWidget(overlay)

    overlay.show_message("Focus drift detected", importance="low", reason="App switch rate high")
    assert overlay.isVisible()
    qtbot.waitUntil(lambda: not overlay.isVisible(), timeout=3000)


def test_overlay_engagement_callback(qtbot):
    from output.overlay import Overlay
    from config import OverlayConfig
    engaged = []
    cfg = OverlayConfig(position="bottom-right", auto_dismiss_seconds=8)
    overlay = Overlay(cfg, on_engage=lambda: engaged.append(True))
    qtbot.addWidget(overlay)

    overlay.show_message("Hey", importance="high", reason="test")
    overlay._on_got_it()
    assert engaged == [True]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_output.py -k "overlay" -v
```

Expected: `ImportError: cannot import name 'Overlay'`

- [ ] **Step 3: Write output/overlay.py**

```python
# output/overlay.py
from __future__ import annotations
from typing import Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from config import OverlayConfig


class Overlay(QWidget):
    dismissed = pyqtSignal()

    def __init__(
        self,
        config: OverlayConfig,
        on_engage: Callable | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._on_engage = on_engage
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self.hide_message)
        self._setup_ui()
        self.hide()

    def _setup_ui(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        self._msg_label = QLabel()
        self._msg_label.setWordWrap(True)
        self._msg_label.setStyleSheet("color: #e2e8f0; font-size: 13px;")
        layout.addWidget(self._msg_label)

        self._reason_label = QLabel()
        self._reason_label.setWordWrap(True)
        self._reason_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        layout.addWidget(self._reason_label)

        btn_row = QHBoxLayout()
        self._got_it_btn = QPushButton("Got it")
        self._got_it_btn.setStyleSheet(
            "QPushButton { background:#1e40af; color:white; border-radius:4px; padding:4px 12px; }"
            "QPushButton:hover { background:#2563eb; }"
        )
        self._got_it_btn.clicked.connect(self._on_got_it)
        btn_row.addStretch()
        btn_row.addWidget(self._got_it_btn)
        layout.addLayout(btn_row)

        self.setStyleSheet(
            "QWidget { background: rgba(15,23,42,0.95); border: 1px solid rgba(99,102,241,0.4); border-radius: 8px; }"
        )
        self._position_window()

    def _position_window(self) -> None:
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        self.move(geo.right() - self.width() - 20, geo.bottom() - 200)

    def show_message(self, message: str, importance: str, reason: str) -> None:
        self._msg_label.setText(message)
        self._reason_label.setText(f"Why: {reason}")
        self._got_it_btn.setVisible(importance == "high")
        self.adjustSize()
        self.show()
        if importance != "high":
            self._auto_timer.start(self._config.auto_dismiss_seconds * 1000)

    def hide_message(self) -> None:
        self._auto_timer.stop()
        self.hide()
        self.dismissed.emit()

    def _on_got_it(self) -> None:
        if self._on_engage:
            self._on_engage()
        self.hide_message()
```

- [ ] **Step 4: Run all output tests**

```bash
pytest tests/test_output.py -v --tb=short
```

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add output/overlay.py tests/test_output.py
git commit -m "feat(output): PyQt6 HUD complete — frameless, always-on-top, explains itself. More self-aware than most meetings."
```

---

## Task 4: Update build/status.json + push

- [ ] **Step 1: Update echo entry in build/status.json**

```json
"echo": {
  "state": "active",
  "task": "All output modules complete. Requesting CIPHER review.",
  "progress": 100,
  "last_update": "<ISO timestamp>",
  "log": [
    "tts.py complete — Piper TTS, aplay→paplay fallback, idle unload",
    "hotkey.py complete — pynput global hotkey, fires HOTKEY_PRESSED",
    "overlay.py complete — PyQt6 frameless HUD, reason field, engagement callback",
    "All tests passing"
  ]
}
```

- [ ] **Step 2: Request ARIA password for push**

```
GIT OPERATION REQUEST
Operation: git push origin feat/output
Why: ECHO module complete — TTS, hotkey, HUD overlay with tests passing.
```

- [ ] **Step 3: Push (after ARIA received)**

```bash
git checkout -b feat/output
git push origin feat/output
```

---

## Self-Review

**Spec coverage:**
- ✅ `output/tts.py` — Piper, aplay→paplay, idle unload → Task 1
- ✅ `hotkey.py` — pynput, HOTKEY_PRESSED event → Task 2
- ✅ `output/overlay.py` — frameless HUD, reason, auto-dismiss, engagement → Task 3
- ✅ push + status update → Task 4

**Type consistency:**
- `Overlay.show_message(message, importance, reason)` consistent in tests and impl ✓
- `TTS(IdleUnloadConfig)` consistent ✓
- `HotkeyListener(queue, hotkey_str)` consistent ✓
