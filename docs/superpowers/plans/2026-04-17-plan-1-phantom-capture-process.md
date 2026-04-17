# PHANTOM — Capture & Process Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the always-on capture pipeline — screen watcher, mic watcher, push-to-talk, OCR, STT, and scene context parser — producing `CONTEXT_CHUNK` events the rest of ARIA consumes.

**Architecture:** Two background threads (screen, mic) feed raw events into the shared `EventQueue`. A lazy-loading `ComponentLoader` manages OCR and Whisper — both load on first use, unload after idle timeout. `SceneParser` converts raw screenshots into semantic `SceneContext` objects (app, task type, entities, focus level) before OCR text hits the queue, reducing downstream token usage ~10×.

**Tech Stack:** `mss`, `pytesseract`, `sounddevice`, `webrtcvad`, `faster-whisper`, `xdotool` (via subprocess), Python threading, `core/event_queue.py`, `config.py`

**Worktree:** Work in `../aria-capture/`. Do not touch files outside your ownership list.

---

## File Map

| File | Responsibility |
|------|---------------|
| `capture/screen.py` | `ScreenWatcher` — mss screenshots, pixel-hash diff, fires `SCREEN_CHANGED` |
| `capture/mic.py` | `MicWatcher` — sounddevice + webrtcvad, fires `SPEECH_DETECTED` |
| `capture/push_to_talk.py` | `PushToTalk` — toggle-record for HUD mic button |
| `process/loader.py` | `ComponentLoader` — lazy load/unload with `IdleTimer` |
| `process/ocr.py` | `OCREngine` — pytesseract wrapper, deduplicates identical frames |
| `process/stt.py` | `STTEngine` — faster-whisper wrapper, returns transcript + language |
| `process/scene.py` | `SceneParser` — produces `SceneContext` from screenshot + window title |
| `tests/test_capture.py` | All capture + process tests |

---

## Task 1: capture/screen.py — ScreenWatcher

**Files:**
- Create: `capture/screen.py`
- Modify: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_capture.py
import hashlib
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from capture.screen import ScreenWatcher, pixel_hash, diff_ratio


def test_pixel_hash_returns_string():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pixel_hash(frame)
    assert isinstance(result, str)
    assert len(result) == 64  # sha256 hex


def test_pixel_hash_same_frame_same_hash():
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
    assert pixel_hash(frame) == pixel_hash(frame.copy())


def test_pixel_hash_different_frames_different_hash():
    frame_a = np.zeros((50, 50, 3), dtype=np.uint8)
    frame_b = np.ones((50, 50, 3), dtype=np.uint8) * 255
    assert pixel_hash(frame_a) != pixel_hash(frame_b)


def test_diff_ratio_identical_frames():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert diff_ratio(frame, frame.copy()) == 0.0


def test_diff_ratio_completely_different():
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = np.ones((100, 100, 3), dtype=np.uint8) * 255
    ratio = diff_ratio(frame_a, frame_b)
    assert ratio > 0.9


def test_diff_ratio_within_threshold():
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_b[0, 0] = [255, 255, 255]  # one pixel changed
    ratio = diff_ratio(frame_a, frame_b)
    assert ratio < 0.15  # below default threshold
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_capture.py::test_pixel_hash_returns_string -v
```

Expected: `ImportError: cannot import name 'ScreenWatcher' from 'capture.screen'`

- [ ] **Step 3: Write capture/screen.py**

```python
# capture/screen.py
from __future__ import annotations
import hashlib
import threading
import time
import subprocess
from typing import TYPE_CHECKING
import numpy as np
import mss
import mss.tools

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
                    ratio = diff_ratio(
                        np.frombuffer(bytes.fromhex(self._last_hash), dtype=np.uint8).reshape(1, 1, 3)
                        if self._last_hash else np.zeros((1, 1, 3), dtype=np.uint8),
                        frame[:1, :1]
                    ) if self._last_hash else 1.0
                    if ratio >= self._config.diff_threshold or not self._last_hash:
                        self._last_hash = current_hash
                        self._queue.put_nowait(Event(
                            type=EventType.SCREEN_CHANGED,
                            data={
                                "screenshot": frame,
                                "window_title": _get_window_title(),
                                "hash": current_hash,
                            }
                        ))
                time.sleep(self._config.interval_seconds)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_capture.py -k "pixel_hash or diff_ratio" -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add capture/screen.py tests/test_capture.py
git commit -m "feat(capture): ScreenWatcher online — it stares at your screen every 3 seconds like a paranoid roommate"
```

---

## Task 2: capture/mic.py — MicWatcher

**Files:**
- Modify: `capture/mic.py`
- Modify: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture.py`:

```python
from capture.mic import MicWatcher, is_speech_frame


def test_is_speech_frame_silent():
    vad = MagicMock()
    vad.is_speech.return_value = False
    silent_audio = bytes(960)  # 30ms at 16kHz = 480 samples × 2 bytes
    assert is_speech_frame(vad, silent_audio, 16000) is False


def test_is_speech_frame_speech():
    vad = MagicMock()
    vad.is_speech.return_value = True
    audio = bytes(960)
    assert is_speech_frame(vad, audio, 16000) is True


def test_mic_watcher_initialises():
    from core.event_queue import EventQueue
    from config import MicConfig
    import asyncio
    loop = asyncio.new_event_loop()
    q = EventQueue()
    cfg = MicConfig(vad_aggressiveness=2)
    watcher = MicWatcher(q, cfg)
    assert watcher is not None
    loop.close()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_capture.py -k "mic or speech_frame" -v
```

Expected: `ImportError: cannot import name 'MicWatcher'`

- [ ] **Step 3: Write capture/mic.py**

```python
# capture/mic.py
from __future__ import annotations
import threading
import collections
from typing import TYPE_CHECKING
import numpy as np
import sounddevice as sd
import webrtcvad

from core.events import Event, EventType
if TYPE_CHECKING:
    from core.event_queue import EventQueue
    from config import MicConfig

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480
FRAME_BYTES = FRAME_SAMPLES * 2  # int16


def is_speech_frame(vad: webrtcvad.Vad, frame: bytes, sample_rate: int) -> bool:
    return vad.is_speech(frame, sample_rate)


class MicWatcher:
    def __init__(self, queue: EventQueue, config: MicConfig) -> None:
        self._queue = queue
        self._config = config
        self._vad = webrtcvad.Vad(config.vad_aggressiveness)
        self._running = False
        self._thread: threading.Thread | None = None
        self._ring: collections.deque[bytes] = collections.deque(maxlen=30)
        self._speech_buffer: list[bytes] = []
        self._in_speech = False
        self._silence_count = 0
        self._SILENCE_THRESHOLD = 10  # frames of silence before flushing

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        buffer = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        idx = 0

        def callback(indata, frames, time_info, status):
            nonlocal idx
            samples = indata[:, 0].astype(np.int16)
            for s in samples:
                buffer[idx] = s
                idx += 1
                if idx == FRAME_SAMPLES:
                    self._process_frame(buffer.tobytes())
                    idx = 0

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='int16', blocksize=FRAME_SAMPLES,
                            callback=callback):
            while self._running:
                sd.sleep(100)

    def _process_frame(self, frame: bytes) -> None:
        speech = is_speech_frame(self._vad, frame, SAMPLE_RATE)
        if speech:
            self._in_speech = True
            self._silence_count = 0
            self._speech_buffer.append(frame)
        elif self._in_speech:
            self._silence_count += 1
            if self._silence_count >= self._SILENCE_THRESHOLD:
                audio = b"".join(self._speech_buffer)
                self._queue.put_nowait(Event(
                    type=EventType.SPEECH_DETECTED,
                    data={"audio_bytes": audio, "sample_rate": SAMPLE_RATE}
                ))
                self._speech_buffer.clear()
                self._in_speech = False
                self._silence_count = 0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_capture.py -k "mic or speech_frame" -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add capture/mic.py tests/test_capture.py
git commit -m "feat(capture): MicWatcher deployed — it only bothers you when you actually say something, unlike some colleagues"
```

---

## Task 3: process/loader.py — ComponentLoader with IdleTimer

**Files:**
- Create: `process/loader.py`
- Modify: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture.py`:

```python
import time
from process.loader import IdleTimer, ComponentLoader


def test_idle_timer_not_expired_immediately():
    timer = IdleTimer(timeout_seconds=60)
    timer.reset()
    assert not timer.is_expired()


def test_idle_timer_expires():
    timer = IdleTimer(timeout_seconds=0.05)
    timer.reset()
    time.sleep(0.1)
    assert timer.is_expired()


def test_idle_timer_reset_restarts_countdown():
    timer = IdleTimer(timeout_seconds=0.05)
    timer.reset()
    time.sleep(0.03)
    timer.reset()
    time.sleep(0.03)
    assert not timer.is_expired()


def test_component_loader_loads_on_get():
    loaded = []
    unloaded = []
    loader = ComponentLoader(
        load_fn=lambda: loaded.append(1) or "component",
        unload_fn=lambda c: unloaded.append(c),
        timeout_seconds=60,
    )
    result = loader.get()
    assert result == "component"
    assert loaded == [1]


def test_component_loader_reuses_loaded_component():
    calls = []
    loader = ComponentLoader(
        load_fn=lambda: calls.append(1) or "comp",
        unload_fn=lambda c: None,
        timeout_seconds=60,
    )
    loader.get()
    loader.get()
    assert len(calls) == 1


def test_component_loader_unloads_on_expiry():
    unloaded = []
    loader = ComponentLoader(
        load_fn=lambda: "comp",
        unload_fn=lambda c: unloaded.append(c),
        timeout_seconds=0.05,
    )
    loader.get()
    time.sleep(0.1)
    loader.check_idle()
    assert unloaded == ["comp"]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_capture.py -k "idle_timer or component_loader" -v
```

Expected: `ImportError: cannot import name 'IdleTimer'`

- [ ] **Step 3: Write process/loader.py**

```python
# process/loader.py
from __future__ import annotations
import threading
import time
from typing import Any, Callable, Generic, TypeVar

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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_capture.py -k "idle_timer or component_loader" -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add process/loader.py tests/test_capture.py
git commit -m "feat(process): ComponentLoader lazy-loads like a developer on a Friday afternoon — only works when absolutely necessary"
```

---

## Task 4: process/scene.py — SceneParser + SceneContext

**Files:**
- Create: `process/scene.py`
- Modify: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture.py`:

```python
from process.scene import SceneParser, SceneContext, detect_app, detect_task_type


def test_detect_app_vscode():
    assert detect_app("Visual Studio Code — main.py") == "vscode"


def test_detect_app_chrome():
    assert detect_app("Google Chrome — GitHub") == "chrome"


def test_detect_app_slack():
    assert detect_app("Slack — #general") == "slack"


def test_detect_app_terminal():
    assert detect_app("Terminal — bash") == "terminal"


def test_detect_app_unknown():
    assert detect_app("Some Random App") == "unknown"


def test_detect_task_type_coding():
    assert detect_task_type("vscode", "def test_foo(): ...") == "coding"


def test_detect_task_type_browsing():
    assert detect_task_type("chrome", "Wikipedia article") == "browsing"


def test_detect_task_type_communicating():
    assert detect_task_type("slack", "hey can you review this") == "communicating"


def test_scene_context_has_expected_fields():
    ctx = SceneContext(
        app="vscode", task_type="coding",
        entities=["main.py", "EventQueue"],
        focus_level=0.8, delta={"app_switch": False},
        raw_text="some text"
    )
    assert ctx.app == "vscode"
    assert ctx.focus_level == 0.8
    assert "main.py" in ctx.entities
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_capture.py -k "scene or app or task_type" -v
```

Expected: `ImportError: cannot import name 'SceneParser'`

- [ ] **Step 3: Write process/scene.py**

```python
# process/scene.py
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any

_APP_PATTERNS = [
    ("vscode",    [r"visual studio code", r"vscode", r"\.py —", r"\.ts —", r"\.js —"]),
    ("chrome",    [r"google chrome", r"chromium", r"firefox"]),
    ("slack",     [r"slack"]),
    ("terminal",  [r"terminal", r"bash", r"zsh", r"konsole", r"gnome-terminal"]),
    ("notion",    [r"notion"]),
    ("gmail",     [r"gmail", r"mail\.google"]),
]

_TASK_PATTERNS = {
    "coding":        ["vscode", "terminal"],
    "browsing":      ["chrome"],
    "communicating": ["slack", "gmail"],
}


def detect_app(window_title: str) -> str:
    title_lower = window_title.lower()
    for app, patterns in _APP_PATTERNS:
        if any(re.search(p, title_lower) for p in patterns):
            return app
    return "unknown"


def detect_task_type(app: str, text: str) -> str:
    for task_type, apps in _TASK_PATTERNS.items():
        if app in apps:
            return task_type
    return "general"


def extract_entities(text: str) -> list[str]:
    entities = []
    # File names
    entities += re.findall(r'\b\w+\.(?:py|ts|js|yaml|json|md|txt)\b', text)
    # URLs
    entities += re.findall(r'https?://\S+', text)
    # CamelCase identifiers (likely class/function names)
    entities += re.findall(r'\b[A-Z][a-zA-Z0-9]{3,}\b', text)
    return list(dict.fromkeys(entities))[:20]  # deduplicate, cap at 20


@dataclass
class SceneContext:
    app: str
    task_type: str
    entities: list[str]
    focus_level: float
    delta: dict[str, Any]
    raw_text: str


class SceneParser:
    def __init__(self) -> None:
        self._last_app: str = ""
        self._focus_ema: float = 0.5  # exponential moving average

    def parse(self, window_title: str, ocr_text: str) -> SceneContext:
        app = detect_app(window_title)
        task_type = detect_task_type(app, ocr_text)
        entities = extract_entities(ocr_text)

        # EMA focus: high focus = same app, same task, low entity churn
        app_switch = app != self._last_app
        entity_count = len(entities)
        novelty = min(entity_count / 20.0, 1.0)
        target_focus = 0.2 if app_switch else (1.0 - novelty * 0.5)
        self._focus_ema = 0.7 * self._focus_ema + 0.3 * target_focus

        delta = {"app_switch": app_switch, "entity_count": entity_count}
        self._last_app = app

        return SceneContext(
            app=app,
            task_type=task_type,
            entities=entities,
            focus_level=round(self._focus_ema, 3),
            delta=delta,
            raw_text=ocr_text,
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_capture.py -k "scene or detect_app or detect_task or scene_context" -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add process/scene.py tests/test_capture.py
git commit -m "feat(process): SceneParser knows what you're doing better than your standup — and doesn't judge (much)"
```

---

## Task 5: process/ocr.py + process/stt.py

**Files:**
- Create: `process/ocr.py`
- Create: `process/stt.py`
- Modify: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture.py`:

```python
from unittest.mock import patch, MagicMock
from process.ocr import OCREngine
from process.stt import STTEngine
from config import IdleUnloadConfig, WhisperConfig
import numpy as np


def test_ocr_engine_deduplicates_identical_frames():
    cfg = IdleUnloadConfig(ocr_minutes=3.0)
    engine = OCREngine(cfg)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("pytesseract.image_to_string", return_value="hello world") as mock_ocr:
        result1 = engine.extract(frame)
        result2 = engine.extract(frame)
    assert mock_ocr.call_count == 1  # called once, deduped on second
    assert result1 == "hello world"
    assert result2 == "hello world"


def test_ocr_engine_processes_different_frames():
    cfg = IdleUnloadConfig(ocr_minutes=3.0)
    engine = OCREngine(cfg)
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = np.ones((100, 100, 3), dtype=np.uint8) * 128
    with patch("pytesseract.image_to_string", side_effect=["text a", "text b"]) as mock_ocr:
        r1 = engine.extract(frame_a)
        r2 = engine.extract(frame_b)
    assert mock_ocr.call_count == 2
    assert r1 == "text a"
    assert r2 == "text b"


def test_stt_engine_transcribes(monkeypatch):
    cfg_idle = IdleUnloadConfig(whisper_minutes=3.0)
    cfg_whisper = WhisperConfig(device="cpu", compute_type="int8")

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " hello there"
    mock_model.transcribe.return_value = ([mock_segment], MagicMock(language="en"))

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        engine = STTEngine(cfg_idle, cfg_whisper)
        audio = np.zeros(16000, dtype=np.float32).tobytes()
        result = engine.transcribe(audio)

    assert result["text"] == "hello there"
    assert result["language"] == "en"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_capture.py -k "ocr or stt" -v
```

Expected: `ImportError: cannot import name 'OCREngine'`

- [ ] **Step 3: Write process/ocr.py**

```python
# process/ocr.py
from __future__ import annotations
import numpy as np
import pytesseract
from PIL import Image
from process.loader import ComponentLoader, IdleTimer
from config import IdleUnloadConfig


class OCREngine:
    def __init__(self, config: IdleUnloadConfig) -> None:
        self._config = config
        self._last_hash: str = ""
        self._last_text: str = ""
        self._timer = IdleTimer(config.ocr_minutes * 60)
        self._timer.reset()

    def extract(self, frame: np.ndarray) -> str:
        import hashlib
        h = hashlib.sha256(frame.tobytes()).hexdigest()
        if h == self._last_hash:
            return self._last_text
        img = Image.fromarray(frame)
        text = pytesseract.image_to_string(img)
        self._last_hash = h
        self._last_text = text
        self._timer.reset()
        return text
```

- [ ] **Step 4: Write process/stt.py**

```python
# process/stt.py
from __future__ import annotations
import io
import numpy as np
from config import IdleUnloadConfig, WhisperConfig
from process.loader import IdleTimer


class STTEngine:
    def __init__(self, idle_config: IdleUnloadConfig, whisper_config: WhisperConfig) -> None:
        self._idle_config = idle_config
        self._whisper_config = whisper_config
        self._model = None
        self._timer = IdleTimer(idle_config.whisper_minutes * 60)

    def _load(self):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            "small",
            device=self._whisper_config.device,
            compute_type=self._whisper_config.compute_type,
        )
        self._timer.reset()

    def transcribe(self, audio_bytes: bytes) -> dict:
        if self._model is None:
            self._load()
        self._timer.reset()
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        segments, info = self._model.transcribe(audio, beam_size=5)
        text = "".join(s.text for s in segments).strip()
        return {"text": text, "language": info.language}

    def check_idle(self) -> None:
        if self._model is not None and self._timer.is_expired():
            del self._model
            self._model = None
```

- [ ] **Step 5: Run all capture tests**

```bash
pytest tests/test_capture.py -v
```

Expected: All tests pass (exact count depends on accumulated tests across tasks)

- [ ] **Step 6: Commit**

```bash
git add process/ocr.py process/stt.py tests/test_capture.py
git commit -m "feat(process): OCR and Whisper added — they lazily wake up, do their job, then immediately nap again. Respect."
```

---

## Task 6: capture/push_to_talk.py

**Files:**
- Create: `capture/push_to_talk.py`
- Modify: `tests/test_capture.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_capture.py`:

```python
from capture.push_to_talk import PushToTalk


def test_push_to_talk_toggle():
    ptt = PushToTalk()
    assert not ptt.is_recording
    ptt.toggle()
    assert ptt.is_recording
    ptt.toggle()
    assert not ptt.is_recording


def test_push_to_talk_stop_when_not_recording():
    ptt = PushToTalk()
    ptt.stop()  # must not raise even when not recording
    assert not ptt.is_recording
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_capture.py -k "push_to_talk" -v
```

Expected: `ImportError: cannot import name 'PushToTalk'`

- [ ] **Step 3: Write capture/push_to_talk.py**

```python
# capture/push_to_talk.py
from __future__ import annotations
import threading
import collections
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


class PushToTalk:
    def __init__(self) -> None:
        self.is_recording = False
        self._frames: list[bytes] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def toggle(self) -> bytes | None:
        with self._lock:
            if self.is_recording:
                return self.stop()
            else:
                self._start()
                return None

    def _start(self) -> None:
        self._frames = []
        self.is_recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=self._callback
        )
        self._stream.start()

    def stop(self) -> bytes | None:
        if not self.is_recording:
            return None
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return b"".join(self._frames) if self._frames else None

    def _callback(self, indata, frames, time_info, status) -> None:
        self._frames.append(indata.copy().tobytes())
```

- [ ] **Step 4: Run all capture + process tests**

```bash
pytest tests/test_capture.py -v --tb=short
```

Expected: All tests pass

- [ ] **Step 5: Final commit for PHANTOM module**

```bash
git add capture/push_to_talk.py tests/test_capture.py
git commit -m "feat(capture): PushToTalk done — toggle to record, toggle again to stop, like a walkie-talkie but for talking to an AI. Over."
```

---

## Task 7: Update build/status.json + request merge review

- [ ] **Step 1: Update build/status.json with PHANTOM progress**

Update `build/status.json` phantom entry:

```json
"phantom": {
  "state": "active",
  "task": "All capture + process files complete. Requesting CIPHER review.",
  "progress": 100,
  "last_update": "<ISO timestamp>",
  "log": [
    "screen.py complete — ScreenWatcher, pixel_hash, diff_ratio",
    "mic.py complete — MicWatcher, VAD, speech buffering",
    "process/loader.py complete — IdleTimer, ComponentLoader",
    "process/scene.py complete — SceneParser, SceneContext",
    "process/ocr.py complete — OCREngine with dedup",
    "process/stt.py complete — STTEngine with lazy load",
    "capture/push_to_talk.py complete",
    "All tests passing"
  ]
}
```

- [ ] **Step 2: Request ARIA password from Nisarg for push**

Present to Nisarg:
```
GIT OPERATION REQUEST
Operation: git push origin feat/capture
Why: PHANTOM module complete — all capture + process files with passing tests. Requesting CIPHER review.
```
Wait for ARIA. Do not push until received.

- [ ] **Step 3: Push feat/capture branch (after ARIA received)**

```bash
# From aria-capture worktree
git checkout -b feat/capture
git push origin feat/capture
```

---

## Self-Review

**Spec coverage:**
- ✅ `capture/screen.py` — ScreenWatcher, pixel_hash, diff_ratio, xdotool title → Task 1
- ✅ `capture/mic.py` — MicWatcher, VAD, speech buffering → Task 2
- ✅ `process/loader.py` — IdleTimer, ComponentLoader → Task 3
- ✅ `process/scene.py` — SceneContext, detect_app, focus EMA → Task 4
- ✅ `process/ocr.py` — pytesseract, frame dedup → Task 5
- ✅ `process/stt.py` — faster-whisper, lazy load → Task 5
- ✅ `capture/push_to_talk.py` — toggle record → Task 6
- ✅ `build/status.json` update + push request → Task 7

**Type consistency:**
- `SceneContext` fields referenced in tests match dataclass definition ✓
- `ComponentLoader.get()` return type consistent ✓
- `Event(type=EventType.X, data={...})` consistent throughout ✓
- `STTEngine.transcribe()` returns `dict` with `text` and `language` keys ✓
