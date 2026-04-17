# NEXUS Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the entire ARIA project — Python scaffolding, core event system, config, git worktrees for all squads, and the build status file that powers the live dashboard.

**Architecture:** Pure Python 3.11+ asyncio event bus at the center. All pipeline stages communicate exclusively through `EventQueue`. Config is a typed dataclass loaded from `config.yaml`. Git worktrees give each agent a fully isolated workspace with zero merge conflicts.

**Tech Stack:** Python 3.11+, asyncio, PyYAML, pytest, pytest-asyncio, git worktrees

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All Python dependencies for the full project |
| `core/__init__.py` | Package marker |
| `core/events.py` | `EventType` enum + `Event` dataclass |
| `core/event_queue.py` | `EventQueue` — asyncio wrapper, thread-safe `put_nowait` |
| `config.py` | Typed `Config` dataclass + `Config.from_yaml()` loader |
| `config.yaml` | All tunable settings with defaults |
| `build/status.json` | Live agent status — polled by dashboard every 2s |
| `.gitignore` | Excludes venv, `__pycache__`, `.superpowers/`, `~/.aria/` |
| `tests/__init__.py` | Package marker |
| `tests/test_events.py` | Tests for `Event` + `EventType` |
| `tests/test_event_queue.py` | Tests for `EventQueue` (async) |
| `tests/test_config.py` | Tests for `Config.from_yaml()` |

---

## Task 1: Python project skeleton + requirements.txt

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `core/__init__.py`
- Create: `tests/__init__.py`
- Create: `capture/__init__.py`
- Create: `process/__init__.py`
- Create: `memory/__init__.py`
- Create: `decide/__init__.py`
- Create: `output/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
# Capture
mss>=9.0.1
pytesseract>=0.3.10
sounddevice>=0.4.6
webrtcvad>=2.0.10

# Speech-to-text
faster-whisper>=1.0.0

# Memory
chromadb>=0.5.0
sentence-transformers>=3.0.0

# LLM
ollama>=0.2.0

# TTS
piper-tts>=1.2.0

# UI
PyQt6>=6.7.0

# Hotkey
pynput>=1.7.6

# Utilities
PyYAML>=6.0.1
numpy>=1.26.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-qt>=4.4.0
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*.so
.venv/
venv/
env/
*.egg-info/
dist/
build/
.superpowers/
~/.aria/
*.db
*.sqlite
chroma_data/
.pytest_cache/
.coverage
htmlcov/
```

- [ ] **Step 3: Create all package `__init__.py` files**

```bash
mkdir -p core capture process memory decide output tests build
touch core/__init__.py capture/__init__.py process/__init__.py \
      memory/__init__.py decide/__init__.py output/__init__.py \
      tests/__init__.py
```

- [ ] **Step 4: Create virtual environment and install deps**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install without error. `pip list` shows `chromadb`, `faster-whisper`, `PyQt6`, etc.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore core/__init__.py capture/__init__.py \
        process/__init__.py memory/__init__.py decide/__init__.py \
        output/__init__.py tests/__init__.py
git commit -m "chore(scaffold): summoned a forest of __init__.py files into existence, pray they multiply"
```

---

## Task 2: core/events.py

**Files:**
- Create: `core/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_events.py -v
```

Expected: `ImportError: cannot import name 'Event' from 'core.events'`

- [ ] **Step 3: Write core/events.py**

```python
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


@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_events.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add core/events.py tests/test_events.py
git commit -m "feat(core): events.py born — it knows 4 things and timestamps itself like a paranoid journalist"
```

---

## Task 3: core/event_queue.py

**Files:**
- Create: `core/event_queue.py`
- Create: `tests/test_event_queue.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_event_queue.py -v
```

Expected: `ImportError: cannot import name 'EventQueue'`

- [ ] **Step 3: Write core/event_queue.py**

```python
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
```

- [ ] **Step 4: Add pytest-asyncio config (required for async tests)**

Add `pytest.ini` at project root:

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_event_queue.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add core/event_queue.py tests/test_event_queue.py pytest.ini
git commit -m "feat(core): event queue online — events now travel through it like commuters who never look up from their phones"
```

---

## Task 4: config.py + config.yaml

**Files:**
- Create: `config.py`
- Create: `config.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import textwrap
import tempfile
import os
import pytest
from config import Config


def test_default_config():
    cfg = Config()
    assert cfg.screen.interval_seconds == 3.0
    assert cfg.screen.diff_threshold == 0.15
    assert cfg.mic.vad_aggressiveness == 2
    assert cfg.memory.rolling_days == 7
    assert cfg.llm.model == "llama3.1:8b"
    assert cfg.hotkey == "ctrl+shift+space"
    assert cfg.overlay.auto_dismiss_seconds == 8


def test_from_yaml_overrides_values(tmp_path):
    yaml_content = textwrap.dedent("""
        screen:
          interval_seconds: 5.0
          diff_threshold: 0.20
        llm:
          model: llama3.2:3b
        hotkey: ctrl+shift+a
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    cfg = Config.from_yaml(str(config_file))
    assert cfg.screen.interval_seconds == 5.0
    assert cfg.screen.diff_threshold == 0.20
    assert cfg.llm.model == "llama3.2:3b"
    assert cfg.hotkey == "ctrl+shift+a"
    assert cfg.mic.vad_aggressiveness == 2  # unset = default


def test_from_yaml_missing_file_returns_defaults():
    cfg = Config.from_yaml("/nonexistent/path/config.yaml")
    assert cfg.screen.interval_seconds == 3.0


def test_ebbinghaus_thresholds():
    cfg = Config()
    assert cfg.memory.initial_stability_days == 1.0
    assert cfg.memory.important_stability_days == 30.0
    assert cfg.memory.ebbinghaus_purge_threshold == 0.2


def test_idle_unload_minutes():
    cfg = Config()
    assert cfg.idle_unload.ocr_minutes == 3.0
    assert cfg.idle_unload.whisper_minutes == 3.0
    assert cfg.idle_unload.tts_minutes == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Write config.py**

```python
# config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ScreenConfig:
    interval_seconds: float = 3.0
    diff_threshold: float = 0.15
    monitor_index: int = 1


@dataclass
class MicConfig:
    vad_aggressiveness: int = 2


@dataclass
class MemoryConfig:
    rolling_days: int = 7
    context_window_seconds: int = 120
    ebbinghaus_purge_threshold: float = 0.2
    initial_stability_days: float = 1.0
    important_stability_days: float = 30.0


@dataclass
class EpisodicConfig:
    summary_interval_minutes: int = 30


@dataclass
class LLMConfig:
    model: str = "llama3.1:8b"
    keep_alive: str = "3m"
    response_language: str = "english"


@dataclass
class EnergyConfig:
    min_interval_seconds: int = 20
    max_interval_seconds: int = 120


@dataclass
class CalibratorConfig:
    min_samples: int = 50
    retrain_every: int = 10


@dataclass
class IdleUnloadConfig:
    ocr_minutes: float = 3.0
    whisper_minutes: float = 3.0
    tts_minutes: float = 2.0


@dataclass
class WhisperConfig:
    device: str = "cpu"
    compute_type: str = "int8"


@dataclass
class OverlayConfig:
    position: str = "bottom-right"
    auto_dismiss_seconds: int = 8


@dataclass
class SystemConfig:
    nice_level: int = 10


@dataclass
class Config:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    mic: MicConfig = field(default_factory=MicConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    calibrator: CalibratorConfig = field(default_factory=CalibratorConfig)
    idle_unload: IdleUnloadConfig = field(default_factory=IdleUnloadConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    hotkey: str = "ctrl+shift+space"
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> Config:
        if not Path(path).exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        _section_types = {
            "screen": ScreenConfig, "mic": MicConfig, "memory": MemoryConfig,
            "episodic": EpisodicConfig, "llm": LLMConfig, "energy": EnergyConfig,
            "calibrator": CalibratorConfig, "idle_unload": IdleUnloadConfig,
            "whisper": WhisperConfig, "overlay": OverlayConfig, "system": SystemConfig,
        }
        for key, value in data.items():
            if key in _section_types and isinstance(value, dict):
                section = getattr(cfg, key)
                for k, v in value.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
```

- [ ] **Step 4: Write config.yaml**

```yaml
screen:
  interval_seconds: 3
  diff_threshold: 0.15
  monitor_index: 1

mic:
  vad_aggressiveness: 2

memory:
  rolling_days: 7
  context_window_seconds: 120
  ebbinghaus_purge_threshold: 0.2
  initial_stability_days: 1.0
  important_stability_days: 30.0

episodic:
  summary_interval_minutes: 30

llm:
  model: llama3.1:8b
  keep_alive: 3m
  response_language: english

energy:
  min_interval_seconds: 20
  max_interval_seconds: 120

calibrator:
  min_samples: 50
  retrain_every: 10

idle_unload:
  ocr_minutes: 3.0
  whisper_minutes: 3.0
  tts_minutes: 2.0

whisper:
  device: cpu
  compute_type: int8

hotkey: ctrl+shift+space

overlay:
  position: bottom-right
  auto_dismiss_seconds: 8

system:
  nice_level: 10
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add config.py config.yaml tests/test_config.py
git commit -m "feat(config): 12 dataclasses walk into a YAML file — only one walks out, fully typed and slightly smug"
```

---

## Task 5: Run full test suite + verify green baseline

**Files:** None new — verification only

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected:
```
tests/test_config.py::test_default_config PASSED
tests/test_config.py::test_from_yaml_overrides_values PASSED
tests/test_config.py::test_from_yaml_missing_file_returns_defaults PASSED
tests/test_config.py::test_ebbinghaus_thresholds PASSED
tests/test_config.py::test_idle_unload_minutes PASSED
tests/test_event_queue.py::test_put_and_get PASSED
tests/test_event_queue.py::test_put_nowait_and_get PASSED
tests/test_event_queue.py::test_fifo_order PASSED
tests/test_event_queue.py::test_task_done PASSED
tests/test_event_queue.py::test_empty_queue_blocks PASSED
tests/test_events.py::test_event_type_values PASSED
tests/test_events.py::test_event_defaults_timestamp PASSED
tests/test_events.py::test_event_explicit_timestamp PASSED
tests/test_events.py::test_event_data_preserved PASSED
tests/test_events.py::test_event_type_is_enum PASSED
15 passed
```

- [ ] **Step 2: Commit baseline marker**

```bash
git commit --allow-empty -m "test(core): 15/15 green — the foundation doesn't collapse, unlike my confidence during code review"
```

---

## Task 6: Git worktrees for all squads

**Files:**
- Create: `build/status.json`
- Create: `build/README.md`

This task runs from the **main repo directory** (`/home/mtpc-359/Desktop/aria`).

- [ ] **Step 1: Create worktrees for all squads**

```bash
git worktree add ../aria-capture main
git worktree add ../aria-memory main
git worktree add ../aria-output main
git worktree add ../aria-decide main
```

Expected: 4 directories created at `../aria-capture`, `../aria-memory`, `../aria-output`, `../aria-decide`

- [ ] **Step 2: Verify worktrees**

```bash
git worktree list
```

Expected output:
```
/home/mtpc-359/Desktop/aria         [main]
/home/mtpc-359/Desktop/aria-capture [main]
/home/mtpc-359/Desktop/aria-memory  [main]
/home/mtpc-359/Desktop/aria-output  [main]
/home/mtpc-359/Desktop/aria-decide  [main]
```

- [ ] **Step 3: Create build/status.json skeleton**

```json
{
  "nexus": {
    "state": "active",
    "task": "Infrastructure complete — squads deployed",
    "progress": 100,
    "last_update": "",
    "log": []
  },
  "phantom": {
    "state": "pending",
    "task": "Awaiting Day 0 completion",
    "progress": 0,
    "last_update": "",
    "log": []
  },
  "mnemon": {
    "state": "pending",
    "task": "Awaiting Day 0 completion",
    "progress": 0,
    "last_update": "",
    "log": []
  },
  "echo": {
    "state": "pending",
    "task": "Awaiting Day 0 completion",
    "progress": 0,
    "last_update": "",
    "log": []
  },
  "oracle": {
    "state": "blocked",
    "task": "Waiting for MNEMON >= 70%",
    "progress": 0,
    "last_update": "",
    "log": []
  },
  "cipher": {
    "state": "idle",
    "task": "Standing by for first PR",
    "progress": 0,
    "last_update": "",
    "log": []
  },
  "sigma": {
    "state": "idle",
    "task": "Standing by for first module",
    "progress": 0,
    "last_update": "",
    "log": []
  }
}
```

Save as `build/status.json`.

- [ ] **Step 4: Create build/README.md**

```markdown
# Build Infrastructure

## Agent Status
Agents write their status to `build/status.json` every 60 seconds.

## Dashboard
Serve from repo root:
    python -m http.server 8765

Open: http://localhost:8765/build/dashboard/index.html

## Worktrees
| Agent   | Directory          | Branch        |
|---------|--------------------|---------------|
| PHANTOM | ../aria-capture    | feat/capture  |
| MNEMON  | ../aria-memory     | feat/memory   |
| ECHO    | ../aria-output     | feat/output   |
| ORACLE  | ../aria-decide     | feat/decide   |
```

- [ ] **Step 5: Commit**

```bash
git add build/status.json build/README.md
git commit -m "feat(build): worktrees spawned — 4 parallel dimensions of ARIA now exist, each blissfully unaware of the others"
```

---

## Task 7: Push everything to GitHub

**Files:** None — git operation only

- [ ] **Step 1: Request ARIA password from Nisarg before pushing**

Present to Nisarg:
```
GIT OPERATION REQUEST
Operation: git push origin main
Why: Push all infrastructure (core, config, tests, worktree setup) to GitHub
```
Wait for ARIA password. Do not push until received.

- [ ] **Step 2: Push (after ARIA received)**

```bash
git -c credential.helper='' -c "credential.helper=store --file=/home/mtpc-359/.config/git-aria/credentials" push origin main
```

Expected:
```
To https://github.com/nisargvibhakardev/ARIA.git
   xxxxxxx..xxxxxxx  main -> main
```

---

## Task 8: Brief squad agents

This task is coordination — no code. NEXUS sends each squad agent their contract and plan location.

- [ ] **Step 1: Confirm squad plans exist (or note they are pending)**

The following plans must be written (in parallel with infrastructure) before squads start:
- `docs/superpowers/plans/2026-04-17-plan-1-phantom-capture-process.md`
- `docs/superpowers/plans/2026-04-17-plan-2-mnemon-memory.md`
- `docs/superpowers/plans/2026-04-17-plan-3-echo-output.md`
- `docs/superpowers/plans/2026-04-17-plan-4-oracle-decide.md`

- [ ] **Step 2: Fire squads**

Open 3 new Claude Code sessions (PHANTOM, MNEMON, ECHO simultaneously):
- Each session opens its worktree directory
- Each reads its plan
- Each updates `build/status.json` with their progress every 60s

ORACLE session opens when MNEMON's `build/status.json` progress ≥ 70.

---

## Self-Review

**Spec coverage check:**
- ✅ Python project structure + requirements.txt → Task 1
- ✅ core/events.py (EventType, Event) → Task 2
- ✅ core/event_queue.py → Task 3
- ✅ config.py + config.yaml → Task 4
- ✅ Full test suite baseline → Task 5
- ✅ Git worktrees for all squads → Task 6
- ✅ build/status.json → Task 6
- ✅ Push to GitHub (with ARIA password gate) → Task 7
- ✅ Squad briefing → Task 8

**Placeholder scan:** No TBDs, no "implement later", all steps have exact code.

**Type consistency:**
- `Event(type=EventType.X, data={})` used consistently across all test files ✓
- `Config.from_yaml(path)` signature consistent in tests and implementation ✓
- `EventQueue.put_nowait()` / `.get()` / `.task_done()` consistent ✓
