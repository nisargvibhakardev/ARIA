# ARIA Build Execution Design
**Date:** 2026-04-17  
**Status:** Approved  
**Hardware:** Linux, 20GB RAM, no GPU  

---

## Overview

ARIA (Always Running Intelligence Assistant) is built using a parallel multi-agent strategy — Option A: Module Squads. Seven specialized Claude Code agents work simultaneously in isolated git worktrees, each owning one module end-to-end. A coordinator agent (NEXUS) manages dependencies, triggers reviews, and merges. A reviewer (CIPHER) and tester (SIGMA) run rolling across all squads.

Two dashboards are part of the deliverable:
1. **Build Dashboard** — tracks the 7 agents during construction (`build/dashboard/index.html`)
2. **ARIA System Monitor** — ships with ARIA, shows the live pipeline (`output/monitor.html`)

---

## Hardware

Use the **Linux machine (20GB RAM, no GPU)**.

The Windows machine (16GB RAM, RTX 3050 4GB) is a trap:
- Llama 3.1 8B Q4 requires ~4.65GB VRAM — RTX 3050 has 4GB. It won't fit; falls back to CPU anyway.
- `xdotool`, `pytesseract`, `piper TTS`, `sounddevice` are Linux-native; Windows requires WSL2 with audio/screen capture friction.
- 20GB RAM provides headroom for the full ARIA stack (Whisper + LLM + ChromaDB + Qt peaks at ~7-8GB active).
- The GPU only helps faster-whisper small (~500MB VRAM) — insufficient upside for the platform cost.

---

## Agent Roster

| Agent | Character | Module Ownership | Starts |
|-------|-----------|-----------------|--------|
| 🎯 NEXUS | Coordinator | `core/`, `config.*`, `main.py`, `build/` | Day 0 |
| 👁️ PHANTOM | Capture Squad | `capture/screen.py`, `capture/mic.py`, `capture/push_to_talk.py`, `process/loader.py`, `process/ocr.py`, `process/stt.py`, `process/scene.py` | Day 0 |
| 🧠 MNEMON | Memory Squad | `memory/vector.py`, `memory/structured.py`, `memory/knowledge_graph.py`, `memory/episodic.py`, `memory/classifier.py` | Day 0 |
| 📢 ECHO | Output Squad | `output/overlay.py`, `output/tts.py`, `hotkey.py` | Day 0 |
| ⚖️ ORACLE | Decide Squad | `decide/agent.py`, `decide/energy.py`, `decide/calibrator.py` | When MNEMON ≥ 70% |
| 🔍 CIPHER | Reviewer | No file ownership — reviews PRs as they land | Rolling |
| ⚗️ SIGMA | Tester | `tests/test_capture.py`, `tests/test_memory.py`, `tests/test_output.py`, `tests/test_decide.py`, `tests/test_integration.py` | Rolling |

**Rule:** Every agent owns only the files in their list. NEXUS enforces this. No cross-ownership.

---

## File Contracts

Each agent exposes a public interface. Other agents import only from these contracts — never from internals.

### PHANTOM — Capture + Process
```
Produces:  Event(type=SCREEN_CHANGED, data={screenshot, timestamp})
           Event(type=SPEECH_DETECTED, data={audio_bytes, timestamp})
           Event(type=CONTEXT_CHUNK,   data={text, source, scene_context, timestamp})
           SceneContext(app, task_type, entities, focus_level, delta)
Depends on: core/event_queue.py (NEXUS delivers this on Day 0)
Interface:  ScreenWatcher(config).start()  → pushes SCREEN_CHANGED to queue
            MicWatcher(config).start()     → pushes SPEECH_DETECTED to queue
            ComponentLoader.get_ocr()      → lazy-loaded pytesseract wrapper
            ComponentLoader.get_whisper()  → lazy-loaded faster-whisper wrapper
            SceneParser.parse(screenshot, window_title) → SceneContext
```

### MNEMON — Memory
```
Produces:  VectorStore.add(chunk) / .query(text, k) / .purge()
           StructuredStore.add_task() / .add_commitment() / .get_at_risk()
           KnowledgeGraph.add_node() / .add_edge() / .subgraph(entities)
           EpisodicMemory.summarize(window) / .get_recent()
           MemoryClassifier.route(llm_json) → dispatches to stores
Depends on: nothing (fully standalone)
Interface:  from memory import VectorStore, StructuredStore, KnowledgeGraph
```

### ECHO — Output
```
Produces:  Overlay.show(message, importance, reason) / .hide()
           TTS.speak(text)
           Event(type=HOTKEY_PRESSED) via hotkey.py
Depends on: core/event_queue.py
Interface:  Overlay(config).show(msg, importance, reason)
            TTS(config).speak(text)  — blocking, returns when done
```

### ORACLE — Decide
```
Produces:  DecisionAgent.evaluate(context) → {say, message, type, importance, reason}
           EnergyScheduler.next_interval() → seconds
           InterruptCalibrator.record(event) / .should_suppress(context) → bool
Depends on: memory/* (all MNEMON interfaces)
Interface:  DecisionAgent(memory_clients, config).run() → async loop
```

### CIPHER — Reviewer checklist
```
□ All contract interfaces exported correctly
□ Lazy-load: component unloads after idle timeout
□ No cross-module imports (only core/ and memory/ contracts)
□ SIGMA has ≥ 3 passing tests before CIPHER approves
□ No hardcoded paths — all config-driven via config.yaml
□ Healthcare tenant isolation: no data leaks between contexts
□ No auth changes without NEXUS sign-off
□ No schema migrations without NEXUS sign-off
```

### SIGMA — Test gates (nothing merges without)
```
□ Happy path test
□ Edge case test (empty input, timeout, silence, idle unload)
□ Lazy-load test (loads on demand, unloads after idle timeout)

Per module:
  tests/test_capture.py   → screen diff logic, VAD silence, push-to-talk toggle, lazy-load/unload, SceneContext parsing
  tests/test_memory.py    → ChromaDB CRUD, Ebbinghaus decay, KG edges, SQLite ops
  tests/test_output.py    → overlay show/hide, TTS fallback (aplay→paplay)
  tests/test_decide.py    → two-stage logic, rate limiting, calibrator training
  tests/test_integration.py → full pipeline: screen event → LLM → overlay
```

---

## Execution Sequence

### Day 0 — NEXUS bootstraps (before any squad touches code)

```
1. git init aria/ + initial commit
2. Create 5 git worktrees:
     aria-capture/  → PHANTOM
     aria-memory/   → MNEMON
     aria-output/   → ECHO
     aria-decide/   → ORACLE (created now, used when MNEMON ≥ 70%)
     aria-build/    → NEXUS (core/, config, build/dashboard/)
3. Write and commit:
     core/event_queue.py
     core/events.py
     config.py
     config.yaml
4. All worktrees pull from main → everyone has the foundation
5. Write build/status.json skeleton
6. Launch build/dashboard/index.html
7. Brief each agent with their contract
8. Fire PHANTOM, MNEMON, ECHO simultaneously
```

### NEXUS coordination loop (every 30 minutes)

```
Read build/status.json
→ MNEMON ≥ 70%?           → fire ORACLE  (70% = VectorStore + StructuredStore + KnowledgeGraph implemented with passing tests; Episodic + Classifier may still be in progress)
→ Any module PR-ready?    → assign CIPHER
→ CIPHER approved + SIGMA green? → merge to main, update dashboard
→ Any agent blocked?      → unblock or reassign
→ Append standup entry to build/dashboard/log
```

### Merge order into main (sequential — one at a time)

```
1. core/         ← NEXUS (Day 0)
2. capture/      ← PHANTOM (simplest, finishes first)
3. memory/       ← MNEMON (largest, most critical)
4. output/       ← ECHO
5. decide/       ← ORACLE (requires memory/ merged first)
6. main.py       ← NEXUS wires all stages together
7. build/dashboard/ → switches to ARIA Monitor mode
```

---

## Build Dashboard

**File:** `build/dashboard/index.html` — standalone HTML, zero dependencies, served via `python -m http.server 8765` from the repo root.

**Data source:** `build/status.json` — polled every 2 seconds via `fetch()`.

**Agent status schema:**
```json
{
  "phantom": {
    "state": "active",
    "task": "Building mic.py — VAD buffer logic",
    "progress": 74,
    "last_update": "2026-04-17T14:32:11",
    "log": ["Loaded sounddevice", "VAD aggressiveness=2 confirmed"]
  }
}
```

**Visual design:** Deep-space command center. Starfield background, CRT scanline overlay, agent constellation with animated orbs travelling connection lines. Each agent has a character (emoji + codename + role). CPU mini-graphs, progress bars, live feed panel, task queue panel, mission clock. Dark theme, cyan/amber/coral accent palette, Orbitron + Share Tech Mono fonts.

**Two modes (same app, toggle button):**
- **Build Mode** — 7 agents, constellation layout, construction progress
- **ARIA Monitor Mode** — 5 pipeline stages, ARIA internals

---

## ARIA System Monitor

**File:** `output/monitor.html` — ships with ARIA, same visual aesthetic as build dashboard.

**Data source:** `~/.aria/monitor.json` — ARIA's pipeline writes here every second when active.

```json
{
  "capture": { "vad": "speech", "diff_pct": 0.22, "ocr_loaded": true },
  "process": { "ocr_idle_countdown": 142, "whisper_loaded": false },
  "memory":  { "chroma_docs": 1847, "kg_nodes": 203, "tasks": 4 },
  "decide":  { "interval": 45, "last_say": true, "calibrator_conf": 0.71 },
  "output":  { "overlay_visible": false, "tts_state": "idle" }
}
```

**Signature feature — Interrupt Replay Panel:**  
Every time ARIA fires an interruption, the monitor shows the ghost overlay appearing inside the dashboard itself — exactly as the user saw it. Includes the Generator + Critic reasoning chain in real-time:
```
Generator: YES — commitment deadline in 2h detected
Critic:    SUPPRESS — user in VSCode, focus level HIGH
Final:     NO INTERRUPT
```
This makes ARIA's decision logic fully transparent and observable.

---

## Dependency Map

```
NEXUS (core/) ──┬──▶ PHANTOM (capture/)
                ├──▶ MNEMON  (memory/)  ──▶ ORACLE (decide/)
                └──▶ ECHO    (output/)

CIPHER reviews: PHANTOM → MNEMON → ECHO → ORACLE (in merge order)
SIGMA tests:    alongside each, gates each merge
```

ORACLE is the only hard dependency. All other squads are fully parallel from Day 0.

---

## Non-Goals for This Build Phase

- Multilingual STT optimization (Gujarati/Hindi)
- Mobile companion app
- Cloud sync or backup
- GUI settings panel (config.yaml is sufficient)
- Wake-word (hotkey only)
- Calendar integration
- Fine-tuned smaller LLM
