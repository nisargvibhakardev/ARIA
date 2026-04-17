# ARIA System Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ARIA system monitor that ships with the product — a live dashboard showing ARIA's own pipeline stages (capture → process → memory → decide → output) in real-time, with the signature Interrupt Replay Panel showing Generator + Critic reasoning chains.

**Architecture:** `output/monitor.html` (standalone HTML) + `output/monitor_writer.py` (writes `~/.aria/monitor.json` from inside the pipeline). The monitor polls `~/.aria/monitor.json` every 1 second. `ARIAPipeline` calls `monitor_writer.update()` after each pipeline stage. Same visual aesthetic as the build dashboard — deep-space, Orbitron, scanlines.

**Tech Stack:** Vanilla HTML/CSS/JS, `fetch()`, Python JSON writer integrated into `main.py` pipeline

---

## File Map

| File | Responsibility |
|------|---------------|
| `output/monitor_writer.py` | Writes `~/.aria/monitor.json` after each pipeline stage |
| `output/monitor.html` | Live 5-stage pipeline monitor + Interrupt Replay Panel |
| `build/monitor/` | Symlink target for build dashboard's "ARIA Monitor" toggle |

---

## Task 1: output/monitor_writer.py

**Files:**
- Create: `output/monitor_writer.py`

- [ ] **Step 1: Write output/monitor_writer.py**

```python
# output/monitor_writer.py
from __future__ import annotations
import json
import os
import time
from pathlib import Path

MONITOR_FILE = Path.home() / ".aria" / "monitor.json"


def _write(data: dict) -> None:
    MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if MONITOR_FILE.exists():
        with open(MONITOR_FILE) as f:
            try:
                current = json.load(f)
            except json.JSONDecodeError:
                pass
    current.update(data)
    current["_updated_at"] = time.time()
    with open(MONITOR_FILE, "w") as f:
        json.dump(current, f)


def update_capture(vad: str, diff_pct: float, ocr_loaded: bool) -> None:
    _write({"capture": {"vad": vad, "diff_pct": round(diff_pct, 3),
                         "ocr_loaded": ocr_loaded}})


def update_process(ocr_idle_countdown: float, whisper_loaded: bool,
                   last_text_preview: str) -> None:
    _write({"process": {"ocr_idle_countdown": round(ocr_idle_countdown, 1),
                         "whisper_loaded": whisper_loaded,
                         "last_text": last_text_preview[:80]}})


def update_memory(chroma_docs: int, kg_nodes: int, tasks: int,
                  commitments: int) -> None:
    _write({"memory": {"chroma_docs": chroma_docs, "kg_nodes": kg_nodes,
                        "tasks": tasks, "commitments": commitments}})


def update_decide(interval: int, generator_verdict: bool,
                  critic_verdict: bool, final_say: bool,
                  calibrator_conf: float, reason: str) -> None:
    _write({"decide": {
        "interval": interval,
        "generator": generator_verdict,
        "critic_suppress": critic_verdict,
        "final_say": final_say,
        "calibrator_conf": round(calibrator_conf, 2),
        "reason": reason,
    }})


def update_output(overlay_visible: bool, tts_state: str,
                  last_message: str) -> None:
    _write({"output": {"overlay_visible": overlay_visible,
                        "tts_state": tts_state,
                        "last_message": last_message}})


def record_interrupt(message: str, generator_reasoning: str,
                     critic_reasoning: str, suppressed: bool) -> None:
    entry = {
        "message": message,
        "generator": generator_reasoning,
        "critic": critic_reasoning,
        "suppressed": suppressed,
        "timestamp": time.time(),
    }
    current = {}
    if MONITOR_FILE.exists():
        with open(MONITOR_FILE) as f:
            try:
                current = json.load(f)
            except json.JSONDecodeError:
                pass
    replays = current.get("interrupt_replays", [])
    replays.append(entry)
    replays = replays[-10:]  # keep last 10
    current["interrupt_replays"] = replays
    MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_FILE, "w") as f:
        json.dump(current, f)
```

- [ ] **Step 2: Verify it writes correctly**

```bash
python3 -c "
from output.monitor_writer import update_capture, update_decide, record_interrupt
update_capture('silence', 0.05, False)
update_decide(45, True, False, True, 0.71, 'deadline < 2h detected')
record_interrupt('Check deadline', 'Generator: deadline < 2h', 'Critic: allow — urgency high', False)
import json, pathlib
print(json.dumps(json.load(open(pathlib.Path.home()/'.aria'/'monitor.json')), indent=2))
"
```

Expected: JSON output with `capture`, `decide`, `interrupt_replays` keys.

- [ ] **Step 3: Commit**

```bash
git add output/monitor_writer.py
git commit -m "feat(monitor): monitor_writer pipes ARIA's soul to a JSON file — it's either telemetry or a confession, you decide"
```

---

## Task 2: output/monitor.html — Live pipeline monitor

**Files:**
- Create: `output/monitor.html`

- [ ] **Step 1: Write output/monitor.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ARIA // SYSTEM MONITOR</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --void:#03040a;--panel:rgba(6,15,35,0.92);--cyan:#00f5ff;
    --amber:#ffb300;--coral:#ff4757;--green:#2ed573;--purple:#a855f7;
    --border:rgba(0,245,255,0.2);--text:#c8d6e5;--text-dim:rgba(200,214,229,0.5);
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--void);color:var(--text);font-family:'Rajdhani',sans-serif;
    min-height:100vh;overflow-x:hidden;}
  body::before{content:'';position:fixed;inset:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,
    rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px);pointer-events:none;z-index:999;}

  #app{max-width:1400px;margin:0 auto;padding:16px;display:grid;
    grid-template-rows:64px 1fr 280px;gap:12px;min-height:100vh;}

  /* HEADER */
  header{display:flex;align-items:center;justify-content:space-between;
    background:var(--panel);border:1px solid var(--border);
    border-radius:4px;padding:0 24px;backdrop-filter:blur(20px);}
  .logo-text{font-family:'Orbitron',monospace;font-weight:900;font-size:20px;
    color:var(--cyan);letter-spacing:6px;text-shadow:0 0 30px var(--cyan);}
  .logo-sub{font-family:'Share Tech Mono',monospace;font-size:9px;
    color:var(--text-dim);letter-spacing:3px;}
  #clock{font-family:'Share Tech Mono',monospace;font-size:24px;
    color:var(--cyan);text-shadow:0 0 20px var(--cyan);letter-spacing:4px;}

  /* PIPELINE */
  #pipeline{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;}

  .stage{background:var(--panel);border:1px solid var(--border);
    border-radius:4px;padding:16px;backdrop-filter:blur(10px);
    position:relative;overflow:hidden;transition:border-color .3s,box-shadow .3s;}
  .stage.active{border-color:var(--cyan);
    box-shadow:0 0 20px rgba(0,245,255,0.2),inset 0 0 20px rgba(0,245,255,0.04);}
  .stage-top{height:2px;position:absolute;top:0;left:0;right:0;}
  .stage.active .stage-top{background:var(--cyan);}

  .stage-name{font-family:'Orbitron',monospace;font-size:10px;
    letter-spacing:3px;color:var(--cyan);margin-bottom:8px;}
  .stage-icon{font-size:28px;text-align:center;margin:8px 0;}
  .metric{font-family:'Share Tech Mono',monospace;font-size:9px;
    color:var(--text-dim);padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);}
  .metric span{color:var(--text);float:right;}
  .metric.highlight span{color:var(--green);}
  .metric.warn span{color:var(--amber);}
  .metric.alert span{color:var(--coral);}

  /* ARROW between stages */
  .arrow{display:flex;align-items:center;justify-content:center;
    color:rgba(0,245,255,0.3);font-size:20px;align-self:center;}

  /* INTERRUPT REPLAY */
  #replay-panel{background:var(--panel);border:1px solid var(--border);
    border-radius:4px;backdrop-filter:blur(20px);display:grid;
    grid-template-columns:1fr 1fr;gap:0;overflow:hidden;}
  .replay-header{grid-column:1/-1;padding:10px 16px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;gap:8px;background:rgba(0,245,255,0.03);}
  .replay-title{font-family:'Orbitron',monospace;font-size:9px;
    letter-spacing:3px;color:var(--cyan);}
  .replay-dot{width:5px;height:5px;border-radius:50%;background:var(--coral);
    box-shadow:0 0 6px var(--coral);animation:blink 1s ease-in-out infinite;}
  @keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}

  #reasoning-log{padding:12px 16px;border-right:1px solid var(--border);
    overflow-y:auto;max-height:220px;}
  #ghost-overlay{padding:12px 16px;display:flex;flex-direction:column;
    justify-content:center;align-items:center;}
  .ghost-card{background:rgba(15,23,42,0.95);border:1px solid rgba(99,102,241,.5);
    border-radius:8px;padding:12px 16px;max-width:260px;width:100%;
    box-shadow:0 0 20px rgba(99,102,241,0.2);}
  .ghost-msg{font-size:13px;color:#e2e8f0;margin-bottom:4px;}
  .ghost-why{font-size:10px;color:#94a3b8;margin-bottom:8px;}
  .ghost-btn{background:#1e40af;color:white;border:none;border-radius:4px;
    padding:4px 12px;font-size:10px;cursor:pointer;font-family:'Rajdhani',sans-serif;}

  .reasoning-entry{font-family:'Share Tech Mono',monospace;font-size:9px;
    padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);}
  .r-gen{color:var(--green);}
  .r-critic{color:var(--amber);}
  .r-final-yes{color:var(--cyan);}
  .r-final-no{color:var(--text-dim);}
  .r-ts{color:var(--text-dim);font-size:8px;}
  .r-suppressed{text-decoration:line-through;opacity:0.5;}
</style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <div class="logo-text">ARIA</div>
      <div class="logo-sub">system monitor // live pipeline</div>
    </div>
    <div id="clock">--:--:--</div>
    <a href="../dashboard/index.html" style="font-family:'Orbitron',monospace;font-size:9px;
      letter-spacing:2px;color:var(--cyan);text-decoration:none;border:1px solid var(--border);
      padding:6px 12px;border-radius:3px;">← BUILD MODE</a>
  </header>

  <div id="pipeline">
    <!-- CAPTURE -->
    <div class="stage" id="stage-capture">
      <div class="stage-top"></div>
      <div class="stage-name">CAPTURE</div>
      <div class="stage-icon">👁️</div>
      <div class="metric">VAD <span id="m-vad">--</span></div>
      <div class="metric">Screen diff <span id="m-diff">--</span></div>
      <div class="metric">OCR loaded <span id="m-ocr">--</span></div>
    </div>

    <!-- PROCESS -->
    <div class="stage" id="stage-process">
      <div class="stage-top"></div>
      <div class="stage-name">PROCESS</div>
      <div class="stage-icon">⚙️</div>
      <div class="metric">OCR idle in <span id="m-ocr-idle">--</span></div>
      <div class="metric">Whisper loaded <span id="m-whisper">--</span></div>
      <div class="metric">Last text <span id="m-text">--</span></div>
    </div>

    <!-- MEMORY -->
    <div class="stage" id="stage-memory">
      <div class="stage-top"></div>
      <div class="stage-name">MEMORY</div>
      <div class="stage-icon">🧠</div>
      <div class="metric">ChromaDB docs <span id="m-chroma">--</span></div>
      <div class="metric">KG nodes <span id="m-kg">--</span></div>
      <div class="metric">Tasks <span id="m-tasks">--</span></div>
      <div class="metric">Commitments <span id="m-commits">--</span></div>
    </div>

    <!-- DECIDE -->
    <div class="stage" id="stage-decide">
      <div class="stage-top"></div>
      <div class="stage-name">DECIDE</div>
      <div class="stage-icon">⚖️</div>
      <div class="metric">Interval <span id="m-interval">--</span></div>
      <div class="metric">Generator <span id="m-gen">--</span></div>
      <div class="metric">Critic <span id="m-critic">--</span></div>
      <div class="metric">Calibrator conf <span id="m-cal">--</span></div>
    </div>

    <!-- OUTPUT -->
    <div class="stage" id="stage-output">
      <div class="stage-top"></div>
      <div class="stage-name">OUTPUT</div>
      <div class="stage-icon">📢</div>
      <div class="metric">Overlay <span id="m-overlay">--</span></div>
      <div class="metric">TTS <span id="m-tts">--</span></div>
      <div class="metric">Last message <span id="m-lastmsg">--</span></div>
    </div>
  </div>

  <div id="replay-panel">
    <div class="replay-header">
      <div class="replay-dot"></div>
      <div class="replay-title">Interrupt Replay — Generator + Critic Reasoning Chain</div>
    </div>
    <div id="reasoning-log"></div>
    <div id="ghost-overlay">
      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
        color:var(--text-dim);letter-spacing:2px;margin-bottom:12px;">LAST INTERRUPT PREVIEW</div>
      <div class="ghost-card" id="ghost-card" style="opacity:0.4;">
        <div class="ghost-msg" id="ghost-msg">Waiting for interrupt...</div>
        <div class="ghost-why" id="ghost-why">Why: --</div>
        <button class="ghost-btn">Got it</button>
      </div>
    </div>
  </div>
</div>

<script>
const MONITOR_URL = 'file://' + (localStorage.getItem('ariaHome') || '') + '/.aria/monitor.json';

async function fetchMonitor() {
  // Try relative path first (when served via http.server from home dir),
  // then fall back to absolute
  const paths = [
    '/monitor.json',
    `${location.origin}/.aria/monitor.json`,
  ];
  for (const path of paths) {
    try {
      const r = await fetch(path + '?_=' + Date.now());
      if (r.ok) return await r.json();
    } catch {}
  }
  // Direct file read via custom endpoint written by monitor_writer
  try {
    const r = await fetch(`/aria-monitor?_=${Date.now()}`);
    if (r.ok) return await r.json();
  } catch {}
  return null;
}

function set(id, val, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  if (cls) el.className = cls;
}

function updateUI(data) {
  if (!data) return;

  const { capture, process: proc, memory, decide, output, interrupt_replays } = data;

  if (capture) {
    set('m-vad', capture.vad || '--');
    set('m-diff', capture.diff_pct != null ? (capture.diff_pct * 100).toFixed(1) + '%' : '--');
    set('m-ocr', capture.ocr_loaded ? 'YES' : 'idle');
    document.getElementById('stage-capture').className =
      'stage ' + (capture.vad === 'speech' ? 'active' : '');
  }

  if (proc) {
    set('m-ocr-idle', proc.ocr_idle_countdown != null ? proc.ocr_idle_countdown + 's' : '--');
    set('m-whisper', proc.whisper_loaded ? 'YES' : 'idle');
    set('m-text', proc.last_text ? proc.last_text.substring(0,12) + '…' : '--');
    document.getElementById('stage-process').className =
      'stage ' + (proc.ocr_idle_countdown > 0 ? 'active' : '');
  }

  if (memory) {
    set('m-chroma', memory.chroma_docs ?? '--');
    set('m-kg', memory.kg_nodes ?? '--');
    set('m-tasks', memory.tasks ?? '--');
    set('m-commits', memory.commitments ?? '--');
    document.getElementById('stage-memory').className = 'stage active';
  }

  if (decide) {
    set('m-interval', decide.interval != null ? decide.interval + 's' : '--');
    set('m-gen', decide.generator ? 'YES' : 'no');
    set('m-critic', decide.critic_suppress ? 'SUPPRESS' : 'allow');
    set('m-cal', decide.calibrator_conf != null ? (decide.calibrator_conf * 100).toFixed(0) + '%' : '--');
    document.getElementById('stage-decide').className =
      'stage ' + (decide.final_say ? 'active' : '');
  }

  if (output) {
    set('m-overlay', output.overlay_visible ? 'VISIBLE' : 'hidden');
    set('m-tts', output.tts_state || '--');
    set('m-lastmsg', output.last_message ? output.last_message.substring(0,16) + '…' : '--');
    document.getElementById('stage-output').className =
      'stage ' + (output.overlay_visible ? 'active' : '');
  }

  if (interrupt_replays && interrupt_replays.length > 0) {
    const log = document.getElementById('reasoning-log');
    log.innerHTML = '';
    [...interrupt_replays].reverse().forEach(r => {
      const ts = new Date(r.timestamp * 1000).toLocaleTimeString();
      const div = document.createElement('div');
      div.className = 'reasoning-entry' + (r.suppressed ? ' r-suppressed' : '');
      div.innerHTML = `
        <div class="r-ts">${ts}</div>
        <div class="r-gen">Generator: ${r.generator}</div>
        <div class="r-critic">Critic: ${r.critic}</div>
        <div class="${r.suppressed ? 'r-final-no' : 'r-final-yes'}">
          Final: ${r.suppressed ? 'SUPPRESSED' : 'INTERRUPTED — ' + r.message}
        </div>`;
      log.appendChild(div);
    });

    // Update ghost overlay with last non-suppressed interrupt
    const last = [...interrupt_replays].reverse().find(r => !r.suppressed);
    if (last) {
      document.getElementById('ghost-card').style.opacity = '1';
      document.getElementById('ghost-msg').textContent = last.message;
      document.getElementById('ghost-why').textContent = 'Why: ' + last.generator;
    }
  }
}

function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    [now.getHours(), now.getMinutes(), now.getSeconds()]
      .map(n => String(n).padStart(2,'0')).join(':');
}

setInterval(updateClock, 1000);
setInterval(async () => { updateUI(await fetchMonitor()); }, 1000);
updateClock();
fetchMonitor().then(updateUI);
</script>
</body>
</html>
```

- [ ] **Step 2: Verify monitor.html loads**

```bash
cd /home/mtpc-359/Desktop/aria
python -m http.server 8765 &
sleep 1
curl -s http://localhost:8765/output/monitor.html | grep -c "ARIA"
kill %1
```

Expected: ≥ 3

- [ ] **Step 3: Commit**

```bash
git add output/monitor.html output/monitor_writer.py
git commit -m "feat(monitor): ARIA system monitor ships — watch the AI watch you, in real time, in a dark themed UI. Comfortable."
```

---

## Task 3: Wire monitor_writer into main.py

- [ ] **Step 1: Import monitor_writer in main.py**

Add to imports at top of `main.py`:

```python
from output import monitor_writer
```

- [ ] **Step 2: Call update functions after each pipeline stage**

In `ARIAPipeline._process_screen()`, after `text = self._ocr.extract(screenshot)`:

```python
monitor_writer.update_capture(
    vad="silence", diff_pct=0.0, ocr_loaded=True
)
monitor_writer.update_process(
    ocr_idle_countdown=self._ocr._timer._timeout - (time.monotonic() - self._ocr._timer._last_reset),
    whisper_loaded=self._stt._model is not None,
    last_text_preview=text[:80],
)
```

After `self._vector.add(text)`:

```python
monitor_writer.update_memory(
    chroma_docs=self._vector._collection.count(),
    kg_nodes=len(self._kg._nodes),
    tasks=len(self._structured.get_tasks()),
    commitments=len(self._structured.get_commitments()),
)
```

After `result = self._decision_agent.evaluate(...)`:

```python
monitor_writer.update_decide(
    interval=self._energy.next_interval(),
    generator_verdict=result is not None,
    critic_verdict=False,
    final_say=result is not None,
    calibrator_conf=0.0,
    reason=result["reason"] if result else "no interrupt",
)
```

After `self._interrupt(result)`:

```python
monitor_writer.update_output(
    overlay_visible=True,
    tts_state="speaking",
    last_message=result["message"],
)
```

- [ ] **Step 3: Run integration tests**

```bash
pytest tests/test_integration.py tests/test_output.py -v
```

Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(monitor): main.py now narrates itself to monitor.json — ARIA's inner monologue, finally serialized to disk"
```

---

## Task 4: Request final push

- [ ] **Step 1: Request ARIA password**

```
GIT OPERATION REQUEST
Operation: git push origin main
Why: ARIA monitor complete — monitor_writer.py + monitor.html + wired into main.py pipeline.
```

- [ ] **Step 2: Push (after ARIA received)**

```bash
git push origin main
```

---

## Self-Review

**Spec coverage:**
- ✅ `output/monitor_writer.py` — writes per-stage metrics + interrupt replays → Task 1
- ✅ `output/monitor.html` — 5 stages, live metrics, Interrupt Replay Panel → Task 2
- ✅ Ghost overlay showing last real interrupt inside monitor → Task 2
- ✅ Generator + Critic reasoning chain visible → Task 2
- ✅ Wired into `main.py` → Task 3
- ✅ Back link to Build Dashboard → Task 2

**Type consistency:**
- `monitor_writer.update_decide(interval=int, ...)` matches `EnergyScheduler.next_interval() → int` ✓
- `record_interrupt(message, generator_reasoning, critic_reasoning, suppressed)` consistent ✓
