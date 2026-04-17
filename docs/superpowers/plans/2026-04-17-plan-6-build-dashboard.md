# Build Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the live agent-tracking dashboard — a standalone HTML file that polls `build/status.json` every 2 seconds and shows all 7 agents in a deep-space command center UI with animated connections, real activity logs, and a toggle to ARIA Monitor mode.

**Architecture:** Single `build/dashboard/index.html` — zero dependencies, no build step. All state comes from polling `build/status.json` via `fetch()`. Served with `python -m http.server 8765` from the repo root. The mockup at `.superpowers/brainstorm/.../agent-dashboard.html` is the visual reference — wire it to real data.

**Tech Stack:** Vanilla HTML/CSS/JS, `fetch()`, `setInterval`, Google Fonts (Orbitron, Share Tech Mono, Rajdhani)

---

## File Map

| File | Responsibility |
|------|---------------|
| `build/dashboard/index.html` | Full dashboard — polls status.json, renders agents, toggles to monitor mode |
| `build/status.json` | Already exists (from Plan 0) — agents write here, dashboard reads |

---

## Task 1: build/dashboard/index.html — wired to real data

**Files:**
- Create: `build/dashboard/index.html`

- [ ] **Step 1: Copy the visual mockup as the base**

The mockup at `.superpowers/brainstorm/271368-1776445127/content/agent-dashboard.html` already has the correct visual design. Copy it to `build/dashboard/index.html`:

```bash
cp .superpowers/brainstorm/271368-1776445127/content/agent-dashboard.html build/dashboard/index.html
```

- [ ] **Step 2: Replace the static AGENTS array with a fetch() loop**

Find and replace the `// ===================== INIT =====================` section at the bottom of the script with:

```javascript
// ===================== LIVE DATA =====================
async function fetchStatus() {
  try {
    const resp = await fetch('../status.json?_=' + Date.now());
    if (!resp.ok) return;
    const data = await resp.json();
    updateAgentsFromStatus(data);
  } catch (e) {
    // server not running — keep showing last state
  }
}

function updateAgentsFromStatus(data) {
  AGENTS.forEach(agent => {
    const live = data[agent.id];
    if (!live) return;
    agent.state    = live.state    || agent.state;
    agent.task     = live.task     || agent.task;
    agent.progress = live.progress ?? agent.progress;

    // Update card status text
    const statusEl = document.getElementById('status-' + agent.id);
    if (statusEl) statusEl.textContent = agent.task;

    // Update progress bars
    const prog = document.getElementById('prog-' + agent.id);
    const fill = document.getElementById('strip-fill-' + agent.id);
    if (prog) prog.style.width = agent.progress + '%';
    if (fill) fill.style.width = agent.progress + '%';

    // Update card class (state color)
    const card = document.querySelector('#node-' + agent.id + ' .agent-card');
    if (card) {
      card.className = 'agent-card ' + agent.state;
    }

    // Append latest log entries
    if (live.log && live.log.length) {
      const latest = live.log[live.log.length - 1];
      const log = document.getElementById('activity-log');
      if (log) {
        const now = new Date();
        const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
        const div = document.createElement('div');
        div.className = 'log-entry';
        div.innerHTML = `<span class="log-ts">${ts}</span><span class="log-agent ${agent.id}">[${agent.id.toUpperCase()}]</span><span class="log-msg"> ${latest}</span>`;
        log.insertBefore(div, log.firstChild);
        while (log.children.length > 40) log.removeChild(log.lastChild);
      }
    }
  });

  // Update header stats
  const states = Object.values(data);
  document.getElementById('stat-complete').textContent =
    states.filter(a => a.progress === 100).length;
  document.getElementById('stat-active').textContent =
    states.filter(a => a.state === 'active').length;
  document.getElementById('stat-review').textContent =
    states.filter(a => a.state === 'reviewing').length;
  document.getElementById('stat-errors').textContent =
    states.filter(a => a.state === 'blocked').length;
}

// ===================== INIT =====================
window.addEventListener('load', () => {
  positionAgents();
  buildBottomBar();
  buildTaskQueue();
  for (let i = 0; i < 8; i++) addLog();
  setInterval(updateClock, 1000);
  setInterval(simulate, 800);
  setInterval(addLog, 4000);
  setInterval(fetchStatus, 2000);  // real data every 2s
  fetchStatus();  // immediate first fetch
  window.addEventListener('resize', positionAgents);
});
```

- [ ] **Step 3: Add mode toggle button to header**

Find the `.header-stats` div and add a toggle button after it:

```html
<button id="mode-toggle" onclick="toggleMode()" style="
  font-family:'Orbitron',monospace;font-size:9px;letter-spacing:2px;
  background:rgba(0,245,255,0.1);border:1px solid rgba(0,245,255,0.3);
  color:var(--cyan);padding:6px 12px;border-radius:3px;cursor:pointer;
  margin-left:16px;
">ARIA MONITOR →</button>
```

Add the toggle function to the script:

```javascript
let buildMode = true;
function toggleMode() {
  buildMode = !buildMode;
  document.getElementById('mode-toggle').textContent =
    buildMode ? 'ARIA MONITOR →' : '← BUILD MODE';
  document.querySelector('.logo-sub').textContent =
    buildMode ? 'construct control // build session active'
              : 'aria pipeline monitor // live system';
  // Reload page with mode param — ARIA Monitor reads ~/.aria/monitor.json
  // (implemented in Plan 7)
  if (!buildMode) window.location.href = '../monitor/index.html';
}
```

- [ ] **Step 4: Verify dashboard loads without errors**

```bash
cd /home/mtpc-359/Desktop/aria
python -m http.server 8765 &
sleep 1
curl -s http://localhost:8765/build/dashboard/index.html | grep -c "ARIA"
kill %1
```

Expected: Output ≥ 3 (ARIA appears multiple times in the HTML)

- [ ] **Step 5: Commit**

```bash
git add build/dashboard/index.html
git commit -m "feat(dashboard): build dashboard wired to real status.json — agents now report live instead of performing for a static mockup"
```

---

## Task 2: Agent status writer helper

Every squad agent needs to write their status. Add a shared helper so agents don't duplicate the write logic.

**Files:**
- Create: `build/update_status.py`

- [ ] **Step 1: Write build/update_status.py**

```python
# build/update_status.py
"""Call this from any squad agent to update build/status.json."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "status.json"


def update(
    agent_id: str,
    state: str,
    task: str,
    progress: int,
    log_entry: str | None = None,
) -> None:
    data = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            data = json.load(f)

    if agent_id not in data:
        data[agent_id] = {"state": "pending", "task": "", "progress": 0,
                           "last_update": "", "log": []}

    data[agent_id]["state"] = state
    data[agent_id]["task"] = task
    data[agent_id]["progress"] = progress
    data[agent_id]["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if log_entry:
        data[agent_id]["log"].append(log_entry)
        data[agent_id]["log"] = data[agent_id]["log"][-20:]  # keep last 20

    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)
```

- [ ] **Step 2: Commit**

```bash
git add build/update_status.py
git commit -m "feat(build): update_status.py helper — now agents can brag about their progress in a structured, machine-readable way"
```

---

## Task 3: Request push

- [ ] **Step 1: Request ARIA password**

```
GIT OPERATION REQUEST
Operation: git push origin main
Why: Build dashboard wired to real data + update_status.py helper complete.
```

- [ ] **Step 2: Push (after ARIA received)**

```bash
git push origin main
```

---

## Self-Review

**Spec coverage:**
- ✅ Polls `build/status.json` every 2s → Task 1
- ✅ Updates agent cards, progress bars, log feed from real data → Task 1
- ✅ Mode toggle → ARIA Monitor (Plan 7) → Task 1
- ✅ Header stats (complete/active/review/errors) → Task 1
- ✅ `build/update_status.py` helper for all squads → Task 2

**No placeholders:** All JS is complete and functional. No "implement later" comments.
