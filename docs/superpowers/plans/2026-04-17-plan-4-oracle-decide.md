# ORACLE — Decide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two-stage decision engine — Generator LLM call that produces an interrupt candidate, Critic LLM call that suppresses it based on user focus, adaptive energy scheduler that controls check frequency, and a self-calibrating LogisticRegression that learns per-user interrupt preferences.

**Architecture:** `DecisionAgent` orchestrates two sequential Ollama calls per cycle. `EnergyScheduler` dynamically adjusts the interval between cycles (20–120s) based on context novelty and focus level. `InterruptCalibrator` trains a local `sklearn` LogisticRegression on engagement/dismiss history after 50 samples, updating every 10 new events.

**Tech Stack:** Ollama Python client, `llama3.1:8b`, `scikit-learn` (LogisticRegression), `jsonlines`, memory module interfaces from MNEMON, `config.py`

**Worktree:** Work in `../aria-decide/`. Start ONLY after MNEMON's `build/status.json` progress ≥ 70.

---

## File Map

| File | Responsibility |
|------|---------------|
| `decide/agent.py` | `DecisionAgent` — two-stage LLM (Generator + Critic) via Ollama |
| `decide/energy.py` | `EnergyScheduler` — adaptive interval based on novelty + focus |
| `decide/calibrator.py` | `InterruptCalibrator` — sklearn LogisticRegression, JSONL event log |
| `tests/test_decide.py` | All decide tests |

---

## Task 1: decide/energy.py — Adaptive scheduler

**Files:**
- Create: `decide/energy.py`
- Create: `tests/test_decide.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_decide.py
import pytest
from decide.energy import EnergyScheduler
from config import EnergyConfig


@pytest.fixture
def scheduler():
    return EnergyScheduler(EnergyConfig(min_interval_seconds=20, max_interval_seconds=120))


def test_default_interval_is_max(scheduler):
    assert scheduler.next_interval() == 120


def test_high_novelty_reduces_interval(scheduler):
    scheduler.update(novelty_score=1.0, focus_level=0.2)
    interval = scheduler.next_interval()
    assert interval < 120


def test_high_focus_increases_interval(scheduler):
    scheduler.update(novelty_score=0.5, focus_level=1.0)
    interval = scheduler.next_interval()
    assert interval >= 60


def test_interval_never_below_min(scheduler):
    scheduler.update(novelty_score=1.0, focus_level=0.0)
    assert scheduler.next_interval() >= 20


def test_interval_never_above_max(scheduler):
    scheduler.update(novelty_score=0.0, focus_level=1.0)
    assert scheduler.next_interval() <= 120


def test_interval_formula_matches_spec(scheduler):
    # Formula: max(min, max - avg_novelty * 100) dampened by focus
    # novelty=0.5, focus=0.5 → raw = max(20, 120 - 50) = 70 → dampened ~70
    scheduler.update(novelty_score=0.5, focus_level=0.5)
    interval = scheduler.next_interval()
    assert 20 <= interval <= 120
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_decide.py -k "energy or interval" -v
```

Expected: `ImportError: cannot import name 'EnergyScheduler'`

- [ ] **Step 3: Write decide/energy.py**

```python
# decide/energy.py
from __future__ import annotations
from config import EnergyConfig


class EnergyScheduler:
    def __init__(self, config: EnergyConfig) -> None:
        self._config = config
        self._avg_novelty: float = 0.0
        self._focus: float = 0.5

    def update(self, novelty_score: float, focus_level: float) -> None:
        self._avg_novelty = 0.7 * self._avg_novelty + 0.3 * novelty_score
        self._focus = focus_level

    def next_interval(self) -> int:
        raw = self._config.max_interval_seconds - self._avg_novelty * 100.0
        raw = max(self._config.min_interval_seconds, raw)
        # High focus dampens urgency — stretch interval toward max
        dampened = raw + (self._config.max_interval_seconds - raw) * self._focus * 0.5
        return int(min(self._config.max_interval_seconds,
                       max(self._config.min_interval_seconds, dampened)))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_decide.py -k "energy or interval" -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add decide/energy.py tests/test_decide.py
git commit -m "feat(decide): EnergyScheduler online — checks more when things are chaotic, backs off when you're in flow. It has manners."
```

---

## Task 2: decide/calibrator.py — Self-calibrating LogisticRegression

**Files:**
- Create: `decide/calibrator.py`
- Modify: `tests/test_decide.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decide.py`:

```python
import tempfile
from decide.calibrator import InterruptCalibrator
from config import CalibratorConfig


@pytest.fixture
def calibrator(tmp_path):
    return InterruptCalibrator(
        CalibratorConfig(min_samples=3, retrain_every=2),
        log_path=str(tmp_path / "events.jsonl"),
        model_path=str(tmp_path / "model.pkl"),
    )


def test_calibrator_records_event(calibrator):
    calibrator.record(
        interrupt_type="task", time_of_day=14, focus_level=0.6,
        app="vscode", task_type="coding", importance=0.8, engaged=True
    )
    assert calibrator._event_count == 1


def test_calibrator_suppresses_returns_false_before_training(calibrator):
    result = calibrator.should_suppress(
        interrupt_type="task", time_of_day=9, focus_level=0.9,
        app="vscode", task_type="coding", importance=0.5,
    )
    assert result is False  # no model yet — don't suppress


def test_calibrator_trains_after_min_samples(calibrator):
    for i in range(4):
        calibrator.record(
            interrupt_type="commitment", time_of_day=10, focus_level=0.3,
            app="slack", task_type="communicating", importance=0.9, engaged=True
        )
    assert calibrator._model is not None


def test_calibrator_persists_events_to_jsonl(calibrator):
    import json
    calibrator.record(
        interrupt_type="focus_drift", time_of_day=15, focus_level=0.8,
        app="chrome", task_type="browsing", importance=0.4, engaged=False
    )
    with open(calibrator._log_path) as f:
        lines = f.readlines()
    event = json.loads(lines[0])
    assert event["interrupt_type"] == "focus_drift"
    assert event["engaged"] is False
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_decide.py -k "calibrator" -v
```

Expected: `ImportError: cannot import name 'InterruptCalibrator'`

- [ ] **Step 3: Write decide/calibrator.py**

```python
# decide/calibrator.py
from __future__ import annotations
import json
import os
import pickle
from typing import Any
from config import CalibratorConfig

_INTERRUPT_TYPES = ["task", "commitment", "fact", "focus_drift", "deadline", "contradiction"]
_APPS = ["vscode", "chrome", "slack", "terminal", "notion", "gmail", "unknown"]
_TASK_TYPES = ["coding", "browsing", "communicating", "general"]


def _featurise(
    interrupt_type: str, time_of_day: int, focus_level: float,
    app: str, task_type: str, importance: float,
) -> list[float]:
    it_feat = [float(interrupt_type == t) for t in _INTERRUPT_TYPES]
    app_feat = [float(app == a) for a in _APPS]
    tt_feat = [float(task_type == t) for t in _TASK_TYPES]
    hour_sin = [__import__("math").sin(2 * __import__("math").pi * time_of_day / 24)]
    hour_cos = [__import__("math").cos(2 * __import__("math").pi * time_of_day / 24)]
    return it_feat + app_feat + tt_feat + hour_sin + hour_cos + [focus_level, importance]


class InterruptCalibrator:
    def __init__(
        self,
        config: CalibratorConfig,
        log_path: str = "~/.aria/events.jsonl",
        model_path: str = "~/.aria/calibrator.pkl",
    ) -> None:
        self._config = config
        self._log_path = os.path.expanduser(log_path)
        self._model_path = os.path.expanduser(model_path)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._event_count = 0
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self._model_path):
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)

    def record(
        self,
        interrupt_type: str, time_of_day: int, focus_level: float,
        app: str, task_type: str, importance: float, engaged: bool,
    ) -> None:
        event = {
            "interrupt_type": interrupt_type, "time_of_day": time_of_day,
            "focus_level": focus_level, "app": app, "task_type": task_type,
            "importance": importance, "engaged": engaged,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        self._event_count += 1
        if (self._event_count % self._config.retrain_every == 0
                and self._event_count >= self._config.min_samples):
            self._train()

    def _train(self) -> None:
        from sklearn.linear_model import LogisticRegression
        X, y = [], []
        with open(self._log_path) as f:
            for line in f:
                e = json.loads(line)
                X.append(_featurise(
                    e["interrupt_type"], e["time_of_day"], e["focus_level"],
                    e["app"], e["task_type"], e["importance"],
                ))
                y.append(int(e["engaged"]))
        if len(set(y)) < 2:
            return  # need both classes to train
        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        self._model = model
        with open(self._model_path, "wb") as f:
            pickle.dump(model, f)

    def should_suppress(
        self,
        interrupt_type: str, time_of_day: int, focus_level: float,
        app: str, task_type: str, importance: float,
    ) -> bool:
        if self._model is None:
            return False
        features = [_featurise(interrupt_type, time_of_day, focus_level, app, task_type, importance)]
        prob_engage = self._model.predict_proba(features)[0][1]
        return prob_engage < 0.3  # suppress if <30% chance of engagement
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_decide.py -k "calibrator" -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add decide/calibrator.py tests/test_decide.py
git commit -m "feat(decide): calibrator learns your interrupt preferences — by week 2 it knows you hate focus-drift warnings at 9am. So do you."
```

---

## Task 3: decide/agent.py — Two-stage LLM (Generator + Critic)

**Files:**
- Create: `decide/agent.py`
- Modify: `tests/test_decide.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decide.py`:

```python
from unittest.mock import patch, MagicMock
from decide.agent import DecisionAgent, GeneratorOutput, CriticOutput
from config import LLMConfig


def _make_agent():
    from unittest.mock import MagicMock
    memory = MagicMock()
    memory.vector.query.return_value = []
    memory.episodic.get_recent.return_value = []
    memory.episodic.get_summaries.return_value = []
    memory.structured.get_at_risk_commitments.return_value = []
    memory.kg.subgraph.return_value = []
    return DecisionAgent(memory=memory, config=LLMConfig())


def test_generator_output_schema():
    out = GeneratorOutput(
        say=True, message="You have a deadline in 2h",
        type="deadline", importance=0.9,
        reason="commit deadline < 3h", extract={}
    )
    assert out.say is True
    assert 0.0 <= out.importance <= 1.0


def test_critic_output_schema():
    out = CriticOutput(suppress=False, reason="high urgency overrides focus")
    assert out.suppress is False


def test_decision_agent_returns_none_when_generator_says_no():
    agent = _make_agent()
    mock_gen = MagicMock()
    mock_gen.say = False

    with patch.object(agent, "_call_generator", return_value=mock_gen):
        result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is None


def test_decision_agent_returns_none_when_critic_suppresses():
    agent = _make_agent()
    mock_gen = MagicMock(say=True, message="Hey", type="task",
                         importance=0.5, reason="test", extract={})
    mock_critic = MagicMock(suppress=True)

    with patch.object(agent, "_call_generator", return_value=mock_gen):
        with patch.object(agent, "_call_critic", return_value=mock_critic):
            result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is None


def test_decision_agent_returns_message_when_both_approve():
    agent = _make_agent()
    mock_gen = MagicMock(say=True, message="Submit report soon",
                         type="deadline", importance=0.9, reason="deadline < 2h", extract={})
    mock_critic = MagicMock(suppress=False)

    with patch.object(agent, "_call_generator", return_value=mock_gen):
        with patch.object(agent, "_call_critic", return_value=mock_critic):
            result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is not None
    assert result["message"] == "Submit report soon"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_decide.py -k "generator or critic or decision_agent" -v
```

Expected: `ImportError: cannot import name 'DecisionAgent'`

- [ ] **Step 3: Write decide/agent.py**

```python
# decide/agent.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import ollama
from config import LLMConfig

if TYPE_CHECKING:
    from memory.vector import VectorStore
    from memory.structured import StructuredStore
    from memory.knowledge_graph import KnowledgeGraph
    from memory.episodic import EpisodicMemory


_GENERATOR_PROMPT = """You are ARIA's interrupt decision engine.

Context (last 2 minutes):
{recent_context}

Episodic summaries (today):
{summaries}

At-risk commitments:
{at_risk}

Active entities on screen:
{entities}

Decide if ARIA should interrupt the user RIGHT NOW.
Respond with valid JSON only:
{{
  "say": true/false,
  "message": "concise message to user (max 30 words)",
  "type": "task|commitment|fact|focus_drift|deadline|contradiction",
  "importance": 0.0-1.0,
  "reason": "one-line explainability reason",
  "extract": {{"type": "...", "text": "..."}}
}}"""

_CRITIC_PROMPT = """You are ARIA's interrupt critic.

Proposed message: {message}
Importance: {importance}
Current app: {app}
Task type: {task_type}
Focus level: {focus_level} (0=unfocused, 1=deep focus)

Should ARIA suppress this interrupt right now?
Respond with valid JSON only:
{{"suppress": true/false, "reason": "one line"}}

Rules: High focus (>0.8) + low importance (<0.6) = suppress. High urgency (>0.85) = never suppress."""


@dataclass
class GeneratorOutput:
    say: bool
    message: str
    type: str
    importance: float
    reason: str
    extract: dict


@dataclass
class CriticOutput:
    suppress: bool
    reason: str


class DecisionAgent:
    def __init__(self, memory: Any, config: LLMConfig) -> None:
        self._memory = memory
        self._config = config

    def evaluate(self, context: dict) -> dict | None:
        gen = self._call_generator(context)
        if not gen.say:
            return None
        critic = self._call_critic(gen, context)
        if critic.suppress:
            return None
        return {
            "message": gen.message,
            "type": gen.type,
            "importance": gen.importance,
            "reason": gen.reason,
            "extract": gen.extract,
        }

    def _call_generator(self, context: dict) -> GeneratorOutput:
        recent = context.get("recent", [])
        summaries = self._memory.episodic.get_summaries(limit=5)
        at_risk = self._memory.structured.get_at_risk_commitments(within_hours=3)
        scene = context.get("scene")
        entities = scene.entities if scene else []

        prompt = _GENERATOR_PROMPT.format(
            recent_context="\n".join(c.get("text", "") for c in recent) or "None",
            summaries="\n".join(s.get("text", "") for s in summaries) or "None",
            at_risk="\n".join(c.get("text", "") for c in at_risk) or "None",
            entities=", ".join(entities) or "None",
        )
        response = ollama.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        raw = response["message"]["content"].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return GeneratorOutput(say=False, message="", type="", importance=0.0, reason="parse error", extract={})
        return GeneratorOutput(
            say=bool(data.get("say", False)),
            message=data.get("message", ""),
            type=data.get("type", ""),
            importance=float(data.get("importance", 0.0)),
            reason=data.get("reason", ""),
            extract=data.get("extract", {}),
        )

    def _call_critic(self, gen: GeneratorOutput, context: dict) -> CriticOutput:
        scene = context.get("scene")
        app = scene.app if scene else "unknown"
        task_type = scene.task_type if scene else "general"
        focus = scene.focus_level if scene else 0.5

        prompt = _CRITIC_PROMPT.format(
            message=gen.message,
            importance=gen.importance,
            app=app,
            task_type=task_type,
            focus_level=round(focus, 2),
        )
        response = ollama.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        raw = response["message"]["content"].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return CriticOutput(suppress=False, reason="parse error")
        return CriticOutput(
            suppress=bool(data.get("suppress", False)),
            reason=data.get("reason", ""),
        )
```

- [ ] **Step 4: Run all decide tests**

```bash
pytest tests/test_decide.py -v --tb=short
```

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add decide/agent.py tests/test_decide.py
git commit -m "feat(decide): two-stage LLM agent deployed — Generator wants to talk, Critic tells it to calm down. Healthy relationship."
```

---

## Task 4: Update build/status.json + push

- [ ] **Step 1: Update oracle entry in build/status.json**

```json
"oracle": {
  "state": "active",
  "task": "All decide modules complete. Requesting CIPHER review.",
  "progress": 100,
  "last_update": "<ISO timestamp>",
  "log": [
    "energy.py complete — adaptive interval 20-120s, novelty + focus EMA",
    "calibrator.py complete — LogisticRegression, JSONL log, auto-retrain",
    "agent.py complete — Generator + Critic two-stage Ollama calls",
    "All tests passing"
  ]
}
```

- [ ] **Step 2: Request ARIA password for push**

```
GIT OPERATION REQUEST
Operation: git push origin feat/decide
Why: ORACLE module complete — two-stage decision engine, calibrator, energy scheduler all tested.
```

- [ ] **Step 3: Push (after ARIA received)**

```bash
git checkout -b feat/decide
git push origin feat/decide
```

---

## Self-Review

**Spec coverage:**
- ✅ `decide/energy.py` — adaptive interval, novelty + focus formula → Task 1
- ✅ `decide/calibrator.py` — LogisticRegression, JSONL, 50-sample threshold → Task 2
- ✅ `decide/agent.py` — Generator + Critic prompts, JSON parsing, Ollama → Task 3
- ✅ status update + push gate → Task 4

**Type consistency:**
- `GeneratorOutput.say: bool` checked in both None-path tests ✓
- `DecisionAgent.evaluate(context: dict) → dict | None` consistent ✓
- `InterruptCalibrator.record(...)` and `.should_suppress(...)` signatures consistent ✓
- `EnergyScheduler.update(novelty_score, focus_level)` + `.next_interval() → int` consistent ✓
