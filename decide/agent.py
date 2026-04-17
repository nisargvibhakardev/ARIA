from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from config import LLMConfig

if TYPE_CHECKING:
    pass


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
        import ollama
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
        import ollama
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
