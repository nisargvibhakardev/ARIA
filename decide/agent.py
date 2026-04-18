from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from config import LLMConfig

if TYPE_CHECKING:
    pass


_SINGLE_PROMPT = """ARIA assistant. Screen context: {context}

Reply JSON only: {{"say":true/false,"message":"max 20 words","importance":0.0-1.0,"reason":"5 words"}}"""


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
        try:
            return self._call_single(context)
        except Exception as e:
            msg = str(e).lower()
            if "timed out" in msg or "timeout" in msg:
                return {"message": "Ollama timed out — model too slow for this hardware.",
                        "type": "focus_drift", "importance": 0.3,
                        "reason": "LLM timeout", "extract": {}}
            raise

    def _call_single(self, context: dict) -> dict | None:
        import ollama
        client = ollama.Client(timeout=60)
        recent = context.get("recent", [])
        scene = context.get("scene")
        ctx_text = " | ".join(c.get("text", "")[:60] for c in recent[-3:]) or "idle"
        if scene:
            ctx_text = f"[{scene.app}] {ctx_text}"

        prompt = _SINGLE_PROMPT.format(context=ctx_text[:200])
        response = client.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 80},
            keep_alive="10m",
        )
        raw = response["message"]["content"].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # try extracting JSON from response
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None
        if not data.get("say", False):
            return None
        return {
            "message": data.get("message", ""),
            "type": "general",
            "importance": float(data.get("importance", 0.5)),
            "reason": data.get("reason", ""),
            "extract": {},
        }
