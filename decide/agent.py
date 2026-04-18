from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from config import LLMConfig

if TYPE_CHECKING:
    pass


_SINGLE_PROMPT = """You are ARIA, an assistant. Respond in English only.
{context_label}: {context}

Reply with JSON only, no other text: {{"say":true/false,"message":"max 20 words in English","importance":0.0-1.0,"reason":"5 words"}}"""



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
        source = context.get("source", "screen")
        ctx_text = " | ".join(c.get("text", "")[:60] for c in recent[-3:]) or "idle"
        if scene:
            ctx_text = f"[{scene.app}] {ctx_text}"
        context_label = "User said" if source == "speech" else "Screen context"

        prompt = _SINGLE_PROMPT.format(context_label=context_label, context=ctx_text[:200])
        kwargs: dict = dict(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": self._config.num_predict},
            keep_alive=self._config.keep_alive,
        )
        if self._config.format_json:
            kwargs["format"] = "json"
        response = client.chat(**kwargs)
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
