from __future__ import annotations
import re
from dataclasses import dataclass
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
    entities += re.findall(r'\b\w+\.(?:py|ts|js|yaml|json|md|txt)\b', text)
    entities += re.findall(r'https?://\S+', text)
    entities += re.findall(r'\b[A-Z][a-zA-Z0-9]{3,}\b', text)
    return list(dict.fromkeys(entities))[:20]


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
        self._focus_ema: float = 0.5

    def parse(self, window_title: str, ocr_text: str) -> SceneContext:
        app = detect_app(window_title)
        task_type = detect_task_type(app, ocr_text)
        entities = extract_entities(ocr_text)
        app_switch = app != self._last_app
        entity_count = len(entities)
        novelty = min(entity_count / 20.0, 1.0)
        target_focus = 0.2 if app_switch else (1.0 - novelty * 0.5)
        self._focus_ema = 0.7 * self._focus_ema + 0.3 * target_focus
        delta = {"app_switch": app_switch, "entity_count": entity_count}
        self._last_app = app
        return SceneContext(
            app=app, task_type=task_type, entities=entities,
            focus_level=round(self._focus_ema, 3), delta=delta, raw_text=ocr_text,
        )
