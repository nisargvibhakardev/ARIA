from __future__ import annotations
import hashlib
import numpy as np
import pytesseract
from PIL import Image
from process.loader import IdleTimer
from config import IdleUnloadConfig


class OCREngine:
    def __init__(self, config: IdleUnloadConfig) -> None:
        self._config = config
        self._last_hash: str = ""
        self._last_text: str = ""
        self._timer = IdleTimer(config.ocr_minutes * 60)
        self._timer.reset()

    def extract(self, frame: np.ndarray) -> str:
        h = hashlib.sha256(frame.tobytes()).hexdigest()
        if h == self._last_hash:
            return self._last_text
        img = Image.fromarray(frame)
        text = pytesseract.image_to_string(img)
        self._last_hash = h
        self._last_text = text
        self._timer.reset()
        return text
