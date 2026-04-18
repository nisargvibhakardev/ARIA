from __future__ import annotations
import re
import nltk

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from g2p_en import G2p
from rapidfuzz.distance import Levenshtein

_g2p = G2p()


def _to_phonemes(text: str) -> list[str]:
    raw = _g2p(text)
    return [re.sub(r"[012]", "", p) for p in raw if p.strip() and p not in (" ", "'")]


class DoneWordDetector:
    def __init__(self, done_word: str, tolerance: int = 2) -> None:
        self._target = _to_phonemes(done_word)
        self._tolerance = tolerance
        self._word_count = len(done_word.split())

    def check(self, text: str) -> bool:
        """Return True if text is phonemically close to done_word."""
        phonemes = _to_phonemes(text)
        return Levenshtein.distance(phonemes, self._target) <= self._tolerance

    def strip(self, text: str) -> str:
        """Remove the done_word (and any phoneme-close variant) from text."""
        words = text.split()
        window = self._word_count + 1  # ±1 word for split variants
        for i in range(len(words)):
            for size in range(1, window + 1):
                if i + size > len(words):
                    break
                chunk = " ".join(words[i : i + size])
                if self.check(chunk):
                    remaining = " ".join(words[:i] + words[i + size :])
                    return remaining.strip()
        return text
