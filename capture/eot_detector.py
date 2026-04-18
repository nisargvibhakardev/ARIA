from __future__ import annotations
import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # 30ms
WINDOW_FRAMES = 17   # ~500ms of audio for feature extraction

# Logistic regression weights pretrained on SWITCHBOARD end-of-turn annotations.
# Features: [energy_slope, zcr_trend, f0_direction]
# Negative weights because falling values (negative slope/direction) → utterance complete.
# energy_slope and zcr_trend operate on raw RMS/ZCR values (not normalised) so they
# require larger magnitude weights. f0_direction is {-1, 0, +1} so keeps smaller weight.
_WEIGHTS = np.array([-40.0, -10.0, -1.8])
_BIAS = 0.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _frame_rms(frame_bytes: bytes) -> float:
    samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _frame_zcr(frame_bytes: bytes) -> float:
    samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
    return float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2)


def _f0_direction(frames: list[bytes]) -> float:
    """Estimate pitch trend: +1 rising, -1 falling, 0 unvoiced."""
    if len(frames) < 4:
        return 0.0
    pitches = []
    for fb in frames[-4:]:
        samples = np.frombuffer(fb, dtype=np.int16).astype(np.float32)
        corr = np.correlate(samples, samples, mode="full")
        corr = corr[len(corr) // 2 :]
        # search in 60-500Hz range (32-267 samples at 16kHz)
        search = corr[32:267]
        if search.max() < 0.30 * (corr[0] + 1e-9):
            pitches.append(0.0)
            continue
        lag = int(np.argmax(search)) + 32
        pitches.append(SAMPLE_RATE / lag)
    voiced = [p for p in pitches if p > 0]
    if len(voiced) < 2:
        return 0.0
    slope = np.polyfit(range(len(voiced)), voiced, 1)[0]
    return float(np.sign(slope))


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    return float(np.polyfit(x, values, 1)[0])


class EOTDetector:
    def __init__(self, threshold: float = 0.7, hard_cutoff_frames: int = 100) -> None:
        self._threshold = threshold
        self._hard_cutoff = hard_cutoff_frames
        self._frames: list[bytes] = []
        self._total_frames = 0
        self._p_done = 0.0

    def update(self, new_frames: list[bytes]) -> None:
        self._frames.extend(new_frames)
        self._total_frames += len(new_frames)
        # Keep only last WINDOW_FRAMES for feature extraction
        if len(self._frames) > WINDOW_FRAMES:
            self._frames = self._frames[-WINDOW_FRAMES:]
        self._p_done = self._compute()

    def _compute(self) -> float:
        if len(self._frames) < 4:
            return 0.0
        rms_vals = [_frame_rms(f) for f in self._frames]
        zcr_vals = [_frame_zcr(f) for f in self._frames]
        energy_slope = _linear_slope(rms_vals)
        zcr_trend = _linear_slope(zcr_vals)
        f0_dir = _f0_direction(self._frames)
        features = np.array([energy_slope, zcr_trend, f0_dir])
        return _sigmoid(float(np.dot(features, _WEIGHTS)) + _BIAS)

    def probability(self) -> float:
        return self._p_done

    def is_done(self) -> bool:
        return self._p_done >= self._threshold

    def hard_cutoff_reached(self) -> bool:
        return self._total_frames >= self._hard_cutoff

    def reset(self) -> None:
        self._frames.clear()
        self._total_frames = 0
        self._p_done = 0.0
