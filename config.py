# config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ScreenConfig:
    interval_seconds: float = 3.0
    diff_threshold: float = 0.15
    monitor_index: int = 1


@dataclass
class MicConfig:
    vad_aggressiveness: int = 2


@dataclass
class MemoryConfig:
    rolling_days: int = 7
    context_window_seconds: int = 120
    ebbinghaus_purge_threshold: float = 0.2
    initial_stability_days: float = 1.0
    important_stability_days: float = 30.0


@dataclass
class EpisodicConfig:
    summary_interval_minutes: int = 30


@dataclass
class LLMConfig:
    model: str = "llama3.1:8b"
    keep_alive: str = "3m"
    response_language: str = "english"


@dataclass
class EnergyConfig:
    min_interval_seconds: int = 20
    max_interval_seconds: int = 120


@dataclass
class CalibratorConfig:
    min_samples: int = 50
    retrain_every: int = 10


@dataclass
class IdleUnloadConfig:
    ocr_minutes: float = 3.0
    whisper_minutes: float = 3.0
    tts_minutes: float = 2.0


@dataclass
class WhisperConfig:
    device: str = "cpu"
    compute_type: str = "int8"


@dataclass
class OverlayConfig:
    position: str = "bottom-right"
    auto_dismiss_seconds: int = 8


@dataclass
class SystemConfig:
    nice_level: int = 10


@dataclass
class Config:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    mic: MicConfig = field(default_factory=MicConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    calibrator: CalibratorConfig = field(default_factory=CalibratorConfig)
    idle_unload: IdleUnloadConfig = field(default_factory=IdleUnloadConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    hotkey: str = "ctrl+shift+space"
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> Config:
        if not Path(path).exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        _section_types = {
            "screen": ScreenConfig, "mic": MicConfig, "memory": MemoryConfig,
            "episodic": EpisodicConfig, "llm": LLMConfig, "energy": EnergyConfig,
            "calibrator": CalibratorConfig, "idle_unload": IdleUnloadConfig,
            "whisper": WhisperConfig, "overlay": OverlayConfig, "system": SystemConfig,
        }
        for key, value in data.items():
            if key in _section_types and isinstance(value, dict):
                section = getattr(cfg, key)
                for k, v in value.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
