from __future__ import annotations
import json
import math
import os
import pickle
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
    hour_sin = [math.sin(2 * math.pi * time_of_day / 24)]
    hour_cos = [math.cos(2 * math.pi * time_of_day / 24)]
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
            return
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
        return prob_engage < 0.3
