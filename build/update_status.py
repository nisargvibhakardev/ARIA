# build/update_status.py
"""Call this from any squad agent to update build/status.json."""
from __future__ import annotations
import json
import time
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "status.json"


def update(
    agent_id: str,
    state: str,
    task: str,
    progress: int,
    log_entry: str | None = None,
) -> None:
    data = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            data = json.load(f)

    if agent_id not in data:
        data[agent_id] = {"state": "pending", "task": "", "progress": 0,
                           "last_update": "", "log": []}

    data[agent_id]["state"] = state
    data[agent_id]["task"] = task
    data[agent_id]["progress"] = progress
    data[agent_id]["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if log_entry:
        data[agent_id]["log"].append(log_entry)
        data[agent_id]["log"] = data[agent_id]["log"][-20:]

    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)
