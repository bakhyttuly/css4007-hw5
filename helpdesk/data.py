"""Loading of the supplied fictional fixtures from data/."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_helpdesk() -> dict:
    return _load("helpdesk.json")


def load_status_profiles() -> dict:
    return _load("service_status.json")["profiles"]


def load_scenarios() -> list[dict]:
    return _load("scenarios.json")["scenarios"]


def get_scenario(scenario_id: str) -> dict:
    for sc in load_scenarios():
        if sc["id"].lower() == scenario_id.lower():
            return sc
    raise KeyError(f"Unknown scenario: {scenario_id}")
