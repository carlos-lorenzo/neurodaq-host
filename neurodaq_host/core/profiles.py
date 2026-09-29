"""UI profile: which tabs are enabled + order + shared view settings."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field, asdict

DEFAULT_PROFILE = {
    "tabs": [
        {"id": "scope", "enabled": True, "order": 0},
        {"id": "spectrum", "enabled": True, "order": 1},
        {"id": "brain", "enabled": True, "order": 2},
        {"id": "device", "enabled": True, "order": 3},
        {"id": "impedance", "enabled": True, "order": 4},
        {"id": "events", "enabled": True, "order": 5},
        {"id": "paradigm", "enabled": False, "order": 6},
        {"id": "emg", "enabled": False, "order": 7},
        {"id": "p300", "enabled": False, "order": 8},
        {"id": "p300_speller", "enabled": False, "order": 9},
    ],
    "floating": [],   # retired (pop-out removed); ignored on load, kept for compat
    "panels": {},     # tab id -> panel_state dict (toggles, splits)
    "plot_seconds": 3.0,
    "channel_offset": 100.0,
}

PROFILE_PATH = os.path.expanduser("~/.config/neurodaq/profile.json")


def load_profile(path: str = PROFILE_PATH) -> dict:
    base = {k: (list(v) if isinstance(v, list) else
                dict(v) if isinstance(v, dict) else v)
            for k, v in DEFAULT_PROFILE.items()}
    try:
        with open(path) as f:
            saved = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return base
    # union tab lists by id so new tabs appear even with an old saved profile
    if "tabs" in saved:
        have = {t["id"] for t in saved["tabs"]}
        merged = list(saved["tabs"])
        for t in base["tabs"]:
            if t["id"] not in have:
                merged.append(dict(t))
        saved["tabs"] = merged
    base.update(saved)
    base.setdefault("floating", [])
    base.setdefault("panels", {})
    return base


def save_profile(profile: dict, path: str = PROFILE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
