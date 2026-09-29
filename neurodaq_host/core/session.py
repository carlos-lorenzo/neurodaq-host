"""Unified Session NPZ: always-raw signals + clocks + events + meta.

Replaces the divergent Schema A (continuous) / Schema B (trial epochs).
Converters for legacy files live here too.
"""
from __future__ import annotations
import json
from datetime import datetime
import numpy as np


def save_session(path: str, signals: np.ndarray, fs: float,
                 channel_labels: list[str], electrode_mapping: list[str],
                 t_host: np.ndarray, t_lsl: np.ndarray,
                 events: list, impedance: np.ndarray | None = None,
                 filter_config: dict | None = None,
                 device_config: dict | None = None,
                 extra: dict | None = None):
    ev_arr = np.array(
        [(e.sample_idx, e.t_lsl, e.t_host, e.kind, e.label, e.value) for e in events],
        dtype=[("sample_idx", "i8"), ("t_lsl", "f8"), ("t_host", "f8"),
               ("kind", "U16"), ("label", "U64"), ("value", "f8")],
    ) if events else np.zeros(0, dtype=[("sample_idx", "i8"), ("t_lsl", "f8"),
                                        ("t_host", "f8"), ("kind", "U16"),
                                        ("label", "U64"), ("value", "f8")])
    np.savez_compressed(
        path,
        signals=np.asarray(signals, dtype=np.float32),
        fs=np.float64(fs),
        channel_labels=np.array(channel_labels),
        electrode_mapping=np.array(electrode_mapping),
        t_host=np.asarray(t_host, dtype=np.float64),
        t_lsl=np.asarray(t_lsl, dtype=np.float64),
        events=ev_arr,
        impedance=np.asarray(impedance if impedance is not None else [], dtype=np.float32),
        filter_config=json.dumps(filter_config or {}),
        device_config=json.dumps(device_config or {}),
        saved_at=datetime.now().isoformat(),
        **(extra or {}),
    )


def default_filename(prefix: str = "session") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"


def convert_schema_a(path: str) -> dict:
    """Legacy continuous files: first array key -> (n,8) signals."""
    z = np.load(path, allow_pickle=True)
    key = [k for k in z.files if "stream" in k or k == "data"][0] if len(z.files) else z.files[0]
    return {"signals": np.asarray(z[key], dtype=np.float32), "source_key": key}


def convert_schema_b(path: str) -> dict:
    """Legacy trial files: X (trials,1,8,T) + y labels -> continuous concat + events."""
    z = np.load(path, allow_pickle=True)
    X, y = np.asarray(z["X"]), np.asarray(z["y"])
    n_trials, _, n_ch, n_t = X.shape
    signals = X[:, 0].transpose(0, 2, 1).reshape(n_trials * n_t, n_ch).astype(np.float32)
    bounds = [{"trial": i, "start": i * n_t, "label": str(y[i])} for i in range(n_trials)]
    return {"signals": signals, "trials": bounds}
