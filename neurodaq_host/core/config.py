"""Central dataclasses: hardware + filter + session config (single source of truth)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

N_CHANNELS = 8
N_SAMPLES_PER_PACKET = 25
VREF = 4.5
SERIES_R_OHMS = 2200.0

SAMPLE_RATE_MAP = {0: 16000.0, 1: 8000.0, 2: 4000.0, 3: 2000.0,
                   4: 1000.0, 5: 500.0, 6: 250.0}
GAIN_MAP = {0: 1.0, 1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 12.0, 6: 24.0}
LEADOFF_CURRENT_MAP = {0: 6e-9, 1: 24e-9, 2: 6e-6, 3: 24e-6}
LEADOFF_FREQ_MAP = {0: 0.0, 1: 7.8, 2: 31.2, 3: None}  # 3 = fDR/4, resolved vs fs
LEADOFF_THRESHOLD_PCT = {0: 95.0, 1: 92.5, 2: 90.0, 3: 87.5,
                         4: 85.0, 5: 80.0, 6: 75.0, 7: 70.0}


@dataclass
class FilterConfig:
    dc_remove: bool = True
    hp_enabled: bool = True
    hp_freq: float = 1.0
    hp_order: int = 4
    lp_enabled: bool = True
    lp_freq: float = 40.0
    lp_order: int = 4
    notch_enabled: bool = True
    notch_freq: float = 50.0
    notch_q: float = 30.0
    rev: int = 0  # bumped on every edit; lets caches invalidate cheaply

    def bump(self):
        self.rev += 1


@dataclass
class ChannelConfig:
    gain: float = 24.0
    gain_code: int = 6
    power_down: int = 0
    mux: int = 0
    visible: bool = True
    electrode: str = ""


@dataclass
class DeviceConfig:
    sample_rate: float = 250.0
    rate_code: int = 6
    channels: list = field(default_factory=lambda: [ChannelConfig() for _ in range(N_CHANNELS)])
    bias_p: bool = True
    bias_n: bool = True
    sensp: int = 0xFF
    sensn: int = 0xFF
    # lead-off engine
    loff_enabled: bool = False
    loff_freq_code: int = 2
    loff_curr_code: int = 3
    loff_thresh_code: int = 4

    def to_json(self) -> str:
        d = asdict(self)
        d["channels"] = [asdict(c) for c in self.channels]
        return json.dumps(d)

    @staticmethod
    def from_json(s: str) -> "DeviceConfig":
        d = json.loads(s)
        chs = [ChannelConfig(**c) for c in d.pop("channels", [])]
        cfg = DeviceConfig(**d)
        if chs:
            cfg.channels = chs
        return cfg
