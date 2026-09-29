"""Device profiles: swap EEG<->EMG by config, not code forks."""
from dataclasses import dataclass

DEFAULT_EEG_ELECTRODES = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]

ELECTRODE_1020_POS = {
    "Fp1": (-0.35, 0.75, "Frontal"), "Fp2": (0.35, 0.75, "Frontal"),
    "F7": (-0.75, 0.45, "Frontal"), "F3": (-0.35, 0.40, "Frontal"),
    "Fz": (0.00, 0.40, "Frontal"), "F4": (0.35, 0.40, "Frontal"),
    "F8": (0.75, 0.45, "Frontal"), "T3": (-0.85, 0.00, "Temporal"),
    "C3": (-0.40, 0.00, "Central"), "Cz": (0.00, 0.00, "Central"),
    "C4": (0.40, 0.00, "Central"), "T4": (0.85, 0.00, "Temporal"),
    "T5": (-0.75, -0.45, "Temporal"), "P3": (-0.35, -0.40, "Parietal"),
    "Pz": (0.00, -0.40, "Parietal"), "P4": (0.35, -0.40, "Parietal"),
    "T6": (0.75, -0.45, "Temporal"), "O1": (-0.35, -0.75, "Occipital"),
    "O2": (0.35, -0.75, "Occipital"),
}


@dataclass
class DeviceProfile:
    name: str
    n_channels: int
    units: str
    default_electrodes: list
    supports_impedance: bool = True
    supports_brain_map: bool = True
    default_tabs: tuple = ("scope", "spectrum", "brain", "device", "impedance", "events")


EEG_ADS1299 = DeviceProfile("ADS1299 EEG", 8, "µV", list(DEFAULT_EEG_ELECTRODES))
EMG_PLACEHOLDER = DeviceProfile("EMG (future)", 8, "mV", [f"M{i+1}" for i in range(8)],
                                supports_impedance=False, supports_brain_map=False,
                                default_tabs=("scope", "spectrum", "device", "events"))
