"""AC lead-off / impedance measurement (pure functions, testable)."""
import numpy as np
from neurodaq_host.core.config import SERIES_R_OHMS


def tone_peak_uV(data: np.ndarray, freq_hz: float, fs: float) -> float:
    n = np.arange(len(data))
    w = np.hanning(len(data))
    ang = 2.0 * np.pi * freq_hz * n / fs
    xc = float(np.sum(data * w * np.cos(ang)))
    xs = float(np.sum(data * w * np.sin(ang)))
    return 2.0 * np.hypot(xc, xs) / float(np.sum(w))


def loop_impedance_kohm(data: np.ndarray, fs: float, freq_hz: float,
                        current_a: float, cal_factor: float = 1.0) -> dict:
    res = {"z_kohm": 0.0, "tone_uVpk": 0.0, "snr": 0.0, "flags": []}
    if data.size < 32 or freq_hz <= 0 or current_a <= 0:
        return res
    v = tone_peak_uV(data, freq_hz, fs)
    floor = float(np.median([tone_peak_uV(data, freq_hz + 3.0, fs),
                             tone_peak_uV(data, freq_hz - 3.0, fs)]))
    res["tone_uVpk"] = v
    res["snr"] = v / floor if floor > 0 else float("inf")
    if res["snr"] < 4.0:
        res["flags"].append("LOW_SNR")
        return res
    if res["snr"] < 8.0:
        res["flags"].append("NOISY")
    z = max(0.0, (max(0.0, v - floor) * 1e-6) * (np.pi / 4.0) / current_a
            - 2.0 * SERIES_R_OHMS)
    res["z_kohm"] = z * cal_factor / 1000.0
    return res
