"""DSP: display/record filtering, FFT helpers, band power."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
from neurodaq_host.core.config import FilterConfig


def apply_filters(data: np.ndarray, fs: float, f: FilterConfig) -> np.ndarray:
    if data.size == 0 or data.shape[0] < 15 or fs <= 0:
        return data
    nyq = fs / 2.0
    out = data.copy()
    if f.dc_remove:
        out = out - np.mean(out, axis=0)
    if f.notch_enabled and 0 < f.notch_freq < nyq:
        try:
            b, a = iirnotch(f.notch_freq, f.notch_q, fs=fs)
            out = filtfilt(b, a, out, axis=0)
        except Exception:
            pass
    if f.hp_enabled and 0 < f.hp_freq < nyq:
        try:
            sos = butter(f.hp_order, f.hp_freq, btype="highpass", fs=fs, output="sos")
            out = sosfiltfilt(sos, out, axis=0)
        except Exception:
            pass
    if f.lp_enabled and 0 < f.lp_freq < nyq:
        try:
            sos = butter(f.lp_order, f.lp_freq, btype="lowpass", fs=fs, output="sos")
            out = sosfiltfilt(sos, out, axis=0)
        except Exception:
            pass
    return out


def band_power(col: np.ndarray, fs: float, lo: float, hi: float) -> float:
    n = len(col)
    if n < 16 or fs <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd = (np.abs(np.fft.rfft(col)) / n) ** 2
    return float(np.sum(psd[(freqs >= lo) & (freqs <= hi)]))


EEG_BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Alpha": (8.0, 13.0),
             "Beta": (13.0, 30.0), "Gamma": (30.0, 50.0)}
