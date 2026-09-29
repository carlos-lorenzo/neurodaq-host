"""Session + impedance + filter smoke tests."""
import numpy as np
from neurodaq_host.core import session as S
from neurodaq_host.core.config import FilterConfig
from neurodaq_host.core.events import Event
from neurodaq_host.dsp.filters import apply_filters, band_power
from neurodaq_host.dsp.impedance import loop_impedance_kohm


def test_session_roundtrip(tmp_path):
    n = 500
    sig = (np.random.randn(n, 8) * 20).astype(np.float32)
    th = np.linspace(1000.0, 1002.0, n)
    tl = np.linspace(50.0, 52.0, n)
    ev = [Event(10, 50.04, 1000.04, "manual", "cue/Left", 0.0)]
    fn = str(tmp_path / "s.npz")
    S.save_session(fn, sig, 250.0, [f"Ch{i}" for i in range(8)],
                   ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
                   th, tl, ev, filter_config={"hp_freq": 1.0}, device_config={})
    z = np.load(fn, allow_pickle=True)
    assert z["signals"].shape == (n, 8)
    assert float(z["fs"]) == 250.0
    assert (z["t_host"] != 0).any() and (z["t_lsl"] != 0).any()
    assert len(z["events"]) == 1 and z["events"]["label"][0] == "cue/Left"


def test_filters_passthrough_short():
    d = np.random.randn(10, 8)
    out = apply_filters(d, 250.0, FilterConfig())
    assert out.shape == d.shape  # too short -> untouched, no crash


def test_band_power_alpha_peak():
    fs = 250.0
    t = np.arange(4 * int(fs)) / fs
    col = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz alpha
    a = band_power(col, fs, 8.0, 13.0)
    b = band_power(col, fs, 30.0, 50.0)
    assert a > 10 * b


def test_impedance_clean_tone():
    fs, f0, cur = 250.0, 31.2, 24e-6
    # loop Z of 2x2.2k series + 10k DUT = 14.4k; invert the forward model
    z_loop = 14400.0
    v_sig = z_loop * cur * 4.0 / np.pi  # 0-pk fundamental at ADC
    t = np.arange(2 * int(fs)) / fs
    data = (v_sig * np.sin(2 * np.pi * f0 * t) * 1e6)  # uV
    m = loop_impedance_kohm(data, fs, f0, cur)
    assert "LOW_SNR" not in m["flags"], m
    assert abs(m["z_kohm"] - 10.0) < 3.0, m  # series R removed, ~10k DUT


def test_impedance_no_tone_low_snr():
    rng = np.random.default_rng(0)
    t = np.arange(1024) / 250.0
    data = 50.0 * np.sin(2 * np.pi * 10.0 * t) + rng.standard_normal(1024) * 2.0
    m = loop_impedance_kohm(data, 250.0, 31.2, 24e-6)
    assert "LOW_SNR" in m["flags"] and m["z_kohm"] == 0.0
