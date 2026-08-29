#!/usr/bin/env python
"""
Real-Time LSL Frequency Spectrum & Band Power Analyzer
Modified to display FFT Power Spectral Density (PSD) and compare Eyes Open vs. Eyes Closed states.
"""

import math
from typing import List, TypedDict, cast

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pylsl
import scipy.signal as signal

# Display & Processing Parameters
plot_duration = 2     # Seconds of rolling history used for FFT calculation
update_interval = 60     # ms between UI scrolls/redraws
pull_interval = 100      # ms between LSL pull operations
LINE_NOISE_FREQ_HZ = 50.0
BAND_LOW = 30
BAND_HIGH = 250.0

# Define EEG Frequency Bands
EEG_BANDS = {
    "Delta (1-4 Hz)": (1.0, 4.0, "#4C72B0"),
    "Theta (4-8 Hz)": (4.0, 8.0, "#55A868"),
    "Alpha (8-13 Hz)": (8.0, 13.0, "#C44E52"),
    "Beta (13-30 Hz)": (13.0, 30.0, "#8172B2"),
    "Gamma (30-45 Hz)": (30.0, 45.0, "#CCB974"),
}


class DataInlet:
    """Handles continuous multi-channel streaming and PSD computation."""

    dtypes = [[], np.float32, np.float64, None,
              np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem):
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=plot_duration,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.fs = info.nominal_srate()

        # Ring buffer setup
        self.max_samples = math.ceil(self.fs * plot_duration)
        bufsize = (self.max_samples, self.channel_count)
        self.input_buffer = np.empty(
            bufsize, dtype=self.dtypes[info.channel_format()])
        self.y_buffer = np.zeros(bufsize, dtype=float)
        self.buffer_size = 0
        self.write_index = 0

        # Plot curves
        self.curves = []
        for ch_ix in range(self.channel_count):
            color = pg.intColor(ch_ix, hues=max(
                self.channel_count, 1), values=1, maxHue=360)
            pen = pg.mkPen(color=color, width=1.5, alpha=180)
            curve = plt.plot(pen=pen, name=f"Ch {ch_ix + 1}")
            self.curves.append(curve)

        # Baseline overlay curve (Eyes Open)
        self.baseline_curve = plt.plot(
            pen=pg.mkPen(color="#00FFFF", width=2.0, style=QtCore.Qt.DashLine),
            name="Baseline (Eyes Open)"
        )

        # Filter filters (50 Hz Notch + Bandpass)
        b_notch, a_notch = signal.iirnotch(LINE_NOISE_FREQ_HZ, 30.0, self.fs)
        self.sos_notch = np.hstack((b_notch, a_notch)).reshape(1, 6)
        self.sos_bp = signal.butter(
            4, [BAND_LOW, BAND_HIGH], btype="bandpass", fs=self.fs, output="sos")
        self.notch_state = None
        self.bp_state = None

        self.baseline_psd = None

    def _append_to_history(self, values: np.ndarray) -> None:
        sample_count = values.shape[0]
        if sample_count <= 0:
            return

        if sample_count >= self.max_samples:
            self.y_buffer = values[-self.max_samples:, :]
            self.buffer_size = self.max_samples
            self.write_index = 0
            return

        end_index = self.write_index + sample_count
        if end_index <= self.max_samples:
            self.y_buffer[self.write_index:end_index, :] = values
        else:
            first_part = self.max_samples - self.write_index
            self.y_buffer[self.write_index:, :] = values[:first_part, :]
            remaining = sample_count - first_part
            self.y_buffer[:remaining, :] = values[first_part:, :]

        self.write_index = (self.write_index + sample_count) % self.max_samples
        self.buffer_size = min(
            self.buffer_size + sample_count, self.max_samples)

    def _ordered_history(self) -> np.ndarray | None:
        if self.buffer_size < self.max_samples // 2:
            return None
        if self.write_index == 0:
            return self.y_buffer[:self.buffer_size, :]
        return np.concatenate((self.y_buffer[self.write_index:, :], self.y_buffer[:self.write_index, :]), axis=0)

    def save_baseline(self):
        """Captures current mean spectrum as Eyes Open baseline."""
        data = self._ordered_history()
        if data is not None:
            freqs, psd = signal.welch(data, fs=self.fs, nperseg=min(
                len(data), int(self.fs)), axis=0)
            self.baseline_psd = (freqs, np.mean(psd, axis=1))
            print("Baseline (Eyes Open) captured!")

    def clear_baseline(self):
        self.baseline_psd = None
        self.baseline_curve.setData([], [])
        print("Baseline cleared.")

    def update_psd(self):
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.input_buffer.shape[0], dest_obj=self.input_buffer)
        if not ts:
            return

        y = self.input_buffer[:len(ts), :]

        # Initialize filter states
        if self.notch_state is None:
            zi_notch = signal.sosfilt_zi(self.sos_notch)
            zi_bp = signal.sosfilt_zi(self.sos_bp)
            self.notch_state = zi_notch[:, :, np.newaxis] * y[0, :]
            self.bp_state = zi_bp[:, :, np.newaxis] * y[0, :]

        # Apply filtering
        out_n, self.notch_state = signal.sosfilt(
            self.sos_notch, y, axis=0, zi=self.notch_state)
        filtered_y, self.bp_state = signal.sosfilt(
            self.sos_bp, out_n, axis=0, zi=self.bp_state)

        self._append_to_history(filtered_y)
        data = self._ordered_history()
        if data is None:
            return

        # Compute Welch Power Spectral Density
        nperseg = min(len(data), int(self.fs))
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=nperseg, axis=0)

        # Restrict display range (1 Hz to 45 Hz)
        valid_idx = (freqs >= BAND_LOW) & (freqs <= BAND_HIGH)
        freqs = freqs[valid_idx]
        psd = psd[valid_idx, :]

        # Plot dynamic channel PSDs
        for ch_ix in range(self.channel_count):
            self.curves[ch_ix].setData(freqs, psd[:, ch_ix])

        # Plot baseline comparison if set
        if self.baseline_psd is not None:
            b_freqs, b_psd = self.baseline_psd
            b_valid = (b_freqs >= BAND_LOW) & (b_freqs <= BAND_HIGH)
            self.baseline_curve.setData(b_freqs[b_valid], b_psd[b_valid])


def main():
    print("Looking for LSL streams...")
    streams = pylsl.resolve_streams()

    pg.setConfigOptions(antialias=True)
    pw = cast(pg.PlotWidget, pg.plot(title="EEG Power Spectral Density (PSD)"))
    pw.setBackground("#101318")
    plt = cast(pg.PlotItem, pw.getPlotItem())
    plt.showGrid(x=True, y=True, alpha=0.25)
    plt.setLabel("bottom", "Frequency", units="Hz")
    plt.setLabel("left", "Power Spectral Density", units="µV²/Hz")
    # Logarithmic Y-axis for standard PSD representation
    plt.setLogMode(x=False, y=True)
    plt.setXRange(BAND_LOW, BAND_HIGH)

    # Highlight EEG Frequency Bands with background regions
    for band_name, (low, high, color) in EEG_BANDS.items():
        region = pg.LinearRegionItem(
            values=[low, high], movable=False, brush=pg.mkBrush(color + "22"))
        plt.addItem(region)

        label = pg.TextItem(text=band_name, color=color, anchor=(0.5, 0))
        label.setPos((low + high) / 2.0, 0)
        plt.addItem(label)

    inlets: List[DataInlet] = []
    for info in streams:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE and info.channel_format() != pylsl.cf_string:
            print(f"Adding stream: {info.name()}")
            inlets.append(DataInlet(info, plt))

    print("\n--- Controls ---")
    print(" Press 'B' : Capture baseline (Eyes Open)")
    print(" Press 'C' : Clear captured baseline")
    print("----------------\n")

    def keyPressEvent(ev):
        key = ev.text().lower()
        if key == 'b':
            for inlet in inlets:
                inlet.save_baseline()
            plt.setTitle(
                "EEG PSD — Baseline Captured (cyan dashed line)", color="#00FFFF")
        elif key == 'c':
            for inlet in inlets:
                inlet.clear_baseline()
            plt.setTitle("EEG Power Spectral Density (PSD)")

    pw.keyPressEvent = keyPressEvent

    def update():
        for inlet in inlets:
            inlet.update_psd()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(pull_interval)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        app = QtGui.QGuiApplication.instance()
        if app is not None:
            app.exec_()


if __name__ == "__main__":
    main()
