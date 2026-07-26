#!/usr/bin/env python
"""
ReceiveAndPlot & Collect example for LSL

This script shows real-time data from LSL, filters it, and presents
visual cues to the user to capture specific 2-second windows of data.
The data is collected and exported to an .npz format ready for PyTorch.
"""

import math
import sys
import random
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pylsl
import scipy.signal as signal
from typing import List, Tuple, TypedDict, cast
from datetime import datetime

# Basic parameters for the plotting window
plot_duration = 1       # how many seconds of data to show in the live plot
update_interval = 50    # ms between screen updates (lowered for responsive UI)
# ms between each pull operation (lowered for responsive UI)
pull_interval = 50
CHANNEL_SPACING_UV = 250.0
TRACE_WIDTH = 1.6
LINE_NOISE_FREQ_HZ = 50.0
BAND_LOW = 30
BAND_HIGH = 300

# --- DATA COLLECTION PARAMETERS ---
ACTIONS = ["Rest", "Right Arm"]
TRIALS_PER_ACTION = 20  # Increased to 10 to give the CNN more examples per class
RECORD_DURATION = 4.0   # Seconds to record each action
REST_DURATION = 2.0     # Seconds to rest between actions
# ----------------------------------


class LineNoiseEstimate(TypedDict):
    channel: int
    tone_rms: float
    overall_rms: float
    relative_db: float


class LineNoiseComparison(TypedDict):
    before: LineNoiseEstimate | None
    after: LineNoiseEstimate | None


class Inlet:
    """Base class to represent a plottable inlet"""

    def __init__(self, info: pylsl.StreamInfo):
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=plot_duration,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem) -> LineNoiseComparison | None:
        pass


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data."""
    dtypes = [[], np.float32, np.float64, None,
              np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem, channel_base_index: int = 0):
        super().__init__(info)
        self.channel_base_index = channel_base_index
        self.max_samples = 2 * math.ceil(info.nominal_srate() * plot_duration)
        bufsize = (self.max_samples, info.channel_count())
        self.input_buffer = np.empty(
            bufsize, dtype=self.dtypes[info.channel_format()])
        self.ts_buffer = np.empty(self.max_samples, dtype=float)
        self.y_buffer = np.empty(
            (self.max_samples, info.channel_count()), dtype=float)
        self.buffer_size = 0
        self.write_index = 0
        empty = np.array([])

        self.curves = [
            pg.PlotCurveItem(
                x=empty, y=empty, autoDownsample=True, clipToView=True,
                pen=pg.mkPen(
                    pg.intColor(ch_ix, hues=max(self.channel_count, 1),
                                values=1, maxHue=360, minValue=180, maxValue=255),
                    width=TRACE_WIDTH,
                ),
            )
            for ch_ix in range(self.channel_count)
        ]
        for curve in self.curves:
            plt.addItem(curve)

        self.fs = info.nominal_srate()

        # Filtering setup
        b_notch, a_notch = signal.iirnotch(LINE_NOISE_FREQ_HZ, 30.0, self.fs)
        self.sos_notch = np.hstack((b_notch, a_notch)).reshape(1, 6)
        self.sos_bp = signal.butter(
            4, [BAND_LOW, BAND_HIGH], btype="bandpass", fs=self.fs, output="sos")

        self.notch_state = None
        self.bp_state = None

        # --- Recording State ---
        self.is_recording = False
        self.record_buffer = []

    def start_recording(self):
        """Prepares the inlet to start saving newly pulled data chunks."""
        self.is_recording = True
        self.record_buffer = []

    def stop_recording(self) -> np.ndarray:
        """Stops recording and returns the aggregated data chunk."""
        self.is_recording = False
        if not self.record_buffer:
            return np.array([])

        # Combine all the small chunks we pulled during the 2 seconds
        recorded_data = np.vstack(self.record_buffer)
        self.record_buffer = []
        return recorded_data

    def _estimate_line_noise(self, timestamps: np.ndarray, values: np.ndarray):
        sample_count = timestamps.size
        if sample_count < 4:
            return None
        phasor = np.exp(-2j * np.pi * LINE_NOISE_FREQ_HZ * timestamps)
        tiny = np.finfo(float).tiny
        channel_reports = []
        for ch_ix in range(self.channel_count):
            channel_values = values[:, ch_ix].astype(float, copy=False)
            demeaned = channel_values - np.mean(channel_values)
            tone_complex = np.sum(demeaned * phasor)
            tone_amplitude = 2.0 * np.abs(tone_complex) / sample_count
            tone_rms = tone_amplitude / np.sqrt(2.0)
            overall_rms = np.sqrt(np.mean(demeaned ** 2))
            relative_db = 20.0 * \
                np.log10(max(tone_rms, tiny) / max(overall_rms, tiny))
            channel_reports.append({
                "channel": ch_ix + 1, "tone_rms": tone_rms,
                "overall_rms": overall_rms, "relative_db": relative_db,
            })
        return max(channel_reports, key=lambda report: report["tone_rms"])

    def _append_to_history(self, timestamps: np.ndarray, values: np.ndarray) -> None:
        sample_count = timestamps.size
        if sample_count <= 0:
            return
        if sample_count >= self.max_samples:
            timestamps = timestamps[-self.max_samples:]
            values = values[-self.max_samples:, :]
            sample_count = timestamps.size
            self.ts_buffer[:sample_count] = timestamps
            self.y_buffer[:sample_count, :] = values
            self.buffer_size = sample_count
            self.write_index = sample_count % self.max_samples
            return
        end_index = self.write_index + sample_count
        if end_index <= self.max_samples:
            self.ts_buffer[self.write_index:end_index] = timestamps
            self.y_buffer[self.write_index:end_index, :] = values
        else:
            first_part = self.max_samples - self.write_index
            self.ts_buffer[self.write_index:] = timestamps[:first_part]
            self.y_buffer[self.write_index:] = values[:first_part, :]
            remaining = sample_count - first_part
            self.ts_buffer[:remaining] = timestamps[first_part:]
            self.y_buffer[:remaining, :] = values[first_part:, :]
        self.write_index = (self.write_index + sample_count) % self.max_samples
        self.buffer_size = min(
            self.buffer_size + sample_count, self.max_samples)

    def _ordered_history(self):
        if self.buffer_size == 0:
            return None, None
        if self.buffer_size < self.max_samples:
            return self.ts_buffer[:self.buffer_size], self.y_buffer[:self.buffer_size, :]
        if self.write_index == 0:
            return self.ts_buffer, self.y_buffer
        ordered_ts = np.concatenate(
            (self.ts_buffer[self.write_index:], self.ts_buffer[: self.write_index]))
        ordered_y = np.concatenate(
            (self.y_buffer[self.write_index:, :], self.y_buffer[: self.write_index, :]), axis=0)
        return ordered_ts, ordered_y

    def pull_and_plot(self, plot_time, plt) -> LineNoiseComparison | None:
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.input_buffer.shape[0], dest_obj=self.input_buffer)
        if not ts:
            return

        ts = np.asarray(ts)
        y = self.input_buffer[: ts.size, :]
        line_noise_before = self._estimate_line_noise(ts, y)

        if self.notch_state is None or self.bp_state is None:
            zi_notch_single = signal.sosfilt_zi(self.sos_notch)
            zi_bp_single = signal.sosfilt_zi(self.sos_bp)
            self.notch_state = zi_notch_single[:, :, np.newaxis] * y[0, :]
            self.bp_state = zi_bp_single[:, :, np.newaxis] * y[0, :]

        out_n, self.notch_state = signal.sosfilt(
            self.sos_notch, y, axis=0, zi=self.notch_state)
        filtered_y, self.bp_state = signal.sosfilt(
            self.sos_bp, out_n, axis=0, zi=self.bp_state)

        # --- Capture Filtered Data ---
        if self.is_recording:
            self.record_buffer.append(filtered_y)

        self._append_to_history(ts, filtered_y)

        history_x, history_y = self._ordered_history()
        if history_x is None or history_y is None:
            return

        plot_offset = history_x.searchsorted(plot_time)
        if plot_offset >= history_x.size:
            return

        visible_x = history_x[plot_offset:]
        channel_offset = CHANNEL_SPACING_UV
        for ch_ix in range(self.channel_count):
            visible_y = history_y[plot_offset:, ch_ix] - \
                (self.channel_base_index + ch_ix) * channel_offset
            self.curves[ch_ix].setData(visible_x, visible_y)

        line_noise_after = self._estimate_line_noise(ts, filtered_y)
        return {"before": line_noise_before, "after": line_noise_after}


class SessionManager:
    """Manages the visual cues, timing, and dataset compilation for the training session."""

    def __init__(self, label: QtWidgets.QLabel, inlets: List[Inlet]):
        self.label = label
        self.inlets = [i for i in inlets if isinstance(i, DataInlet)]

        # Build the shuffled sequence of trials
        self.trial_sequence = ACTIONS * TRIALS_PER_ACTION
        random.shuffle(self.trial_sequence)

        self.current_trial = 0
        self.state = "STARTING"
        self.state_start = pylsl.local_clock()

        # Lists to hold final datasets
        self.all_data = []
        self.all_labels = []

    def update(self):
        now = pylsl.local_clock()
        elapsed = now - self.state_start

        if self.state == "STARTING":
            self.label.setText(
                "Starting in 3 seconds...\nRelax and get ready.")
            self.label.setStyleSheet("color: white; font-size: 32px;")
            if elapsed > 3.0:
                self.transition_to_rest(now)

        elif self.state == "REST":
            remaining = max(0.0, REST_DURATION - elapsed)
            next_action = self.trial_sequence[self.current_trial]
            self.label.setText(
                f"Rest...\nNext up: {next_action} ({remaining:.1f}s)")
            self.label.setStyleSheet(
                "color: yellow; font-size: 36px; font-weight: bold;")

            if elapsed >= REST_DURATION:
                self.transition_to_record(now, next_action)

        elif self.state == "RECORD":
            action = self.trial_sequence[self.current_trial]
            self.label.setText(f"ACTION:\n{action}")
            self.label.setStyleSheet(
                "color: #00FF00; font-size: 48px; font-weight: bold;")

            if elapsed >= RECORD_DURATION:
                self.save_trial_and_transition(now, action)

        elif self.state == "DONE":
            self.label.setText(
                "Session Complete!\nData saved to 'dataset.npz'")
            self.label.setStyleSheet(
                "color: #00FFFF; font-size: 40px; font-weight: bold;")

    def transition_to_rest(self, now):
        self.state = "REST"
        self.state_start = now

    def transition_to_record(self, now, action):
        self.state = "RECORD"
        self.state_start = now
        for inlet in self.inlets:
            inlet.start_recording()

    def save_trial_and_transition(self, now, action):
        trial_data = []
        for inlet in self.inlets:
            data = inlet.stop_recording()
            expected_samples = int(inlet.fs * RECORD_DURATION)

            # Strict PyTorch Constraint: Ensure exactly uniform shape
            if data.shape[0] > expected_samples:
                data = data[:expected_samples, :]  # Truncate slightly
            elif data.shape[0] < expected_samples:
                # Pad with zeros if slightly too short
                pad_size = expected_samples - data.shape[0]
                padding = np.zeros((pad_size, data.shape[1]))
                data = np.vstack([data, padding])

            trial_data.append(data)

        if trial_data:
            # Combine all inlets along the channels axis (horizontal)
            combined_data = np.hstack(trial_data)
            self.all_data.append(combined_data)
            self.all_labels.append(action)

        self.current_trial += 1
        if self.current_trial >= len(self.trial_sequence):
            self.state = "DONE"
            self.state_start = now
            self.save_to_disk()
        else:
            self.transition_to_rest(now)

    def save_to_disk(self):
        if not self.all_data:
            return
        # Shape will be: (Trials, TimeSteps, Channels)
        X = np.stack(self.all_data)
        y = np.array(self.all_labels)
        now = datetime.now()
        np.savez(f"data/dataset_{now.minute}_{now.second}.npz", X=X, y=y)
        print(f"\n--- DATASET SAVED ---")
        print(
            f"Saved {len(y)} trials to 'data/dataset_{now.minute}_{now.second}.npz'")
        print(f"X shape: {X.shape} -> (Trials, TimeSteps, Channels)")
        print(f"y shape: {y.shape} -> (Trials,)")


def main():
    print("looking for streams")
    streams = pylsl.resolve_streams()

    # --- Setup Qt Application and Window Layout ---
    app = QtWidgets.QApplication(sys.argv) if hasattr(
        QtWidgets, 'QApplication') else pg.mkQApp()

    main_window = QtWidgets.QWidget()
    main_window.setWindowTitle("LSL Data Collection")
    main_window.resize(1000, 800)
    main_window.setStyleSheet("background-color: #101318;")

    layout = QtWidgets.QVBoxLayout()
    main_window.setLayout(layout)

    # 1. The Large Instruction Label
    instruction_label = QtWidgets.QLabel("Connecting to streams...")
    instruction_label.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(instruction_label)

    # 2. The PyQtGraph PlotWidget
    pg.setConfigOptions(antialias=False)
    pw = pg.PlotWidget()
    pw.setBackground("#101318")
    plt = pw.getPlotItem()
    plt.setMenuEnabled(False)
    plt.showGrid(x=True, y=True, alpha=0.18)
    plt.setLabel("bottom", "Time", units="s")
    plt.setLabel("left", "Stacked channels", units="µV")
    plt.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
    layout.addWidget(pw, stretch=1)

    main_window.show()
    # ----------------------------------------------

    inlets: List[Inlet] = []
    scale_locked = False
    next_channel_base = 0

    for info in streams:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE and info.channel_format() != pylsl.cf_string:
            print("Adding data inlet: " + info.name())
            inlets.append(
                DataInlet(info, plt, channel_base_index=next_channel_base))
            next_channel_base += info.channel_count()

    # Initialize our Session Manager to oversee the UI and Recording
    session_manager = SessionManager(instruction_label, inlets)

    def scroll():
        fudge_factor = pull_interval * 0.002
        plot_time = pylsl.local_clock()
        pw.setXRange(plot_time - plot_duration +
                     fudge_factor, plot_time - fudge_factor)

    def update():
        mintime = pylsl.local_clock() - plot_duration

        # 1. Pull data for the graph
        for inlet in inlets:
            inlet.pull_and_plot(mintime, plt)

        # 2. Update the session manager (UI cues and recording states)
        session_manager.update()

    def keyPressEvent(ev):
        nonlocal scale_locked
        if ev.text().lower() == 'l':
            scale_locked = not scale_locked
            plt.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=not scale_locked)

    main_window.keyPressEvent = keyPressEvent

    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        app.exec_()


if __name__ == "__main__":
    main()
