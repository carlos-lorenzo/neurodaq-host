#!/usr/bin/env python

import math
import sys
import os
import random
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pylsl
import scipy.signal as signal
from typing import List, TypedDict
from datetime import datetime

# --- SIGNAL PROCESSING CONSTANTS ---
LINE_NOISE_FREQ_HZ = 50.0
BAND_LOW = 8.0  # Optimized for Mu band (motor imagery)
BAND_HIGH = 30.0  # Optimized for Beta band (motor imagery)

# --- DATA COLLECTION & PARADIGM TIMINGS ---
# Based on motor_imager_conditioning.pdf
ACTIONS = ["Rest", "Left Side", "Right Side"]
TRIALS_PER_ACTION = 2
PREP_DURATION = 2.0		# Crosshair to increase attention
CUE_DURATION = 1.0		 # Text indicating the task
RECORD_DURATION = 4.0	  # Dynamic GIF execution and data recording
# Rest between trials (Paper suggests up to 9s, adjusted for pace)
RELAX_DURATION = 6.0

# --- INTERFACE CONSTANTS ---
PULL_INTERVAL_MS = 50


class LineNoiseEstimate(TypedDict):
    channel: int
    tone_rms: float
    overall_rms: float
    relative_db: float


class DataInlet:
    """Handles connecting to an LSL stream, pulling data, filtering, and estimating noise."""
    dtypes = [[], np.float32, np.float64, None,
              np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, channel_base_index: int = 0):
        # Setup LSL Inlet
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=int(math.ceil(RECORD_DURATION)),
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.channel_base_index = channel_base_index
        self.fs = info.nominal_srate()

        # Buffers
        self.max_samples = 2 * math.ceil(self.fs * RECORD_DURATION)
        bufsize = (self.max_samples, self.channel_count)
        self.input_buffer = np.empty(
            bufsize, dtype=self.dtypes[info.channel_format()])

        # Filtering setup (Notch for line noise, Bandpass for ERD/ERS 8-30Hz)
        b_notch, a_notch = signal.iirnotch(LINE_NOISE_FREQ_HZ, 30.0, self.fs)
        self.sos_notch = np.hstack((b_notch, a_notch)).reshape(1, 6)
        self.sos_bp = signal.butter(
            4, [BAND_LOW, BAND_HIGH], btype="bandpass", fs=self.fs, output="sos")

        self.notch_state = None
        self.bp_state = None

        # Recording State
        self.is_recording = False
        self.record_buffer = []
        self.latest_noise_report: List[LineNoiseEstimate] = []

    def _estimate_line_noise(self, timestamps: np.ndarray, values: np.ndarray) -> List[LineNoiseEstimate]:
        sample_count = timestamps.size
        if sample_count < 4:
            return []

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
                "channel": self.channel_base_index + ch_ix + 1,
                "tone_rms": tone_rms,
                "overall_rms": overall_rms,
                "relative_db": relative_db,
            })
        return channel_reports

    def pull_and_process(self):
        """Pulls new data from LSL, applies filters, and records if active."""
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.input_buffer.shape[0], dest_obj=self.input_buffer)

        if not ts:
            return

        ts = np.asarray(ts)
        y = self.input_buffer[: ts.size, :]

        # Initialize filter states on first pass
        if self.notch_state is None or self.bp_state is None:
            zi_notch_single = signal.sosfilt_zi(self.sos_notch)
            zi_bp_single = signal.sosfilt_zi(self.sos_bp)
            self.notch_state = zi_notch_single[:, :, np.newaxis] * y[0, :]
            self.bp_state = zi_bp_single[:, :, np.newaxis] * y[0, :]

        # Apply filters
        out_n, self.notch_state = signal.sosfilt(
            self.sos_notch, y, axis=0, zi=self.notch_state)
        filtered_y, self.bp_state = signal.sosfilt(
            self.sos_bp, out_n, axis=0, zi=self.bp_state)

        # Calculate noise metrics on the filtered data for the debug menu
        self.latest_noise_report = self._estimate_line_noise(ts, filtered_y)

        # Append to record buffer if recording is active
        if self.is_recording:
            self.record_buffer.append(y)

    def start_recording(self):
        self.is_recording = True
        self.record_buffer = []

    def stop_recording(self) -> np.ndarray:
        self.is_recording = False
        if not self.record_buffer:
            return np.array([])
        recorded_data = np.vstack(self.record_buffer)
        self.record_buffer = []
        return recorded_data


class SessionManager:
    """Manages the visual cues, state machine, and data formatting based on the research paper."""

    def __init__(self, cue_label: QtWidgets.QLabel, debug_table: QtWidgets.QTableWidget, status_label: QtWidgets.QLabel, inlets: List[DataInlet]):
        self.cue_label = cue_label
        self.debug_table = debug_table
        self.status_label = status_label
        self.inlets = inlets

        # Build and shuffle trials
        self.trial_sequence = ACTIONS * TRIALS_PER_ACTION
        random.shuffle(self.trial_sequence)
        self.total_trials = len(self.trial_sequence)
        self.current_trial = 0

        # State Machine
        self.state = "STARTING"
        self.state_start = pylsl.local_clock()
        self.current_movie = None

        # Dataset storage
        self.all_data = []
        self.all_labels = []

    def set_visual_cue(self, text: str, image_names: List[str], color: str = "white"):
        """Attempts to load a GIF or Image. Falls back to styled text if missing."""
        if self.current_movie:
            self.current_movie.stop()
            self.current_movie = None

        file_found = False
        for img_name in image_names:
            img_name = f"images/{img_name}"
            if os.path.exists(img_name):
                if img_name.lower().endswith(".gif"):
                    self.current_movie = QtGui.QMovie(img_name)
                    self.cue_label.setMovie(self.current_movie)
                    self.current_movie.start()
                else:
                    pixmap = QtGui.QPixmap(img_name)
                    # Scale to fit while maintaining aspect ratio
                    self.cue_label.setPixmap(pixmap.scaled(self.cue_label.size(
                    ), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                file_found = True
                break

        if not file_found:
            # Fallback to pure text
            self.cue_label.setText(text)
            self.cue_label.setStyleSheet(
                f"color: {color}; font-size: 64px; font-weight: bold; background-color: #101318;")

    def update(self):
        now = pylsl.local_clock()
        elapsed = now - self.state_start

        # Update Debug Menu metrics
        self.update_debug_menu()

        if self.state == "STARTING":
            self.set_visual_cue(
                "Starting shortly...\nRelax and prepare.", [], "#FFFFFF")
            if elapsed > 3.0:
                self.transition_to_prep(now)

        elif self.state == "PREPARATION":
            # Paper: Display crosshair to increase attention
            if elapsed >= PREP_DURATION:
                self.transition_to_task_cue(now)

        elif self.state == "TASK_CUE":
            # Paper: Display text indicating the upcoming task
            if elapsed >= CUE_DURATION:
                self.transition_to_record(now)

        elif self.state == "EXECUTION":
            # Paper: Display dynamic GIF and perform the motor imagery
            if elapsed >= RECORD_DURATION:
                self.save_trial_and_transition(now)

        elif self.state == "RELAX":
            # Paper: Display eye icon to cue relaxation
            if elapsed >= RELAX_DURATION:
                self.transition_to_prep(now)

        elif self.state == "DONE":
            self.set_visual_cue(
                "Session Complete!\nData saved.", [], "#00FFFF")

    def transition_to_prep(self, now):
        self.state = "PREPARATION"
        self.state_start = now
        self.set_visual_cue("+", ["crosshair.png", "crosshair.jpg"], "#FFFFFF")

    def transition_to_task_cue(self, now):
        self.state = "TASK_CUE"
        self.state_start = now
        action = self.trial_sequence[self.current_trial]
        self.set_visual_cue(f"Task:\n{action}", [], "#FFFF00")

    def transition_to_record(self, now):
        self.state = "EXECUTION"
        self.state_start = now
        action = self.trial_sequence[self.current_trial]

        # Define expected filenames for dynamic cues based on the action
        safe_action_name = action.lower().replace(" ", "_")
        image_targets = [f"{safe_action_name}.gif",
                         f"{safe_action_name}.png", f"{safe_action_name}.jpg"]

        self.set_visual_cue(f"Executing:\n{action}", image_targets, "#00FF00")

        # Start recording on all streams
        for inlet in self.inlets:
            inlet.start_recording()

    def transition_to_relax(self, now):
        self.state = "RELAX"
        self.state_start = now
        self.set_visual_cue(
            "Relax", ["relax.png", "relax.jpg", "eye_icon.png"], "#AAAAAA")

    def save_trial_and_transition(self, now):
        action = self.trial_sequence[self.current_trial]
        trial_data = []

        for inlet in self.inlets:
            data = inlet.stop_recording()
            expected_samples = int(inlet.fs * RECORD_DURATION)

            # Ensure exact sample length
            if data.shape[0] > expected_samples:
                data = data[:expected_samples, :]
            elif data.shape[0] < expected_samples:
                pad_size = expected_samples - data.shape[0]
                padding = np.zeros((pad_size, data.shape[1]))
                data = np.vstack([data, padding])

            # Shape manipulation to match requested ML format: (1, channels, time)
            # Currently data is (Time, Channels)
            data = data.T  # Now it is (Channels, Time)
            # Now it is (1, Channels, Time)
            data = np.expand_dims(data, axis=0)

            trial_data.append(data)

        if trial_data:
            # If multiple inlets, stack them along the channels axis (axis=1)
            combined_data = np.concatenate(trial_data, axis=1)
            self.all_data.append(combined_data)
            self.all_labels.append(action)

        self.current_trial += 1

        if self.current_trial >= self.total_trials:
            self.state = "DONE"
            self.state_start = now
            self.save_to_disk()
        else:
            self.transition_to_relax(now)

    def update_debug_menu(self):
        """Updates the table and labels in the right-side debug panel."""
        if self.state == "DONE":
            self.status_label.setText("Status: COMPLETE")
            return

        # Update Status Label
        trials_left = self.total_trials - self.current_trial
        samples_rem = trials_left * \
            int(RECORD_DURATION * pylsl.local_clock())  # Rough proxy
        self.status_label.setText(
            f"Trials Remaining: {trials_left} / {self.total_trials}\nCurrent State: {self.state}")

        # Update Debug Table
        row = 0
        for inlet in self.inlets:
            for report in inlet.latest_noise_report:
                if row >= self.debug_table.rowCount():
                    self.debug_table.insertRow(row)

                # Ch, Overall RMS, Tone RMS, Line dB
                self.debug_table.setItem(
                    row, 0, QtWidgets.QTableWidgetItem(f"Ch {report['channel']}"))
                self.debug_table.setItem(
                    row, 1, QtWidgets.QTableWidgetItem(f"{report['overall_rms']:.2f}"))
                self.debug_table.setItem(
                    row, 2, QtWidgets.QTableWidgetItem(f"{report['tone_rms']:.2f}"))

                db_item = QtWidgets.QTableWidgetItem(
                    f"{report['relative_db']:.1f}")
                if report['relative_db'] > -10.0:  # High line noise warning
                    db_item.setForeground(
                        QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                self.debug_table.setItem(row, 3, db_item)

                row += 1

    def save_to_disk(self):
        if not self.all_data:
            return

        # self.all_data contains arrays of shape (1, channels, time)
        # Stacking them results in X shape: (Trials, 1, channels, time)
        X = np.stack(self.all_data, axis=0)
        y = np.array(self.all_labels)

        os.makedirs("data", exist_ok=True)
        now = datetime.now()
        filename = f"data/dataset_{now.strftime('%H%M%S')}.npz"

        np.savez(filename, X=X, y=y)
        print(f"\n--- DATASET SAVED ---")
        print(f"Saved {len(y)} trials to '{filename}'")
        print(f"X shape: {X.shape} -> (Trials, 1, Channels, Time)")
        print(f"y shape: {y.shape} -> (Trials,)")


def main():
    print("Looking for LSL streams...")
    streams = pylsl.resolve_streams()

    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QWidget()
    main_window.setWindowTitle("Optimized EEG Data Collection")
    main_window.resize(1200, 800)
    main_window.setStyleSheet("background-color: #101318; color: white;")

    # Main Horizontal Layout
    main_layout = QtWidgets.QHBoxLayout()
    main_window.setLayout(main_layout)

    # --- LEFT PANEL: VISUAL CUES (75% Width) ---
    cue_frame = QtWidgets.QFrame()
    cue_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel |
                            QtWidgets.QFrame.Raised)
    cue_frame.setStyleSheet("background-color: #1a1e24; border-radius: 10px;")
    cue_layout = QtWidgets.QVBoxLayout(cue_frame)

    cue_label = QtWidgets.QLabel("Connecting...")
    cue_label.setAlignment(QtCore.Qt.AlignCenter)
    cue_label.setMinimumSize(600, 600)
    cue_layout.addWidget(cue_label)

    main_layout.addWidget(cue_frame, stretch=3)

    # --- RIGHT PANEL: DEBUG MENU (25% Width) ---
    debug_frame = QtWidgets.QFrame()
    debug_frame.setFrameStyle(
        QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Raised)
    debug_frame.setStyleSheet(
        "background-color: #1a1e24; border-radius: 10px;")
    debug_layout = QtWidgets.QVBoxLayout(debug_frame)

    debug_title = QtWidgets.QLabel("DEBUG MENU")
    debug_title.setStyleSheet(
        "font-size: 20px; font-weight: bold; color: #aaaaaa;")
    debug_title.setAlignment(QtCore.Qt.AlignCenter)
    debug_layout.addWidget(debug_title)

    status_label = QtWidgets.QLabel("Initializing...")
    status_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
    debug_layout.addWidget(status_label)

    debug_table = QtWidgets.QTableWidget()
    debug_table.setColumnCount(4)
    debug_table.setHorizontalHeaderLabels(["Ch", "O-RMS", "T-RMS", "dB"])
    debug_table.horizontalHeader().setStretchLastSection(True)
    debug_table.verticalHeader().setVisible(False)
    debug_table.setStyleSheet("""
		QTableWidget { background-color: #101318; color: white; border: none; }
		QHeaderView::section { background-color: #2c313c; color: white; border: 1px solid #101318; }
	""")
    debug_layout.addWidget(debug_table)

    main_layout.addWidget(debug_frame, stretch=1)

    main_window.show()

    inlets: List[DataInlet] = []
    next_channel_base = 0

    for info in streams:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE and info.channel_format() != pylsl.cf_string:
            print("Adding data inlet: " + info.name())
            inlets.append(
                DataInlet(info, channel_base_index=next_channel_base))
            next_channel_base += info.channel_count()

    if not inlets:
        cue_label.setText("No LSL Data Streams Found.")
        cue_label.setStyleSheet("color: red; font-size: 32px;")

    session_manager = SessionManager(
        cue_label, debug_table, status_label, inlets)

    def update_loop():
        # 1. Pull and filter data, estimate noise
        for inlet in inlets:
            inlet.pull_and_process()

        # 2. Update state machine, UI, and save trial data if needed
        session_manager.update()

    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update_loop)
    pull_timer.start(PULL_INTERVAL_MS)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
