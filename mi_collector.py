import math
import sys
import os
import random
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pylsl
import scipy.signal as signal
from typing import List, TypedDict
from datetime import datetime

# --- SIGNAL PROCESSING CONSTANTS ---
LINE_NOISE_FREQ_HZ = 50.0
BAND_LOW = 4.0   # Optimized for Mu band (motor imagery)
BAND_HIGH = 38.0  # Optimized for Beta band (motor imagery)

# --- DATA COLLECTION & PARADIGM TIMINGS ---
ACTIONS = ["Relajado", "Lado Izquierdo", "Lado Derecho"]
TRIALS_PER_ACTION = 10
PREP_DURATION = 2.0     # Crosshair to increase attention
CUE_DURATION = 2.0      # Text indicating the task
RECORD_DURATION = 4.0   # Dynamic GIF execution and data recording
RELAX_DURATION = 5.0    # Rest between trials

# --- INTERFACE CONSTANTS ---
PULL_INTERVAL_MS = 50
PLOT_BUFFER_SAMPLES = 1000  # Samples to retain for live oscilloscope plot


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

        # Oscilloscope rolling plot buffer (filtered data for visual monitor only)
        self.plot_buffer = np.zeros((PLOT_BUFFER_SAMPLES, self.channel_count))

        # Filtering setup (Notch for line noise, Bandpass for ERD/ERS 4-38Hz)
        b_notch, a_notch = signal.iirnotch(LINE_NOISE_FREQ_HZ, 30.0, self.fs)
        self.sos_notch = np.hstack((b_notch, a_notch)).reshape(1, 6)
        self.sos_bp = signal.butter(
            4, [BAND_LOW, BAND_HIGH], btype="bandpass", fs=self.fs, output="sos")

        self.notch_state = None
        self.bp_state = None

        # Recording State (Stores NON-FILTERED raw signal to prevent distortion)
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
        """Pulls new data from LSL, applies display filters, and records raw non-filtered data."""
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.input_buffer.shape[0], dest_obj=self.input_buffer)

        if not ts:
            return

        ts = np.asarray(ts)
        raw_y = self.input_buffer[: ts.size, :].copy()

        # Initialize filter states on first pass
        if self.notch_state is None or self.bp_state is None:
            zi_notch_single = signal.sosfilt_zi(self.sos_notch)
            zi_bp_single = signal.sosfilt_zi(self.sos_bp)
            self.notch_state = zi_notch_single[:, :, np.newaxis] * raw_y[0, :]
            self.bp_state = zi_bp_single[:, :, np.newaxis] * raw_y[0, :]

        # Apply display filters (Used ONLY for oscilloscope plot and noise estimation)
        out_n, self.notch_state = signal.sosfilt(
            self.sos_notch, raw_y, axis=0, zi=self.notch_state)
        filtered_y, self.bp_state = signal.sosfilt(
            self.sos_bp, out_n, axis=0, zi=self.bp_state)

        # Append to oscilloscope rolling plot buffer
        n_samples = filtered_y.shape[0]
        if n_samples >= PLOT_BUFFER_SAMPLES:
            self.plot_buffer = filtered_y[-PLOT_BUFFER_SAMPLES:, :]
        else:
            self.plot_buffer = np.vstack(
                (self.plot_buffer[n_samples:, :], filtered_y))

        # Calculate noise metrics on filtered display data
        self.latest_noise_report = self._estimate_line_noise(ts, filtered_y)

        # Append RAW NON-FILTERED data to record buffer if recording is active
        if self.is_recording:
            self.record_buffer.append(raw_y)

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
    """Manages visual cues, paradigm state transitions, pause states, and signal indicators."""

    def __init__(self, cue_label: QtWidgets.QLabel, debug_table: QtWidgets.QTableWidget,
                 status_label: QtWidgets.QLabel, quality_label: QtWidgets.QLabel,
                 plot_widget: pg.PlotWidget, inlets: List[DataInlet]):
        self.cue_label = cue_label
        self.debug_table = debug_table
        self.status_label = status_label
        self.quality_label = quality_label
        self.plot_widget = plot_widget
        self.inlets = inlets

        # Build and shuffle randomized trials
        self.trial_sequence = ACTIONS * TRIALS_PER_ACTION
        random.shuffle(self.trial_sequence)
        self.total_trials = len(self.trial_sequence)
        self.current_trial = 0

        # State Machine
        self.state = "STARTING"
        self.previous_state = "STARTING"
        self.state_start = pylsl.local_clock()
        self.pause_start_time = 0.0
        self.accumulated_pause_duration = 0.0
        self.is_paused = False
        self.current_movie = None

        # Dataset storage
        self.all_data = []
        self.all_labels = []

        # PyQTGraph setup
        self.plot_widget.setBackground('#101318')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Voltage (uV)')
        self.plot_widget.setLabel('bottom', 'Samples')
        self.curves = []

    def set_visual_cue(self, text: str, image_names: List[str], color: str = "white"):
        """Attempts to load a GIF or Image. Falls back to styled text if missing."""
        if self.current_movie:
            self.current_movie.stop()
            self.current_movie = None

        file_found = False
        for img_name in image_names:
            img_path = f"images/{img_name}"
            if os.path.exists(img_path):
                if img_path.lower().endswith(".gif"):
                    self.current_movie = QtGui.QMovie(img_path)
                    self.cue_label.setMovie(self.current_movie)
                    self.current_movie.start()
                else:
                    pixmap = QtGui.QPixmap(img_path)
                    self.cue_label.setPixmap(pixmap.scaled(
                        self.cue_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                file_found = True
                break

        if not file_found:
            self.cue_label.setText(text)
            self.cue_label.setStyleSheet(
                f"color: {color}; font-size: 56px; font-weight: bold; background-color: #101318;")

    def toggle_pause(self):
        """Toggles pause/resume state without losing trial progress."""
        now = pylsl.local_clock()
        if not self.is_paused:
            self.is_paused = True
            self.previous_state = self.state
            self.state = "PAUSED"
            self.pause_start_time = now
            self.set_visual_cue(
                "PAUSED\nPresiona 'P' o Espacio", [], "#FFA500")
            # Pause inlets if active
            for inlet in self.inlets:
                if inlet.is_recording:
                    inlet.is_recording = False
        else:
            self.is_paused = False
            self.accumulated_pause_duration += (now - self.pause_start_time)
            self.state = self.previous_state
            # Re-enable inlet recording if unpaused during EXECUTION state
            if self.state == "EXECUTION":
                for inlet in self.inlets:
                    inlet.is_recording = True

    def update(self):
        if self.is_paused:
            self.status_label.setText(
                f"Trials Remaining: {self.total_trials - self.current_trial} / {self.total_trials}\nStatus: PAUSED")
            return

        now = pylsl.local_clock()
        elapsed = now - self.state_start - self.accumulated_pause_duration

        # Update Debug Menu metrics & graphs
        self.update_debug_menu()

        if self.state == "STARTING":
            self.set_visual_cue("Empezando en breves\nRelájate", [], "#FFFFFF")
            if elapsed > 3.0:
                self.transition_to_prep(now)

        elif self.state == "PREPARATION":
            if elapsed >= PREP_DURATION:
                self.transition_to_task_cue(now)

        elif self.state == "TASK_CUE":
            if elapsed >= CUE_DURATION:
                self.transition_to_record(now)

        elif self.state == "EXECUTION":
            if elapsed >= RECORD_DURATION:
                self.save_trial_and_transition(now)

        elif self.state == "RELAX":
            if elapsed >= RELAX_DURATION:
                self.transition_to_prep(now)

        elif self.state == "DONE":
            self.set_visual_cue(
                "¡Sesión Completada!\nDatos guardados sin filtro.", [], "#00FFFF")

    def transition_to_prep(self, now):
        self.state = "PREPARATION"
        self.state_start = now
        self.accumulated_pause_duration = 0.0
        self.set_visual_cue("+", ["crosshair.png", "crosshair.jpg"], "#FFFFFF")

    def transition_to_task_cue(self, now):
        self.state = "TASK_CUE"
        self.state_start = now
        self.accumulated_pause_duration = 0.0
        action = self.trial_sequence[self.current_trial]
        self.set_visual_cue(f"Ejecuta:\n{action}", [], "#FFFF00")

    def transition_to_record(self, now):
        self.state = "EXECUTION"
        self.state_start = now
        self.accumulated_pause_duration = 0.0
        action = self.trial_sequence[self.current_trial]

        safe_action_name = action.lower().replace(" ", "_")
        image_targets = [f"{safe_action_name}.gif",
                         f"{safe_action_name}.png", f"{safe_action_name}.jpg"]

        self.set_visual_cue(f"Ejecuta:\n{action}", image_targets, "#00FF00")

        for inlet in self.inlets:
            inlet.start_recording()

    def transition_to_relax(self, now):
        self.state = "RELAX"
        self.state_start = now
        self.accumulated_pause_duration = 0.0
        self.set_visual_cue(
            "Relax", ["relax.png", "relax.jpg", "eye_icon.png"], "#AAAAAA")

    def save_trial_and_transition(self, now):
        action = self.trial_sequence[self.current_trial]
        trial_data = []

        for inlet in self.inlets:
            # Non-filtered raw EEG retrieved here
            raw_data = inlet.stop_recording()
            expected_samples = int(inlet.fs * RECORD_DURATION)

            if raw_data.shape[0] > expected_samples:
                raw_data = raw_data[:expected_samples, :]
            elif raw_data.shape[0] < expected_samples:
                pad_size = expected_samples - raw_data.shape[0]
                padding = np.zeros((pad_size, raw_data.shape[1]))
                raw_data = np.vstack([raw_data, padding])

            # Shape manipulation to ML format: (1, Channels, Time)
            raw_data = raw_data.T
            raw_data = np.expand_dims(raw_data, axis=0)
            trial_data.append(raw_data)

        if trial_data:
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
        """Updates table, live signal oscilloscope plot, and noise indicators."""
        if self.state == "DONE":
            self.status_label.setText("Status: Hecho")
            return

        trials_left = self.total_trials - self.current_trial
        self.status_label.setText(
            f"Trials Remaining: {trials_left} / {self.total_trials}\nCurrent State: {self.state}")

        row = 0
        all_dbs = []
        all_rms = []

        for inlet in self.inlets:
            for report in inlet.latest_noise_report:
                if row >= self.debug_table.rowCount():
                    self.debug_table.insertRow(row)

                self.debug_table.setItem(
                    row, 0, QtWidgets.QTableWidgetItem(f"Ch {report['channel']}"))
                self.debug_table.setItem(
                    row, 1, QtWidgets.QTableWidgetItem(f"{report['overall_rms']:.1f}"))
                self.debug_table.setItem(
                    row, 2, QtWidgets.QTableWidgetItem(f"{report['tone_rms']:.1f}"))

                db_item = QtWidgets.QTableWidgetItem(
                    f"{report['relative_db']:.1f}")
                if report['relative_db'] > -10.0:
                    db_item.setForeground(
                        QtGui.QBrush(QtGui.QColor(255, 60, 60)))
                self.debug_table.setItem(row, 3, db_item)

                all_dbs.append(report['relative_db'])
                all_rms.append(report['overall_rms'])
                row += 1

        # Calculate Signal Quality Indicator
        if all_dbs:
            avg_db = np.mean(all_dbs)
            avg_rms = np.mean(all_rms)

            if avg_db < -15.0 and avg_rms < 100.0:
                quality_text = "Quality: EXCELLENT"
                color = "#00FF7F"  # Spring Green
            elif avg_db < -8.0 and avg_rms < 250.0:
                quality_text = "Quality: GOOD"
                color = "#7CFC00"  # Lawn Green
            elif avg_db < -4.0:
                quality_text = "Quality: FAIR"
                color = "#FFD700"  # Gold
            else:
                quality_text = "Quality: POOR (High Noise)"
                color = "#FF4500"  # Orange Red

            self.quality_label.setText(quality_text)
            self.quality_label.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {color};")

        # Update PyQTGraph Oscilloscope Plot
        if self.inlets:
            # Concatenate plot buffers across inlets
            combined_buffers = np.hstack(
                [inl.plot_buffer for inl in self.inlets])
            total_channels = combined_buffers.shape[1]

            # Ensure matching plot curve objects
            while len(self.curves) < total_channels:
                pen_color = pg.intColor(len(self.curves), hues=total_channels)
                curve = self.plot_widget.plot(
                    pen=pg.mkPen(pen_color, width=1.5))
                self.curves.append(curve)

            # Offset channel curves visually for stacked display
            offset_step = 50.0
            for idx in range(total_channels):
                ch_data = combined_buffers[:, idx]
                offset = idx * offset_step
                self.curves[idx].setData(ch_data + offset)

    def end_early(self):
        """Ends the session immediately and saves completed trials."""
        if self.state != "DONE":
            self.state = "DONE"
            self.set_visual_cue(
                "Sesión Finalizada Temprano\nDatos guardados.", [], "#FF4500")
            self.save_to_disk()

    def save_to_disk(self):
        if not self.all_data:
            return

        # Stacking raw un-filtered data array: (Trials, 1, Channels, Time)
        X = np.stack(self.all_data, axis=0)
        y = np.array(self.all_labels)

        os.makedirs("data", exist_ok=True)
        now = datetime.now()
        filename = f"data/dataset_{now.strftime('%H%M%S')}.npz"

        np.savez(filename, X=X, y=y)
        print(f"\n--- NON-FILTERED DATASET SAVED ---")
        print(f"Saved {len(y)} trials to '{filename}'")
        print(f"X shape: {X.shape} -> (Trials, 1, Channels, Time)")
        print(f"y shape: {y.shape} -> (Trials,)")


class MainWindow(QtWidgets.QWidget):
    """Main Application Window with KeyBindings and Layout Management."""

    def __init__(self, debug_frame: QtWidgets.QFrame, session_manager_holder: list):
        super().__init__()
        self.debug_frame = debug_frame
        self.session_manager_holder = session_manager_holder

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Toggle Debug Menu on 'D' Key
        if event.key() == QtCore.Qt.Key_D:
            is_visible = self.debug_frame.isVisible()
            self.debug_frame.setVisible(not is_visible)
        # Toggle Pause on 'P' or Space Key
        elif event.key() in (QtCore.Qt.Key_P, QtCore.Qt.Key_Space):
            if self.session_manager_holder and self.session_manager_holder[0]:
                self.session_manager_holder[0].toggle_pause()
        # End early on 'Escape'
        elif event.key() == QtCore.Qt.Key_Escape:
            if self.session_manager_holder and self.session_manager_holder[0]:
                self.session_manager_holder[0].end_early()
        else:
            super().keyPressEvent(event)


def main():
    print("Looking for LSL streams...")
    streams = pylsl.resolve_streams()

    app = QtWidgets.QApplication(sys.argv)

    # Main Window setup
    debug_frame = QtWidgets.QFrame()
    session_manager_holder = [None]
    main_window = MainWindow(debug_frame, session_manager_holder)
    main_window.setWindowTitle("Optimized EEG Data Collection")
    main_window.resize(1280, 850)
    main_window.setStyleSheet("background-color: #101318; color: white;")

    # Main Horizontal Layout
    main_layout = QtWidgets.QHBoxLayout()
    main_window.setLayout(main_layout)

    # --- LEFT PANEL: VISUAL CUES ---
    left_container = QtWidgets.QWidget()
    left_layout = QtWidgets.QVBoxLayout(left_container)
    left_layout.setContentsMargins(0, 0, 0, 0)

    top_bar = QtWidgets.QHBoxLayout()
    btn_toggle_debug = QtWidgets.QPushButton("Toggle Debug Menu (D)")
    btn_toggle_debug.setStyleSheet("""
		QPushButton { background-color: #2c313c; color: white; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
		QPushButton:hover { background-color: #3e4451; }
	""")
    top_bar.addWidget(btn_toggle_debug)

    btn_pause = QtWidgets.QPushButton("Pausar / Reanudar (P)")
    btn_pause.setStyleSheet("""
		QPushButton { background-color: #d19a66; color: black; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
		QPushButton:hover { background-color: #e5c07b; }
	""")
    top_bar.addWidget(btn_pause)

    btn_end_early = QtWidgets.QPushButton("Finalizar y Guardar (Esc)")
    btn_end_early.setStyleSheet("""
		QPushButton { background-color: #e06c75; color: white; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
		QPushButton:hover { background-color: #be5046; }
	""")
    top_bar.addWidget(btn_end_early)

    top_bar.addStretch()
    left_layout.addLayout(top_bar)

    cue_frame = QtWidgets.QFrame()
    cue_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel |
                            QtWidgets.QFrame.Raised)
    cue_frame.setStyleSheet("background-color: #1a1e24; border-radius: 10px;")
    cue_layout = QtWidgets.QVBoxLayout(cue_frame)

    cue_label = QtWidgets.QLabel("Conectando...")
    cue_label.setAlignment(QtCore.Qt.AlignCenter)
    cue_label.setMinimumSize(500, 500)
    cue_layout.addWidget(cue_label)

    left_layout.addWidget(cue_frame, stretch=1)
    main_layout.addWidget(left_container, stretch=3)

    # --- RIGHT PANEL: DEBUG MENU (Toggleable) ---
    debug_frame.setFrameStyle(
        QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Raised)
    debug_frame.setStyleSheet(
        "background-color: #1a1e24; border-radius: 10px;")
    debug_layout = QtWidgets.QVBoxLayout(debug_frame)

    debug_title = QtWidgets.QLabel("MENÚ DE DEPURACIÓN")
    debug_title.setStyleSheet(
        "font-size: 18px; font-weight: bold; color: #61afef;")
    debug_title.setAlignment(QtCore.Qt.AlignCenter)
    debug_layout.addWidget(debug_title)

    status_label = QtWidgets.QLabel("Inicializando...")
    status_label.setStyleSheet("font-size: 13px; margin-bottom: 4px;")
    debug_layout.addWidget(status_label)

    quality_label = QtWidgets.QLabel("Quality: Evaluating...")
    quality_label.setStyleSheet(
        "font-size: 14px; font-weight: bold; color: #e5c07b;")
    debug_layout.addWidget(quality_label)

    # Graph Widget for EEG Waveforms
    plot_widget = pg.PlotWidget(title="Live Filtered EEG Waves")
    plot_widget.setMinimumHeight(200)
    debug_layout.addWidget(plot_widget)

    # Noise and RMS metrics table
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

    # Connect UI button actions
    btn_toggle_debug.clicked.connect(
        lambda: debug_frame.setVisible(not debug_frame.isVisible()))

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
        cue_label, debug_table, status_label, quality_label, plot_widget, inlets)
    session_manager_holder[0] = session_manager

    btn_pause.clicked.connect(session_manager.toggle_pause)
    btn_end_early.clicked.connect(session_manager.end_early)

    def update_loop():
        # 1. Pull data, update buffers
        for inlet in inlets:
            inlet.pull_and_process()

        # 2. Update state machine, UI, and live signal plots
        session_manager.update()

    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update_loop)
    pull_timer.start(PULL_INTERVAL_MS)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
