import sys
import os
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pylsl
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

# Configuration Constants
PULL_INTERVAL_MS = 40  # Timer interval for pulling data and updating UI
PLOT_BUFFER_SAMPLES = 5000  # Number of samples displayed on the graph


def apply_signal_filters(data, fs, bp_enabled, bp_low, bp_high, bp_order, notch_enabled, notch_freq, notch_q):
    """Applies Butterworth Bandpass and IIR Notch filters along time axis (axis=0)."""
    if data.size == 0 or data.shape[0] < 15 or fs <= 0:
        return data

    nyq = fs / 2.0
    filtered = data.copy()

    # 1. Notch Filter
    if notch_enabled and 0 < notch_freq < nyq:
        try:
            b, a = iirnotch(notch_freq, notch_q, fs=fs)
            filtered = filtfilt(b, a, filtered, axis=0)
        except Exception:
            pass

    # 2. Bandpass Filter
    if bp_enabled and 0 < bp_low < bp_high < nyq:
        try:
            sos = butter(bp_order, [bp_low, bp_high],
                         btype='bandpass', fs=fs, output='sos')
            filtered = sosfiltfilt(sos, filtered, axis=0)
        except Exception:
            pass

    return filtered


class LSLStreamInlet:
    """Handles connecting to an LSL stream, buffering plot data, and recording raw samples."""

    def __init__(self, info: pylsl.StreamInfo):
        self.inlet = pylsl.StreamInlet(
            info, processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter
        )
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.fs = info.nominal_srate()

        # Rolling buffer for oscilloscope display
        self.plot_buffer = np.zeros((PLOT_BUFFER_SAMPLES, self.channel_count))
        self.is_recording = False
        self.record_buffer = []

    def pull(self):
        """Pulls available chunks from LSL and updates buffers."""
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0)
        if not timestamps:
            return

        data = np.array(chunk)

        # Update rolling plot buffer
        n_samples = data.shape[0]
        if n_samples >= PLOT_BUFFER_SAMPLES:
            self.plot_buffer = data[-PLOT_BUFFER_SAMPLES:, :]
        else:
            self.plot_buffer = np.vstack(
                (self.plot_buffer[n_samples:, :], data))

        # Collect raw samples if actively recording
        if self.is_recording:
            self.record_buffer.append(data)

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


class EEGRecorderApp(QtWidgets.QWidget):
    """Main Application GUI with live graphing, filter options, and recording controls."""

    def __init__(self, inlets):
        super().__init__()
        self.inlets = inlets
        self.curves = []
        self.init_ui()
        self.enforce_nyquist_limits()

    def init_ui(self):
        self.setWindowTitle("LSL EEG Stream Recorder & Signal Filter")
        self.resize(1100, 750)
        self.setStyleSheet("background-color: #101318; color: white;")

        main_layout = QtWidgets.QVBoxLayout(self)

        # Top Control Bar Layout
        control_layout = QtWidgets.QHBoxLayout()

        self.btn_start = QtWidgets.QPushButton("Start Recording")
        self.btn_start.setStyleSheet(
            "background-color: #28a745; color: white; padding: 10px 18px; font-weight: bold; border-radius: 4px;"
        )
        self.btn_start.clicked.connect(self.start_recording)

        self.btn_stop = QtWidgets.QPushButton("Stop & Save Recording")
        self.btn_stop.setStyleSheet(
            "background-color: #dc3545; color: white; padding: 10px 18px; font-weight: bold; border-radius: 4px;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_recording)

        # Sample Rate Display Label
        fs_display = ", ".join(
            [f"{inl.fs:.1f} Hz" for inl in self.inlets]) if self.inlets else "N/A"
        self.lbl_fs = QtWidgets.QLabel(f"Sample Rate: {fs_display}")
        self.lbl_fs.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #17a2b8; margin-left: 10px;")

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; margin-left: 10px;")

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.lbl_fs)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # Filter Control Panel
        filter_group = QtWidgets.QGroupBox("Filter Configuration & Routing")
        filter_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #2a2e37; border-radius: 6px; margin-top: 6px; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #17a2b8; }"
        )
        filter_layout = QtWidgets.QGridLayout(filter_group)

        # Routing Checkboxes
        self.chk_filter_vis = QtWidgets.QCheckBox("Filter Visualization")
        self.chk_filter_vis.setChecked(True)
        self.chk_filter_rec = QtWidgets.QCheckBox("Filter Recording")
        self.chk_filter_rec.setChecked(False)

        # Bandpass Controls
        self.chk_bp = QtWidgets.QCheckBox("Enable Bandpass")
        self.chk_bp.setChecked(True)

        self.spn_bp_low = QtWidgets.QDoubleSpinBox()
        self.spn_bp_low.setPrefix("Low: ")
        self.spn_bp_low.setSuffix(" Hz")
        self.spn_bp_low.setValue(1.0)
        self.spn_bp_low.setSingleStep(0.5)

        self.spn_bp_high = QtWidgets.QDoubleSpinBox()
        self.spn_bp_high.setPrefix("High: ")
        self.spn_bp_high.setSuffix(" Hz")
        self.spn_bp_high.setValue(40.0)
        self.spn_bp_high.setSingleStep(1.0)

        self.spn_bp_order = QtWidgets.QSpinBox()
        self.spn_bp_order.setPrefix("Order: ")
        self.spn_bp_order.setRange(1, 10)
        self.spn_bp_order.setValue(4)

        # Notch Controls
        self.chk_notch = QtWidgets.QCheckBox("Enable Notch")
        self.chk_notch.setChecked(True)

        self.spn_notch_freq = QtWidgets.QDoubleSpinBox()
        self.spn_notch_freq.setPrefix("Freq: ")
        self.spn_notch_freq.setSuffix(" Hz")
        self.spn_notch_freq.setValue(50.0)
        self.spn_notch_freq.setSingleStep(1.0)

        self.spn_notch_q = QtWidgets.QDoubleSpinBox()
        self.spn_notch_q.setPrefix("Q: ")
        self.spn_notch_q.setRange(1.0, 100.0)
        self.spn_notch_q.setValue(30.0)

        # Layout Assembly
        filter_layout.addWidget(self.chk_filter_vis, 0, 0)
        filter_layout.addWidget(self.chk_filter_rec, 0, 1)

        filter_layout.addWidget(self.chk_bp, 1, 0)
        filter_layout.addWidget(self.spn_bp_low, 1, 1)
        filter_layout.addWidget(self.spn_bp_high, 1, 2)
        filter_layout.addWidget(self.spn_bp_order, 1, 3)

        filter_layout.addWidget(self.chk_notch, 2, 0)
        filter_layout.addWidget(self.spn_notch_freq, 2, 1)
        filter_layout.addWidget(self.spn_notch_q, 2, 2)

        main_layout.addWidget(filter_group)

        # Connect frequency change signals to enforce Nyquist limits
        self.spn_bp_low.valueChanged.connect(self.enforce_nyquist_limits)
        self.spn_bp_high.valueChanged.connect(self.enforce_nyquist_limits)
        self.spn_notch_freq.valueChanged.connect(self.enforce_nyquist_limits)

        # PyQtGraph Live Plotting Oscilloscope Widget
        self.plot_widget = pg.PlotWidget(title="Live EEG Signal Waveforms")
        self.plot_widget.setBackground("#101318")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Voltage Offset (uV)")
        self.plot_widget.setLabel("bottom", "Samples")
        main_layout.addWidget(self.plot_widget)

        # Periodic update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(PULL_INTERVAL_MS)

    def enforce_nyquist_limits(self):
        """Ensures selected cutoffs do not exceed the Nyquist frequency limit."""
        if not self.inlets:
            return

        min_fs = min([inl.fs for inl in self.inlets if inl.fs > 0] or [250.0])
        nyquist = min_fs / 2.0
        max_safe_freq = max(0.1, nyquist - 0.1)

        # Update SpinBox Maximums
        self.spn_bp_high.setMaximum(max_safe_freq)
        self.spn_bp_low.setMaximum(max_safe_freq - 0.1)
        self.spn_notch_freq.setMaximum(max_safe_freq)

        # Ensure Low < High
        if self.spn_bp_low.value() >= self.spn_bp_high.value():
            self.spn_bp_low.setValue(max(0.1, self.spn_bp_high.value() - 0.5))

    def get_filter_params(self):
        """Retrieves user filter settings."""
        return {
            'bp_enabled': self.chk_bp.isChecked(),
            'bp_low': self.spn_bp_low.value(),
            'bp_high': self.spn_bp_high.value(),
            'bp_order': self.spn_bp_order.value(),
            'notch_enabled': self.chk_notch.isChecked(),
            'notch_freq': self.spn_notch_freq.value(),
            'notch_q': self.spn_notch_q.value()
        }

    def start_recording(self):
        """Starts recording on all active inlets."""
        for inlet in self.inlets:
            inlet.start_recording()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Status: Recording...")
        self.status_label.setStyleSheet(
            "color: #ffc107; font-size: 13px; font-weight: bold;")

    def stop_recording(self):
        """Stops recording and dumps buffered data into a NumPy archive."""
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/eeg_recording_{timestamp}.npz"

        params = self.get_filter_params()
        apply_rec_filter = self.chk_filter_rec.isChecked()

        recorded_dict = {}
        for idx, inlet in enumerate(self.inlets):
            data = inlet.stop_recording()

            # Filter recorded data if enabled
            if apply_rec_filter and data.size > 0:
                data = apply_signal_filters(data, inlet.fs, **params)

            recorded_dict[f"stream_{idx}_{inlet.name}"] = data

        np.savez(filename, **recorded_dict)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText(f"Status: Saved data to '{filename}'")
        self.status_label.setStyleSheet(
            "color: #28a745; font-size: 13px; font-weight: bold;")

    def update_loop(self):
        """Pulls LSL data and updates signal visualization curves."""
        for inlet in self.inlets:
            inlet.pull()

        if not self.inlets:
            return

        params = self.get_filter_params()
        apply_vis_filter = self.chk_filter_vis.isChecked()

        # Prepare visualization buffers
        processed_buffers = []
        for inl in self.inlets:
            buf = inl.plot_buffer
            if apply_vis_filter:
                buf = apply_signal_filters(buf, inl.fs, **params)
            processed_buffers.append(buf)

        combined_buffers = np.hstack(processed_buffers)
        total_channels = combined_buffers.shape[1]

        # Dynamically allocate graph curves
        while len(self.curves) < total_channels:
            pen_color = pg.intColor(len(self.curves), hues=total_channels)
            curve = self.plot_widget.plot(pen=pg.mkPen(pen_color, width=1.5))
            self.curves.append(curve)

        # Draw stacked channel waveforms
        offset_step = 50.0
        for idx in range(total_channels):
            ch_data = combined_buffers[:, idx]
            self.curves[idx].setData(ch_data + (idx * offset_step))


def main():
    print("Searching for active LSL streams...")
    streams = pylsl.resolve_streams()

    app = QtWidgets.QApplication(sys.argv)

    inlets = []
    for info in streams:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE:
            print(
                f"Connecting to LSL Stream: {info.name()} ({info.nominal_srate()} Hz)")
            inlets.append(LSLStreamInlet(info))

    gui = EEGRecorderApp(inlets)
    gui.show()

    if not inlets:
        gui.status_label.setText("Status: No LSL streams detected!")
        gui.status_label.setStyleSheet(
            "color: #dc3545; font-size: 13px; font-weight: bold;")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
