import sys
import os
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pylsl
import scipy.signal as signal

# Configuration Constants
PULL_INTERVAL_MS = 40  # Timer interval for pulling data and updating the UI
PLOT_BUFFER_SAMPLES = 5000  # Number of samples displayed on the graph


class LSLStreamInlet:
    """Handles connecting to an LSL stream, buffering plot data, and applying dynamic real-time filters."""

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

        # Filter SOS matrices & States
        self.sos_notch = None
        self.sos_bp = None
        self.notch_state = None
        self.bp_state = None
        self.current_filter_params = None

    def update_filter_config(self, enable_bp, bp_low, bp_high, enable_notch, notch_freq):
        """Re-configures SciPy SOS filter matrices when settings change."""
        new_params = (enable_bp, bp_low, bp_high, enable_notch, notch_freq)
        if self.current_filter_params == new_params:
            return  # No change

        self.current_filter_params = new_params
        nyquist = self.fs / 2.0

        # Build Notch Filter SOS
        if enable_notch and 0 < notch_freq < nyquist:
            b_notch, a_notch = signal.iirnotch(notch_freq, 30.0, fs=self.fs)
            self.sos_notch = np.hstack((b_notch, a_notch)).reshape(1, 6)
        else:
            self.sos_notch = None

        # Build Bandpass Filter SOS
        if enable_bp and 0 < bp_low < bp_high < nyquist:
            self.sos_bp = signal.butter(
                4, [bp_low, bp_high], btype="bandpass", fs=self.fs, output="sos"
            )
        else:
            self.sos_bp = None

        # Reset states for fresh filtering
        self.notch_state = None
        self.bp_state = None

    def _apply_live_filter(self, data: np.ndarray) -> np.ndarray:
        """Filters incoming streaming data online using standard stateful sosfilt."""
        filtered_data = data.copy()

        # 1. Apply Notch Filter if enabled
        if self.sos_notch is not None:
            if self.notch_state is None:
                zi_single = signal.sosfilt_zi(self.sos_notch)
                self.notch_state = zi_single[:, :,
                                             np.newaxis] * filtered_data[0, :]
            filtered_data, self.notch_state = signal.sosfilt(
                self.sos_notch, filtered_data, axis=0, zi=self.notch_state
            )

        # 2. Apply Bandpass Filter if enabled
        if self.sos_bp is not None:
            if self.bp_state is None:
                zi_single = signal.sosfilt_zi(self.sos_bp)
                self.bp_state = zi_single[:, :,
                                          np.newaxis] * filtered_data[0, :]
            filtered_data, self.bp_state = signal.sosfilt(
                self.sos_bp, filtered_data, axis=0, zi=self.bp_state
            )

        return filtered_data

    def pull(self, draw_filtered: bool):
        """Pulls available chunks from LSL, updates raw/filtered buffers."""
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0)
        if not timestamps:
            return

        raw_data = np.array(chunk)

        # Process live display buffer
        if draw_filtered:
            plot_data = self._apply_live_filter(raw_data)
        else:
            plot_data = raw_data

        # Update rolling plot buffer
        n_samples = plot_data.shape[0]
        if n_samples >= PLOT_BUFFER_SAMPLES:
            self.plot_buffer = plot_data[-PLOT_BUFFER_SAMPLES:, :]
        else:
            self.plot_buffer = np.vstack(
                (self.plot_buffer[n_samples:, :], plot_data))

        # Collect raw samples for recording
        if self.is_recording:
            self.record_buffer.append(raw_data)

    def start_recording(self):
        self.is_recording = True
        self.record_buffer = []

    def stop_recording(self, save_filtered: bool) -> np.ndarray:
        self.is_recording = False
        if not self.record_buffer:
            return np.array([])

        recorded_data = np.vstack(self.record_buffer)
        self.record_buffer = []

        if not save_filtered:
            return recorded_data

        # Apply zero-phase forward-backward filtering on saved continuous data
        out_data = recorded_data.copy()
        if self.sos_notch is not None and out_data.shape[0] > 15:
            out_data = signal.sosfiltfilt(self.sos_notch, out_data, axis=0)
        if self.sos_bp is not None and out_data.shape[0] > 15:
            out_data = signal.sosfiltfilt(self.sos_bp, out_data, axis=0)

        return out_data


class EEGRecorderApp(QtWidgets.QWidget):
    """Main Application GUI with live graphing, filtering customization, and recording controls."""

    def __init__(self, inlets):
        super().__init__()
        self.inlets = inlets
        self.curves = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LSL EEG Stream Recorder & Real-time Filter")
        self.resize(1150, 700)
        self.setStyleSheet("background-color: #101318; color: white;")

        main_layout = QtWidgets.QVBoxLayout(self)

        # Top Control Bar (Start / Stop / Status)
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

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; margin-left: 10px;")

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # Filtering Settings Panel
        filter_box = QtWidgets.QGroupBox("Filtering Options")
        filter_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #333; margin-top: 5px; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }"
        )
        filter_layout = QtWidgets.QHBoxLayout(filter_box)

        # Target Selection Toggles
        self.chk_draw_filtered = QtWidgets.QCheckBox("Draw Filtered")
        self.chk_draw_filtered.setChecked(True)

        self.chk_save_filtered = QtWidgets.QCheckBox("Save Filtered")
        self.chk_save_filtered.setChecked(False)

        # Bandpass Config
        self.chk_bp = QtWidgets.QCheckBox("Bandpass:")
        self.chk_bp.setChecked(True)

        self.spn_bp_low = QtWidgets.QDoubleSpinBox()
        self.spn_bp_low.setRange(0.1, 100.0)
        self.spn_bp_low.setValue(1.0)
        self.spn_bp_low.setSuffix(" Hz")

        self.spn_bp_high = QtWidgets.QDoubleSpinBox()
        self.spn_bp_high.setRange(1.0, 500.0)
        self.spn_bp_high.setValue(40.0)
        self.spn_bp_high.setSuffix(" Hz")

        # Notch Config
        self.chk_notch = QtWidgets.QCheckBox("Notch:")
        self.chk_notch.setChecked(True)

        self.spn_notch_freq = QtWidgets.QDoubleSpinBox()
        self.spn_notch_freq.setRange(10.0, 200.0)
        self.spn_notch_freq.setValue(50.0)
        self.spn_notch_freq.setSuffix(" Hz")

        # Add items to filter layout
        filter_layout.addWidget(self.chk_draw_filtered)
        filter_layout.addWidget(self.chk_save_filtered)
        filter_layout.addWidget(QtWidgets.QLabel("|"))
        filter_layout.addWidget(self.chk_bp)
        filter_layout.addWidget(QtWidgets.QLabel("Low:"))
        filter_layout.addWidget(self.spn_bp_low)
        filter_layout.addWidget(QtWidgets.QLabel("High:"))
        filter_layout.addWidget(self.spn_bp_high)
        filter_layout.addWidget(QtWidgets.QLabel("|"))
        filter_layout.addWidget(self.chk_notch)
        filter_layout.addWidget(QtWidgets.QLabel("Freq:"))
        filter_layout.addWidget(self.spn_notch_freq)
        filter_layout.addStretch()

        main_layout.addWidget(filter_box)

        # PyQtGraph Plot Widget
        self.plot_widget = pg.PlotWidget(title="Live EEG Signal Waveforms")
        self.plot_widget.setBackground("#101318")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Voltage Offset (uV)")
        self.plot_widget.setLabel("bottom", "Samples")
        main_layout.addWidget(self.plot_widget)

        # Update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(PULL_INTERVAL_MS)

    def update_filters(self):
        """Passes current UI filter parameters to each stream inlet."""
        enable_bp = self.chk_bp.isChecked()
        bp_low = self.spn_bp_low.value()
        bp_high = self.spn_bp_high.value()

        enable_notch = self.chk_notch.isChecked()
        notch_freq = self.spn_notch_freq.value()

        for inlet in self.inlets:
            inlet.update_filter_config(
                enable_bp, bp_low, bp_high, enable_notch, notch_freq)

    def start_recording(self):
        """Starts recording on all active inlets."""
        for inlet in self.inlets:
            inlet.start_recording()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Status: Recording...")
        self.status_label.setStyleSheet(
            "color: #ffc107; font-size: 14px; font-weight: bold;")

    def stop_recording(self):
        """Stops recording and saves data (raw or filtered depending on checkbox)."""
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filtered = self.chk_save_filtered.isChecked()
        suffix = "filtered" if save_filtered else "raw"
        filename = f"recordings/eeg_recording_{suffix}_{timestamp}.npz"

        self.update_filters()
        recorded_dict = {}
        for idx, inlet in enumerate(self.inlets):
            data = inlet.stop_recording(save_filtered=save_filtered)
            recorded_dict[f"stream_{idx}_{inlet.name}"] = data

        np.savez(filename, **recorded_dict)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText(
            f"Status: Saved {suffix} data to '{filename}'")
        self.status_label.setStyleSheet(
            "color: #28a745; font-size: 14px; font-weight: bold;")

    def update_loop(self):
        """Pulls LSL data and updates signal visualization curves."""
        self.update_filters()
        draw_filtered = self.chk_draw_filtered.isChecked()

        for inlet in self.inlets:
            inlet.pull(draw_filtered=draw_filtered)

        if not self.inlets:
            return

        combined_buffers = np.hstack([inl.plot_buffer for inl in self.inlets])
        total_channels = combined_buffers.shape[1]

        while len(self.curves) < total_channels:
            pen_color = pg.intColor(len(self.curves), hues=total_channels)
            curve = self.plot_widget.plot(pen=pg.mkPen(pen_color, width=1.5))
            self.curves.append(curve)

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
            print(f"Connecting to LSL Stream: {info.name()}")
            inlets.append(LSLStreamInlet(info))

    gui = EEGRecorderApp(inlets)
    gui.show()

    if not inlets:
        gui.status_label.setText("Status: No LSL streams detected!")
        gui.status_label.setStyleSheet(
            "color: #dc3545; font-size: 14px; font-weight: bold;")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
