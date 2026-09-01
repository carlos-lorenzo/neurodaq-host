import sys
import os
import time
import json
import socket
import struct
import numpy as np
from datetime import datetime
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pylsl import StreamInfo, StreamOutlet

# ==============================================================================
# DRACULA COLOR PALETTE & CONFIGURATION CONSTANTS
# ==============================================================================
DRACULA = {
    "bg": "#282a36",
    "current_line": "#44475a",
    "fg": "#f8f8f2",
    "comment": "#6272a4",
    "cyan": "#8be9fd",
    "green": "#50fa7b",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "red": "#ff5555",
    "yellow": "#f1fa8c",
}

DRACULA_CHANNEL_COLORS = [
    "#ff79c6",  # Ch 1: Pink
    "#bd93f9",  # Ch 2: Purple
    "#8be9fd",  # Ch 3: Cyan
    "#50fa7b",  # Ch 4: Green
    "#f1fa8c",  # Ch 5: Yellow
    "#ffb86c",  # Ch 6: Orange
    "#ff5555",  # Ch 7: Red
    "#f8f8f2",  # Ch 8: Bright White
]

UDP_IP = "0.0.0.0"
UDP_PORT = 3333
WIRE_PACKET_FORMAT = f"< I I {('3B x 8i 4x Q' * 25)} B"
MAGIC_HEADER = 0x21474545
N_CHANNELS = 8
N_SAMPLES_PER_PACKET = 25
VREF = 4.5  # ADS1299 Internal Reference Voltage

SAMPLE_RATE_MAP = {
    0: 16000.0,
    1: 8000.0,
    2: 4000.0,
    3: 2000.0,
    4: 1000.0,
    5: 500.0,
    6: 250.0,
}

GAIN_MAP = {0: 1.0, 1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 12.0, 6: 24.0}
LEADOFF_CURRENT_MAP = {0: 6e-9, 1: 24e-9, 2: 6e-6, 3: 24e-6}
LEADOFF_FREQ_MAP = {0: 0.0, 1: 7.8, 2: 31.2, 3: None}

# ==============================================================================
# 10-20 ELECTRODE COORDINATES & BRAIN REGION DEFINITIONS
# ==============================================================================
ELECTRODE_1020_POS = {
    "Fp1": (-0.35, 0.75, "Frontal"),
    "Fp2": (0.35, 0.75, "Frontal"),
    "F7": (-0.75, 0.45, "Frontal"),
    "F3": (-0.35, 0.40, "Frontal"),
    "Fz": (0.00, 0.40, "Frontal"),
    "F4": (0.35, 0.40, "Frontal"),
    "F8": (0.75, 0.45, "Frontal"),
    "T3": (-0.85, 0.00, "Temporal"),
    "C3": (-0.40, 0.00, "Central"),
    "Cz": (0.00, 0.00, "Central"),
    "C4": (0.40, 0.00, "Central"),
    "T4": (0.85, 0.00, "Temporal"),
    "T5": (-0.75, -0.45, "Temporal"),
    "P3": (-0.35, -0.40, "Parietal"),
    "Pz": (0.00, -0.40, "Parietal"),
    "P4": (0.35, -0.40, "Parietal"),
    "T6": (0.75, -0.45, "Temporal"),
    "O1": (-0.35, -0.75, "Occipital"),
    "O2": (0.35, -0.75, "Occipital"),
}

DEFAULT_ELECTRODE_MAPPING = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]

EEG_BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4.0),
    "Theta (4-8 Hz)": (4.0, 8.0),
    "Alpha (8-13 Hz)": (8.0, 13.0),
    "Beta (13-30 Hz)": (13.0, 30.0),
    "Gamma (30-50 Hz)": (30.0, 50.0),
}


# ==============================================================================
# CUSTOM PYQTGRAPH INTERACTION UTILITIES
# ==============================================================================
class CustomAxisItem(pg.AxisItem):
    def wheelEvent(self, ev):
        vb = self.linkedView()
        if vb is not None:
            axis_idx = 0 if self.orientation in ("bottom", "top") else 1
            vb.wheelEvent(ev, axis=axis_idx)
            ev.accept()


class CustomViewBox(pg.ViewBox):
    def wheelEvent(self, ev, axis=None):
        if axis is not None:
            super().wheelEvent(ev, axis=axis)
            return
        pos = ev.pos()
        rect = self.rect()
        rel_x = pos.x() / rect.width() if rect.width() > 0 else 0.5
        rel_y = pos.y() / rect.height() if rect.height() > 0 else 0.5

        if (1.0 - rel_y) < rel_x:
            super().wheelEvent(ev, axis=0)
        else:
            super().wheelEvent(ev, axis=1)


# ==============================================================================
# DSP FILTERING UTILITIES
# ==============================================================================
def apply_signal_filters(
    data,
    fs,
    dc_remove,
    bp_enabled,
    bp_low,
    bp_high,
    bp_order,
    notch_enabled,
    notch_freq,
    notch_q,
):
    """Applies DC Offset Removal (Baseline Mean Subtraction), IIR Notch, and Bandpass Filters."""
    if data.size == 0 or data.shape[0] < 15 or fs <= 0:
        return data

    nyq = fs / 2.0
    filtered = data.copy()

    # Subtract per-channel baseline mean (removes 0 Hz DC offset without altering AC spectrum)
    if dc_remove:
        filtered = filtered - np.mean(filtered, axis=0)

    if notch_enabled and 0 < notch_freq < nyq:
        try:
            b, a = iirnotch(notch_freq, notch_q, fs=fs)
            filtered = filtfilt(b, a, filtered, axis=0)
        except Exception:
            pass

    if bp_enabled and 0 < bp_low < bp_high < nyq:
        try:
            sos = butter(
                bp_order, [bp_low, bp_high], btype="bandpass", fs=fs, output="sos"
            )
            filtered = sosfiltfilt(sos, filtered, axis=0)
        except Exception:
            pass

    return filtered


def compute_band_power(data_column, fs, band_limits):
    """Calculates absolute band power within specified frequency range (f_low, f_high)."""
    n = len(data_column)
    if n < 16 or fs <= 0:
        return 0.0

    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(data_column)) / n
    psd = fft_vals**2

    f_low, f_high = band_limits
    idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
    return float(np.sum(psd[idx_band]))


# ==============================================================================
# TCP CLIENT WORKER THREAD
# ==============================================================================
class TCPControlWorker(QtCore.QThread):
    response_received = QtCore.pyqtSignal(dict)
    connection_status = QtCore.pyqtSignal(bool, str)

    def __init__(self, host="127.0.0.1", port=3333):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self.running = True
        self.req_id = 1
        self.cmd_queue = []
        self.lock = QtCore.QMutex()

    def connect_to_device(self, host, port):
        self.host = host
        self.port = port
        self.disconnect_device()

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.connection_status.emit(
                True, f"Connected to {self.host}:{self.port}"
            )
        except Exception as e:
            self.connection_status.emit(False, f"Connection Failed: {e}")
            self.sock = None

    def disconnect_device(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self.connection_status.emit(False, "Disconnected")

    def send_command(self, cmd, params=None):
        self.lock.lock()
        self.cmd_queue.append((cmd, params))
        self.lock.unlock()

    def run(self):
        buf = ""
        while self.running:
            if self.sock is None:
                self.msleep(100)
                continue

            cmd_item = None
            self.lock.lock()
            if self.cmd_queue:
                cmd_item = self.cmd_queue.pop(0)
            self.lock.unlock()

            if cmd_item:
                cmd, params = cmd_item
                request = {"id": self.req_id, "cmd": cmd}
                if params:
                    request["params"] = params
                self.req_id += 1

                try:
                    payload = json.dumps(request) + "\n"
                    self.sock.sendall(payload.encode("utf-8"))

                    buf = ""
                    while self.running:
                        char = self.sock.recv(1).decode("utf-8")
                        if not char:
                            raise ConnectionError("Connection lost")
                        buf += char
                        if char == "\n":
                            break

                    resp = json.loads(buf)
                    self.response_received.emit(resp)
                except Exception as e:
                    self.connection_status.emit(
                        False, f"TCP Error: {str(e)}"
                    )
                    self.sock = None

            self.msleep(20)


# ==============================================================================
# UDP RECEIVER & IMPEDANCE WORKER THREAD
# ==============================================================================
class UDPDataReceiver(QtCore.QThread):
    data_packet_received = QtCore.pyqtSignal(np.ndarray, list, list)

    def __init__(self, sample_rate=250.0):
        super().__init__()
        self.running = True
        self.sample_rate = sample_rate
        self.gains = np.full(N_CHANNELS, 24.0)

        self.loff_enabled = False
        self.loff_freq_val = 31.2
        self.loff_current_val = 6e-9

        self.buffer_size = int(self.sample_rate)
        self.raw_uV_buffers = [
            np.zeros(self.buffer_size) for _ in range(N_CHANNELS)
        ]

        self.impedances = np.zeros(N_CHANNELS)
        self.stat1_flags = [False] * N_CHANNELS
        self.stat2_flags = [False] * N_CHANNELS

        self.lsl_outlet = None
        self.setup_lsl()

    def setup_lsl(self):
        info = StreamInfo(
            name="NeuroDAQ EEG",
            type="EEG",
            channel_count=N_CHANNELS,
            nominal_srate=self.sample_rate,
            channel_format="float32",
            source_id="neurodaq_eeg_stream",
        )
        chns = info.desc().append_child("channels")
        for i in range(N_CHANNELS):
            ch = chns.append_child("channel")
            ch.append_child_value("label", f"Ch{i+1}")
            ch.append_child_value("unit", "microvolts")

        self.lsl_outlet = StreamOutlet(info)

    def update_sample_rate(self, new_srate):
        self.sample_rate = new_srate
        self.buffer_size = max(16, int(self.sample_rate))
        self.raw_uV_buffers = [
            np.zeros(self.buffer_size) for _ in range(N_CHANNELS)
        ]
        self.setup_lsl()

    def update_gain(self, ch_idx, gain_val):
        self.gains[ch_idx] = gain_val

    def update_leadoff_config(self, enabled, freq_val, current_val):
        self.loff_enabled = enabled
        self.loff_freq_val = freq_val
        self.loff_current_val = current_val

    def calculate_impedance(self, ch_idx):
        data = self.raw_uV_buffers[ch_idx]
        if not self.loff_enabled or self.loff_freq_val <= 0 or data.size < 32:
            return 0.0

        freqs = np.fft.rfftfreq(len(data), 1.0 / self.sample_rate)
        fft_vals = np.abs(np.fft.rfft(data)) / len(data)

        target_bin = np.argmin(np.abs(freqs - self.loff_freq_val))
        v_peak_uV = fft_vals[target_bin] * 2.0
        v_rms_volts = (v_peak_uV * 1e-6) / np.sqrt(2)

        i_rms = self.loff_current_val / np.sqrt(2)
        if i_rms <= 0:
            return 0.0

        z_ohms = v_rms_volts / i_rms
        return z_ohms / 1000.0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((UDP_IP, UDP_PORT))
        sock.settimeout(0.5)

        expected_size = struct.calcsize(WIRE_PACKET_FORMAT)

        while self.running:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except Exception:
                break

            if len(data) != expected_size:
                continue

            unpacked = struct.unpack(WIRE_PACKET_FORMAT, data)
            if unpacked[0] != MAGIC_HEADER:
                continue

            packet_uV = np.zeros((N_SAMPLES_PER_PACKET, N_CHANNELS))
            sample_payload = unpacked[2:-1]

            last_stat1 = 0
            last_stat2 = 0

            for s_idx in range(N_SAMPLES_PER_PACKET):
                base_i = s_idx * 12
                b1 = sample_payload[base_i + 1]
                b2 = sample_payload[base_i + 2]

                last_stat1 = b1
                last_stat2 = b2

                raw_channels = sample_payload[base_i + 3: base_i + 11]

                for ch in range(N_CHANNELS):
                    scale_uV = (
                        (2.0 * VREF / self.gains[ch])
                        / (2**24 - 1)
                        * 1_000_000.0
                    )
                    packet_uV[s_idx, ch] = raw_channels[ch] * scale_uV

            for ch in range(N_CHANNELS):
                self.raw_uV_buffers[ch] = np.roll(
                    self.raw_uV_buffers[ch], -N_SAMPLES_PER_PACKET
                )
                self.raw_uV_buffers[ch][-N_SAMPLES_PER_PACKET:] = packet_uV[
                    :, ch
                ]
                self.impedances[ch] = self.calculate_impedance(ch)

                self.stat1_flags[ch] = bool((last_stat1 >> ch) & 0x01)
                self.stat2_flags[ch] = bool((last_stat2 >> ch) & 0x01)

            for sample in packet_uV:
                self.lsl_outlet.push_sample(sample.tolist())

            self.data_packet_received.emit(
                packet_uV,
                list(self.stat1_flags),
                list(self.stat2_flags),
            )

        sock.close()


# ==============================================================================
# MAIN APPLICATION GUI (DRACULA THEME WITH 10-20 BRAIN MAPPING)
# ==============================================================================
class NeuroDAQGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.sample_rate = 250.0
        self.gains = [24.0] * N_CHANNELS
        self.powerdowns = [0] * N_CHANNELS
        self.visible_channels = [True] * N_CHANNELS
        self.electrode_mappings = list(DEFAULT_ELECTRODE_MAPPING)

        self.plot_buffer_len = int(self.sample_rate * 3.0)
        self.data_buffer = np.zeros((self.plot_buffer_len, N_CHANNELS))

        # DSP Filter Parameters
        self.dc_remove_enabled = True
        self.bp_enabled = True
        self.bp_low = 1.0
        self.bp_high = 40.0
        self.bp_order = 4
        self.notch_enabled = True
        self.notch_freq = 50.0
        self.notch_q = 30.0

        self.tcp_worker = TCPControlWorker()
        self.tcp_worker.connection_status.connect(
            self.on_tcp_connection_status
        )
        self.tcp_worker.response_received.connect(self.on_tcp_response)
        self.tcp_worker.start()

        self.udp_worker = UDPDataReceiver(sample_rate=self.sample_rate)
        self.udp_worker.data_packet_received.connect(self.on_udp_data_received)
        self.udp_worker.start()

        self.init_ui()
        self.enforce_nyquist_limits()

    def init_ui(self):
        self.setWindowTitle(
            "NeuroDAQ Master Station — Dracula Control Suite & 10-20 Topo Map"
        )
        self.resize(1450, 950)

        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background-color: {DRACULA['bg']};
                color: {DRACULA['fg']};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {DRACULA['purple']};
            }}
            QPushButton {{
                background-color: {DRACULA['current_line']};
                color: {DRACULA['fg']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['comment']};
                border-color: {DRACULA['cyan']};
            }}
            QTableWidget, QTableView {{
                background-color: {DRACULA['bg']};
                gridline-color: {DRACULA['current_line']};
                color: {DRACULA['fg']};
                border: 1px solid {DRACULA['current_line']};
            }}
            QHeaderView::section {{
                background-color: {DRACULA['current_line']};
                color: {DRACULA['cyan']};
                font-weight: bold;
                border: 1px solid {DRACULA['bg']};
                padding: 4px;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {DRACULA['current_line']};
                color: {DRACULA['fg']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 3px;
                padding: 4px;
            }}
            QTabWidget::pane {{
                border: 1px solid {DRACULA['current_line']};
            }}
            QTabBar::tab {{
                background: {DRACULA['bg']};
                color: {DRACULA['comment']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: {DRACULA['current_line']};
                color: {DRACULA['pink']};
                font-weight: bold;
                border-bottom: 2px solid {DRACULA['pink']};
            }}
        """
        )

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        top_layout = QtWidgets.QVBoxLayout(main_widget)

        # ----------------------------------------------------------------------
        # TOP BAR: TCP Connection & Master Controls
        # ----------------------------------------------------------------------
        bar_layout = QtWidgets.QHBoxLayout()

        self.txt_ip = QtWidgets.QLineEdit("127.0.0.1")
        self.txt_ip.setFixedWidth(110)
        self.txt_port = QtWidgets.QLineEdit("3333")
        self.txt_port.setFixedWidth(60)

        self.btn_connect = QtWidgets.QPushButton("Connect TCP")
        self.btn_connect.clicked.connect(self.toggle_tcp_connection)

        self.lbl_tcp_status = QtWidgets.QLabel("TCP: Disconnected")
        self.lbl_tcp_status.setStyleSheet(
            f"color: {DRACULA['red']}; font-weight: bold; margin-right: 15px;"
        )

        self.btn_start = QtWidgets.QPushButton("START")
        self.btn_start.setStyleSheet(
            f"background-color: {DRACULA['green']}; color: #1e1f29;"
        )
        self.btn_start.clicked.connect(lambda: self.send_cmd("start"))

        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setStyleSheet(
            f"background-color: {DRACULA['red']}; color: {DRACULA['fg']};"
        )
        self.btn_stop.clicked.connect(lambda: self.send_cmd("stop"))

        self.btn_reset = QtWidgets.QPushButton("RESET")
        self.btn_reset.clicked.connect(lambda: self.send_cmd("reset"))

        self.lbl_srate = QtWidgets.QLabel(
            f"Sampling Rate: {self.sample_rate} Hz")
        self.lbl_srate.setStyleSheet(
            f"color: {DRACULA['cyan']}; font-weight: bold;"
        )

        bar_layout.addWidget(QtWidgets.QLabel("Host:"))
        bar_layout.addWidget(self.txt_ip)
        bar_layout.addWidget(QtWidgets.QLabel("Port:"))
        bar_layout.addWidget(self.txt_port)
        bar_layout.addWidget(self.btn_connect)
        bar_layout.addWidget(self.lbl_tcp_status)
        bar_layout.addWidget(self.btn_start)
        bar_layout.addWidget(self.btn_stop)
        bar_layout.addWidget(self.btn_reset)
        bar_layout.addStretch()
        bar_layout.addWidget(self.lbl_srate)

        top_layout.addLayout(bar_layout)

        # ----------------------------------------------------------------------
        # TABBED NAVIGATION
        # ----------------------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        top_layout.addWidget(self.tabs)

        # ======================================================================
        # TAB 1: Oscilloscope & FFT Spectrum
        # ======================================================================
        tab_graphs = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(tab_graphs)

        # DSP Filter Panel with DC Offset Removal Checkbox
        filter_group = QtWidgets.QGroupBox(
            "DSP Signal Conditioning & Dynamic Notch Configuration"
        )
        flt_grid = QtWidgets.QGridLayout(filter_group)

        # DC Offset Removal Option
        self.chk_dc_remove = QtWidgets.QCheckBox(
            "Remove DC Offset (Mean Subtraction)")
        self.chk_dc_remove.setChecked(True)
        self.chk_dc_remove.setToolTip(
            "Subtracts signal mean to remove baseline offset while preserving all AC spectrum frequencies."
        )
        self.chk_dc_remove.toggled.connect(self.on_filter_changed)

        self.chk_bp = QtWidgets.QCheckBox("Enable Bandpass")
        self.chk_bp.setChecked(True)
        self.chk_bp.toggled.connect(self.on_filter_changed)

        self.spn_bp_low = QtWidgets.QDoubleSpinBox()
        self.spn_bp_low.setPrefix("Low: ")
        self.spn_bp_low.setSuffix(" Hz")
        self.spn_bp_low.setValue(1.0)
        self.spn_bp_low.valueChanged.connect(self.on_filter_changed)

        self.spn_bp_high = QtWidgets.QDoubleSpinBox()
        self.spn_bp_high.setPrefix("High: ")
        self.spn_bp_high.setSuffix(" Hz")
        self.spn_bp_high.setValue(40.0)
        self.spn_bp_high.valueChanged.connect(self.on_filter_changed)

        self.chk_notch = QtWidgets.QCheckBox("Enable Notch")
        self.chk_notch.setChecked(True)
        self.chk_notch.toggled.connect(self.on_filter_changed)

        self.spn_notch_freq = QtWidgets.QDoubleSpinBox()
        self.spn_notch_freq.setPrefix("Notch: ")
        self.spn_notch_freq.setSuffix(" Hz")
        self.spn_notch_freq.setValue(50.0)
        self.spn_notch_freq.valueChanged.connect(self.on_filter_changed)

        self.spn_notch_q = QtWidgets.QDoubleSpinBox()
        self.spn_notch_q.setPrefix("Q Factor: ")
        self.spn_notch_q.setRange(1.0, 100.0)
        self.spn_notch_q.setValue(30.0)
        self.spn_notch_q.setSingleStep(1.0)
        self.spn_notch_q.valueChanged.connect(self.on_filter_changed)

        self.chk_show_filtered = QtWidgets.QCheckBox("View Filtered Signals")
        self.chk_show_filtered.setChecked(True)

        flt_grid.addWidget(self.chk_dc_remove, 0, 0, 1, 2)
        flt_grid.addWidget(self.chk_bp, 0, 2)
        flt_grid.addWidget(self.spn_bp_low, 0, 3)
        flt_grid.addWidget(self.spn_bp_high, 0, 4)
        flt_grid.addWidget(self.chk_notch, 0, 5)
        flt_grid.addWidget(self.spn_notch_freq, 0, 6)
        flt_grid.addWidget(self.spn_notch_q, 0, 7)
        flt_grid.addWidget(self.chk_show_filtered, 0, 8)

        graph_layout.addWidget(filter_group)

        # Plot Splitter
        graph_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        axis_x1 = CustomAxisItem("bottom")
        axis_y1 = CustomAxisItem("left")
        vb1 = CustomViewBox()
        self.plot_time = pg.PlotWidget(
            viewBox=vb1,
            axisItems={"bottom": axis_x1, "left": axis_y1},
            title="Time-Domain Waveforms (uV)",
        )
        self.plot_time.setBackground(DRACULA["bg"])
        self.plot_time.showGrid(x=True, y=True, alpha=0.25)
        self.plot_time.setLabel("left", "Voltage Offset (uV)")
        self.plot_time.setLabel("bottom", "Samples")

        axis_x2 = CustomAxisItem("bottom")
        axis_y2 = CustomAxisItem("left")
        vb2 = CustomViewBox()
        self.plot_fft = pg.PlotWidget(
            viewBox=vb2,
            axisItems={"bottom": axis_x2, "left": axis_y2},
            title="Frequency Spectrum / Power Density (FFT)",
        )
        self.plot_fft.setBackground(DRACULA["bg"])
        self.plot_fft.showGrid(x=True, y=True, alpha=0.25)
        self.plot_fft.setLabel("left", "Amplitude (uV)")
        self.plot_fft.setLabel("bottom", "Frequency (Hz)")

        graph_splitter.addWidget(self.plot_time)
        graph_splitter.addWidget(self.plot_fft)
        graph_layout.addWidget(graph_splitter)

        self.time_curves = []
        self.fft_curves = []
        for i in range(N_CHANNELS):
            color_hex = DRACULA_CHANNEL_COLORS[i % len(DRACULA_CHANNEL_COLORS)]
            c1 = self.plot_time.plot(pen=pg.mkPen(color_hex, width=1.5))
            c2 = self.plot_fft.plot(pen=pg.mkPen(color_hex, width=1.5))
            self.time_curves.append(c1)
            self.fft_curves.append(c2)

        self.tabs.addTab(tab_graphs, "Oscilloscope & FFT Spectrum")

        # ======================================================================
        # TAB 2: 10-20 Brain Activity Mapping & Frequency Band Visualizer
        # ======================================================================
        tab_topo = QtWidgets.QWidget()
        topo_layout = QtWidgets.QHBoxLayout(tab_topo)

        # Left Column: 10-20 Scalp Topo Map Plot
        topo_box = QtWidgets.QGroupBox(
            "10-20 System Topographic Head Map"
        )
        topo_vbox = QtWidgets.QVBoxLayout(topo_box)

        self.plot_topo = pg.PlotWidget()
        self.plot_topo.setBackground(DRACULA["bg"])
        self.plot_topo.setAspectLocked(True)
        self.plot_topo.hideAxis("left")
        self.plot_topo.hideAxis("bottom")

        # Draw Scalp Boundary Ellipse & Features (Nose, Ears)
        angles = np.linspace(0, 2 * np.pi, 200)
        scalp_x = np.cos(angles)
        scalp_y = np.sin(angles)
        self.plot_topo.plot(
            scalp_x, scalp_y, pen=pg.mkPen(DRACULA["comment"], width=2)
        )

        # Nose
        nose_x = [-0.15, 0.0, 0.15]
        nose_y = [0.98, 1.15, 0.98]
        self.plot_topo.plot(
            nose_x, nose_y, pen=pg.mkPen(DRACULA["comment"], width=2)
        )

        # Left / Right Ears
        ear_l_x = [-1.0, -1.08, -1.08, -1.0]
        ear_l_y = [0.2, 0.1, -0.1, -0.2]
        ear_r_x = [1.0, 1.08, 1.08, 1.0]
        ear_r_y = [0.2, 0.1, -0.1, -0.2]
        self.plot_topo.plot(
            ear_l_x, ear_l_y, pen=pg.mkPen(DRACULA["comment"], width=2)
        )
        self.plot_topo.plot(
            ear_r_x, ear_r_y, pen=pg.mkPen(DRACULA["comment"], width=2)
        )

        # Electrode Scatter Points & Text Labels
        self.scatter_nodes = pg.ScatterPlotItem(size=24, pen=pg.mkPen(None))
        self.plot_topo.addItem(self.scatter_nodes)
        self.topo_text_items = []

        topo_vbox.addWidget(self.plot_topo)
        topo_layout.addWidget(topo_box, stretch=3)

        # Right Column: Controls, Region Decomposition, and Band Power Charts
        analysis_box = QtWidgets.QGroupBox("Brain Activity & Spectral Power")
        analysis_vbox = QtWidgets.QVBoxLayout(analysis_box)

        # Selector for Band Focus
        band_sel_lay = QtWidgets.QHBoxLayout()
        band_sel_lay.addWidget(QtWidgets.QLabel("Focused EEG Band:"))
        self.cmb_band_select = QtWidgets.QComboBox()
        for b_name in EEG_BANDS.keys():
            self.cmb_band_select.addItem(b_name)
        self.cmb_band_select.setCurrentText("Alpha (8-13 Hz)")
        band_sel_lay.addWidget(self.cmb_band_select)
        band_sel_lay.addStretch()
        analysis_vbox.addLayout(band_sel_lay)

        # Brain Region Activity Bar Chart
        self.plot_region_bar = pg.PlotWidget(
            title="Activity Power by Anatomical Brain Region"
        )
        self.plot_region_bar.setBackground(DRACULA["bg"])
        self.plot_region_bar.setLabel("left", "Power (uV²)")
        self.plot_region_bar.setLabel("bottom", "Region")
        self.region_bars = pg.BarGraphItem(
            x=[1, 2, 3, 4, 5], height=[0] * 5, width=0.6, brush=DRACULA["purple"]
        )
        self.plot_region_bar.addItem(self.region_bars)

        ticks = [
            (1, "Frontal"),
            (2, "Central"),
            (3, "Parietal"),
            (4, "Occipital"),
            (5, "Temporal"),
        ]
        self.plot_region_bar.getAxis("bottom").setTicks([ticks])

        # Frequency Band Power Comparison Bar Chart
        self.plot_band_bar = pg.PlotWidget(title="Global Spectrum Band Power")
        self.plot_band_bar.setBackground(DRACULA["bg"])
        self.plot_band_bar.setLabel("left", "Power (uV²)")
        self.plot_band_bar.setLabel("bottom", "EEG Frequency Band")
        self.band_bars = pg.BarGraphItem(
            x=[1, 2, 3, 4, 5], height=[0] * 5, width=0.6, brush=DRACULA["cyan"]
        )
        self.plot_band_bar.addItem(self.band_bars)

        band_ticks = [(1, "Delta"), (2, "Theta"),
                      (3, "Alpha"), (4, "Beta"), (5, "Gamma")]
        self.plot_band_bar.getAxis("bottom").setTicks([band_ticks])

        analysis_vbox.addWidget(self.plot_region_bar)
        analysis_vbox.addWidget(self.plot_band_bar)
        topo_layout.addWidget(analysis_box, stretch=4)

        self.tabs.addTab(tab_topo, "10-20 Brain Map & Band Power")

        # ======================================================================
        # TAB 3: Hardware Configuration & 10-20 Mapping Setup
        # ======================================================================
        tab_config = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(tab_config)

        # Global Config Group
        grp_global = QtWidgets.QGroupBox("Global Hardware Settings")
        g_lay = QtWidgets.QHBoxLayout(grp_global)

        self.cmb_srate = QtWidgets.QComboBox()
        for k, v in SAMPLE_RATE_MAP.items():
            self.cmb_srate.addItem(f"{v} Hz", k)
        self.cmb_srate.setCurrentIndex(6)

        btn_apply_srate = QtWidgets.QPushButton("Set Sample Rate")
        btn_apply_srate.clicked.connect(self.apply_global_config)

        g_lay.addWidget(QtWidgets.QLabel("Sample Rate:"))
        g_lay.addWidget(self.cmb_srate)
        g_lay.addWidget(btn_apply_srate)
        g_lay.addStretch()
        config_layout.addWidget(grp_global)

        # HARDWARE BIAS DRIVE CONTROL PANEL
        grp_bias = QtWidgets.QGroupBox(
            "BIAS Drive & Common-Mode Noise Rejection")
        bias_vbox = QtWidgets.QVBoxLayout(grp_bias)

        bias_top_lay = QtWidgets.QHBoxLayout()
        self.chk_bias_p = QtWidgets.QCheckBox("Enable BIAS P Drive (bias_p)")
        self.chk_bias_p.setChecked(True)
        self.chk_bias_n = QtWidgets.QCheckBox("Enable BIAS N Drive (bias_n)")
        self.chk_bias_n.setChecked(True)

        btn_bias_apply = QtWidgets.QPushButton("Push BIAS Config")
        btn_bias_apply.setStyleSheet(
            f"background-color: {DRACULA['purple']}; color: #1e1f29;"
        )
        btn_bias_apply.clicked.connect(self.apply_bias_config)

        bias_top_lay.addWidget(self.chk_bias_p)
        bias_top_lay.addWidget(self.chk_bias_n)
        bias_top_lay.addStretch()
        bias_top_lay.addWidget(btn_bias_apply)
        bias_vbox.addLayout(bias_top_lay)

        bias_grid = QtWidgets.QGridLayout()
        self.chk_bias_p_chs = []
        self.chk_bias_n_chs = []

        bias_grid.addWidget(QtWidgets.QLabel("BIAS P Sense Channels:"), 0, 0)
        bias_grid.addWidget(QtWidgets.QLabel("BIAS N Sense Channels:"), 1, 0)

        for ch in range(N_CHANNELS):
            cp = QtWidgets.QCheckBox(f"Ch{ch+1}")
            cp.setChecked(True)
            cn = QtWidgets.QCheckBox(f"Ch{ch+1}")
            cn.setChecked(True)

            self.chk_bias_p_chs.append(cp)
            self.chk_bias_n_chs.append(cn)

            bias_grid.addWidget(cp, 0, ch + 1)
            bias_grid.addWidget(cn, 1, ch + 1)

        bias_vbox.addLayout(bias_grid)
        config_layout.addWidget(grp_bias)

        # Channel & 10-20 Electrode Setup Matrix
        grp_channels = QtWidgets.QGroupBox(
            "Per-Channel Hardware Matrix & 10-20 System Mapping"
        )
        ch_lay = QtWidgets.QVBoxLayout(grp_channels)

        self.tbl_channels = QtWidgets.QTableWidget(N_CHANNELS, 7)
        self.tbl_channels.setHorizontalHeaderLabels(
            [
                "Channel",
                "10-20 Electrode",
                "Visible Graph",
                "Power Down",
                "PGA Gain",
                "Input MUX",
                "Apply Settings",
            ]
        )
        self.tbl_channels.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

        self.ch_1020_combos = []
        self.ch_visibility_chks = []
        self.ch_powerdown_chks = []
        self.ch_gain_combos = []
        self.ch_mux_combos = []

        for ch in range(N_CHANNELS):
            self.tbl_channels.setItem(
                ch, 0, QtWidgets.QTableWidgetItem(f"Channel {ch+1}")
            )

            # 10-20 Electrode Selector Combo Box
            cmb_1020 = QtWidgets.QComboBox()
            for pos_label in ELECTRODE_1020_POS.keys():
                cmb_1020.addItem(pos_label)
            cmb_1020.setCurrentText(DEFAULT_ELECTRODE_MAPPING[ch])
            cmb_1020.currentTextChanged.connect(
                lambda text, c=ch: self.update_electrode_mapping(c, text)
            )
            self.ch_1020_combos.append(cmb_1020)
            self.tbl_channels.setCellWidget(ch, 1, cmb_1020)

            chk_vis = QtWidgets.QCheckBox()
            chk_vis.setChecked(True)
            chk_vis.toggled.connect(
                lambda val, c=ch: self.toggle_ch_visibility(c, val)
            )
            self.ch_visibility_chks.append(chk_vis)
            self.tbl_channels.setCellWidget(ch, 2, chk_vis)

            chk_pd = QtWidgets.QCheckBox("Power Off")
            self.ch_powerdown_chks.append(chk_pd)
            self.tbl_channels.setCellWidget(ch, 3, chk_pd)

            cmb_gain = QtWidgets.QComboBox()
            for g_code, g_val in GAIN_MAP.items():
                cmb_gain.addItem(f"{int(g_val)}x", g_code)
            cmb_gain.setCurrentIndex(6)
            self.ch_gain_combos.append(cmb_gain)
            self.tbl_channels.setCellWidget(ch, 4, cmb_gain)

            cmb_mux = QtWidgets.QComboBox()
            mux_options = [
                "0: Normal",
                "1: Shorted",
                "2: BIAS_MEAS",
                "3: MVDD",
                "4: Temp",
                "5: Test Signal",
                "6: BIAS_DRP",
                "7: BIAS_DRN",
            ]
            for m_idx, m_str in enumerate(mux_options):
                cmb_mux.addItem(m_str, m_idx)
            self.ch_mux_combos.append(cmb_mux)
            self.tbl_channels.setCellWidget(ch, 5, cmb_mux)

            btn_ch_apply = QtWidgets.QPushButton(f"Apply Ch{ch+1}")
            btn_ch_apply.clicked.connect(
                lambda _, c=ch: self.apply_channel_config(c))
            self.tbl_channels.setCellWidget(ch, 6, btn_ch_apply)

        ch_lay.addWidget(self.tbl_channels)
        config_layout.addWidget(grp_channels)

        self.tabs.addTab(tab_config, "Channel & Device Settings")

        # ======================================================================
        # TAB 4: Lead-Off Diagnostics & Debug Dashboard
        # ======================================================================
        tab_debug = QtWidgets.QWidget()
        debug_layout = QtWidgets.QVBoxLayout(tab_debug)

        grp_loff_cfg = QtWidgets.QGroupBox("Hardware Lead-Off Engine Setup")
        loff_grid = QtWidgets.QGridLayout(grp_loff_cfg)

        self.chk_loff_enable = QtWidgets.QCheckBox("Enable Lead-Off Circuit")
        self.chk_loff_enable.setChecked(True)

        self.cmb_loff_freq = QtWidgets.QComboBox()
        self.cmb_loff_freq.addItem("DC Lead-Off", 0)
        self.cmb_loff_freq.addItem("AC: 7.8 Hz", 1)
        self.cmb_loff_freq.addItem("AC: 31.2 Hz", 2)
        self.cmb_loff_freq.addItem("AC: fDR / 4", 3)
        self.cmb_loff_freq.setCurrentIndex(2)

        self.cmb_loff_curr = QtWidgets.QComboBox()
        self.cmb_loff_curr.addItem("6 nA", 0)
        self.cmb_loff_curr.addItem("24 nA", 1)
        self.cmb_loff_curr.addItem("6 uA", 2)
        self.cmb_loff_curr.addItem("24 uA", 3)
        self.cmb_loff_curr.setCurrentIndex(0)

        btn_loff_apply = QtWidgets.QPushButton("Push Lead-Off Config")
        btn_loff_apply.clicked.connect(self.apply_leadoff_config)

        loff_grid.addWidget(self.chk_loff_enable, 0, 0)
        loff_grid.addWidget(QtWidgets.QLabel("Frequency:"), 0, 1)
        loff_grid.addWidget(self.cmb_loff_freq, 0, 2)
        loff_grid.addWidget(QtWidgets.QLabel("Current:"), 0, 3)
        loff_grid.addWidget(self.cmb_loff_curr, 0, 4)
        loff_grid.addWidget(btn_loff_apply, 0, 5)

        debug_layout.addWidget(grp_loff_cfg)

        grp_diag = QtWidgets.QGroupBox(
            "Per-Channel Lead-Off & Impedance Telemetry"
        )
        diag_lay = QtWidgets.QVBoxLayout(grp_diag)

        self.tbl_debug = QtWidgets.QTableWidget(N_CHANNELS, 4)
        self.tbl_debug.setHorizontalHeaderLabels(
            [
                "Channel",
                "STAT1: Pos DC Lead-Off",
                "STAT2: Neg DC Lead-Off",
                "Calculated Impedance Z (kΩ)",
            ]
        )
        self.tbl_debug.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

        for ch in range(N_CHANNELS):
            self.tbl_debug.setItem(
                ch, 0, QtWidgets.QTableWidgetItem(f"Channel {ch+1}")
            )

            lbl_stat1 = QtWidgets.QLabel("CONNECTED")
            lbl_stat1.setAlignment(QtCore.Qt.AlignCenter)
            lbl_stat1.setStyleSheet(
                f"background-color: {DRACULA['green']}; color: #1e1f29; border-radius: 3px; font-weight: bold;"
            )

            lbl_stat2 = QtWidgets.QLabel("CONNECTED")
            lbl_stat2.setAlignment(QtCore.Qt.AlignCenter)
            lbl_stat2.setStyleSheet(
                f"background-color: {DRACULA['green']}; color: #1e1f29; border-radius: 3px; font-weight: bold;"
            )

            lbl_imp = QtWidgets.QLabel("0.0 kΩ")
            lbl_imp.setAlignment(QtCore.Qt.AlignCenter)
            lbl_imp.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {DRACULA['yellow']};"
            )

            self.tbl_debug.setCellWidget(ch, 1, lbl_stat1)
            self.tbl_debug.setCellWidget(ch, 2, lbl_stat2)
            self.tbl_debug.setCellWidget(ch, 3, lbl_imp)

        diag_lay.addWidget(self.tbl_debug)
        debug_layout.addWidget(grp_diag)

        self.tabs.addTab(tab_debug, "Lead-Off Diagnostics & Debug")

        # ----------------------------------------------------------------------
        # GRAPH REFRESH TIMER
        # ----------------------------------------------------------------------
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.redraw_plots)
        self.timer.start(50)

    # ==========================================================================
    # LOGIC & EVENT HANDLERS
    # ==========================================================================
    def toggle_tcp_connection(self):
        if self.tcp_worker.sock is None:
            host = self.txt_ip.text().strip()
            port = int(self.txt_port.text().strip())
            self.tcp_worker.connect_to_device(host, port)
        else:
            self.tcp_worker.disconnect_device()

    def on_tcp_connection_status(self, connected, msg):
        self.lbl_tcp_status.setText(f"TCP: {msg}")
        if connected:
            self.lbl_tcp_status.setStyleSheet(
                f"color: {DRACULA['green']}; font-weight: bold; margin-right: 15px;"
            )
            self.btn_connect.setText("Disconnect TCP")
        else:
            self.lbl_tcp_status.setStyleSheet(
                f"color: {DRACULA['red']}; font-weight: bold; margin-right: 15px;"
            )
            self.btn_connect.setText("Connect TCP")

    def send_cmd(self, cmd, params=None):
        self.tcp_worker.send_command(cmd, params)

    def on_tcp_response(self, resp):
        print(f"[TCP Response] {resp}")

    def update_electrode_mapping(self, ch_idx, pos_label):
        self.electrode_mappings[ch_idx] = pos_label

    def toggle_ch_visibility(self, ch_idx, visible):
        self.visible_channels[ch_idx] = visible
        if not visible or self.powerdowns[ch_idx] == 1:
            self.time_curves[ch_idx].hide()
            self.fft_curves[ch_idx].hide()
        else:
            self.time_curves[ch_idx].show()
            self.fft_curves[ch_idx].show()

    def apply_global_config(self):
        rate_code = self.cmb_srate.currentData()
        self.sample_rate = SAMPLE_RATE_MAP[rate_code]
        self.lbl_srate.setText(f"Sampling Rate: {self.sample_rate} Hz")

        self.plot_buffer_len = int(self.sample_rate * 3.0)
        self.data_buffer = np.zeros((self.plot_buffer_len, N_CHANNELS))
        self.udp_worker.update_sample_rate(self.sample_rate)

        self.enforce_nyquist_limits()
        params = {
            "sample_rate": rate_code,
            "srb1_enabled": True,
            "srb2_enabled": False,
        }
        self.send_cmd("config_global", params)

    def apply_bias_config(self):
        sensp = 0
        sensn = 0
        for i in range(N_CHANNELS):
            if self.chk_bias_p_chs[i].isChecked():
                sensp |= 1 << i
            if self.chk_bias_n_chs[i].isChecked():
                sensn |= 1 << i

        params = {
            "bias_p_enabled": self.chk_bias_p.isChecked(),
            "bias_n_enabled": self.chk_bias_n.isChecked(),
            "bias_sensp": sensp,
            "bias_sensn": sensn,
            "bias_p": self.chk_bias_p.isChecked(),
            "bias_n": self.chk_bias_n.isChecked(),
            "sensp": sensp,
            "sensn": sensn,
        }
        self.send_cmd("config_bias", params)

    def apply_channel_config(self, ch_idx):
        power_down = 1 if self.ch_powerdown_chks[ch_idx].isChecked() else 0
        gain_code = self.ch_gain_combos[ch_idx].currentData()
        mux_code = self.ch_mux_combos[ch_idx].currentData()

        self.powerdowns[ch_idx] = power_down
        self.gains[ch_idx] = GAIN_MAP[gain_code]
        self.udp_worker.update_gain(ch_idx, self.gains[ch_idx])

        if power_down == 1:
            self.time_curves[ch_idx].hide()
            self.fft_curves[ch_idx].hide()
        elif self.visible_channels[ch_idx]:
            self.time_curves[ch_idx].show()
            self.fft_curves[ch_idx].show()

        params = {
            "channel": ch_idx + 1,
            "power_down": power_down,
            "gain": gain_code,
            "mux": mux_code,
        }
        self.send_cmd("config_channel", params)

    def apply_leadoff_config(self):
        enabled = self.chk_loff_enable.isChecked()
        freq_code = self.cmb_loff_freq.currentData()
        curr_code = self.cmb_loff_curr.currentData()

        if freq_code == 3:
            freq_val = self.sample_rate / 4.0
        else:
            freq_val = LEADOFF_FREQ_MAP[freq_code]

        curr_val = LEADOFF_CURRENT_MAP[curr_code]

        self.udp_worker.update_leadoff_config(enabled, freq_val, curr_val)

        params = {
            "enabled": enabled,
            "threshold": 4,
            "current": curr_code,
            "frequency": freq_code,
            "sensp": 255,
            "sensn": 255,
            "flip": 0,
        }
        self.send_cmd("config_leadoff", params)

    def enforce_nyquist_limits(self):
        nyquist = self.sample_rate / 2.0
        max_freq = max(0.1, nyquist - 0.1)
        self.spn_bp_high.setMaximum(max_freq)
        self.spn_bp_low.setMaximum(max_freq - 0.1)
        self.spn_notch_freq.setMaximum(max_freq)

    def on_filter_changed(self):
        self.enforce_nyquist_limits()
        self.dc_remove_enabled = self.chk_dc_remove.isChecked()
        self.bp_enabled = self.chk_bp.isChecked()
        self.bp_low = self.spn_bp_low.value()
        self.bp_high = self.spn_bp_high.value()
        self.notch_enabled = self.chk_notch.isChecked()
        self.notch_freq = self.spn_notch_freq.value()
        self.notch_q = self.spn_notch_q.value()

    def on_udp_data_received(self, packet_uV, stat1_flags, stat2_flags):
        n_samples = packet_uV.shape[0]
        self.data_buffer = np.vstack(
            (self.data_buffer[n_samples:, :], packet_uV)
        )

        for ch in range(N_CHANNELS):
            lbl_s1 = self.tbl_debug.cellWidget(ch, 1)
            if stat1_flags[ch]:
                lbl_s1.setText("LEAD OFF (RAIL+)")
                lbl_s1.setStyleSheet(
                    f"background-color: {DRACULA['red']}; color: {DRACULA['fg']}; border-radius: 3px; font-weight: bold;"
                )
            else:
                lbl_s1.setText("CONNECTED")
                lbl_s1.setStyleSheet(
                    f"background-color: {DRACULA['green']}; color: #1e1f29; border-radius: 3px; font-weight: bold;"
                )

            lbl_s2 = self.tbl_debug.cellWidget(ch, 2)
            if stat2_flags[ch]:
                lbl_s2.setText("LEAD OFF (RAIL-)")
                lbl_s2.setStyleSheet(
                    f"background-color: {DRACULA['red']}; color: {DRACULA['fg']}; border-radius: 3px; font-weight: bold;"
                )
            else:
                lbl_s2.setText("CONNECTED")
                lbl_s2.setStyleSheet(
                    f"background-color: {DRACULA['green']}; color: #1e1f29; border-radius: 3px; font-weight: bold;"
                )

            lbl_imp = self.tbl_debug.cellWidget(ch, 3)
            imp_val = self.udp_worker.impedances[ch]
            lbl_imp.setText(f"{imp_val:.2f} kΩ")

    def redraw_plots(self):
        if self.data_buffer.shape[0] < 30:
            return

        # 1. Apply Filtering (Including optional DC Offset removal)
        if self.chk_show_filtered.isChecked():
            display_data = apply_signal_filters(
                self.data_buffer,
                self.sample_rate,
                self.dc_remove_enabled,
                self.bp_enabled,
                self.bp_low,
                self.bp_high,
                self.bp_order,
                self.notch_enabled,
                self.notch_freq,
                self.notch_q,
            )
        else:
            display_data = self.data_buffer.copy()

        # 2. Update Time Domain Oscilloscope Traces
        offset_step = 100.0
        for ch in range(N_CHANNELS):
            if self.visible_channels[ch] and self.powerdowns[ch] == 0:
                y = display_data[:, ch] + (ch * offset_step)
                self.time_curves[ch].setData(y)

        # 3. Update Frequency Domain FFT Traces
        n_pts = display_data.shape[0]
        freqs = np.fft.rfftfreq(n_pts, 1.0 / self.sample_rate)
        for ch in range(N_CHANNELS):
            if self.visible_channels[ch] and self.powerdowns[ch] == 0:
                fft_vals = (
                    np.abs(np.fft.rfft(display_data[:, ch])) / n_pts
                ) * 2.0
                self.fft_curves[ch].setData(freqs, fft_vals)

        # 4. Update 10-20 Topographic Brain Map & Band Power Analysis
        self.update_brain_map_and_bands(display_data)

    def update_brain_map_and_bands(self, display_data):
        selected_band_name = self.cmb_band_select.currentText()
        selected_band_limits = EEG_BANDS[selected_band_name]

        spot_list = []
        for t_item in self.topo_text_items:
            self.plot_topo.removeItem(t_item)
        self.topo_text_items.clear()

        # Compute Band Powers per Channel
        channel_band_powers = np.zeros(N_CHANNELS)
        for ch in range(N_CHANNELS):
            if self.powerdowns[ch] == 0:
                channel_band_powers[ch] = compute_band_power(
                    display_data[:, ch], self.sample_rate, selected_band_limits
                )

        max_p = max(1e-6, np.max(channel_band_powers))

        # Build Scatter Nodes for Active Mapped 10-20 Electrodes
        for ch in range(N_CHANNELS):
            if self.powerdowns[ch] == 1:
                continue

            pos_label = self.electrode_mappings[ch]
            pos_info = ELECTRODE_1020_POS.get(pos_label, (0.0, 0.0, "Unknown"))
            x_pos, y_pos, _ = pos_info

            norm_val = np.clip(channel_band_powers[ch] / max_p, 0.0, 1.0)
            # Dracula Color Intensity gradient (Purple -> Pink -> Yellow)
            r = int(189 + norm_val * (255 - 189))
            g = int(147 + norm_val * (121 - 147))
            b = int(249 + norm_val * (198 - 249))

            spot_list.append(
                {
                    "pos": (x_pos, y_pos),
                    "brush": pg.mkBrush(r, g, b, 230),
                    "pen": pg.mkPen(DRACULA["fg"], width=1.5),
                }
            )

            # Text Label for Electrode
            txt = pg.TextItem(
                text=f"{pos_label}\n(Ch{ch+1})", color=DRACULA["fg"], anchor=(0.5, 0.5)
            )
            txt.setPos(x_pos, y_pos - 0.12)
            self.plot_topo.addItem(txt)
            self.topo_text_items.append(txt)

        self.scatter_nodes.setData(spot_list)

        # 5. Calculate Activity Power by Brain Region
        region_power = {
            "Frontal": 0.0,
            "Central": 0.0,
            "Parietal": 0.0,
            "Occipital": 0.0,
            "Temporal": 0.0,
        }
        region_counts = {k: 0 for k in region_power.keys()}

        for ch in range(N_CHANNELS):
            if self.powerdowns[ch] == 0:
                pos_label = self.electrode_mappings[ch]
                reg = ELECTRODE_1020_POS.get(pos_label, (0, 0, "Frontal"))[2]
                region_power[reg] += channel_band_powers[ch]
                region_counts[reg] += 1

        reg_heights = []
        for reg_key in ["Frontal", "Central", "Parietal", "Occipital", "Temporal"]:
            cnt = region_counts[reg_key]
            reg_heights.append(
                region_power[reg_key] / cnt if cnt > 0 else 0.0
            )

        self.region_bars.setOpts(height=reg_heights)

        # 6. Calculate Spectrum Power across Frequency Bands
        band_heights = []
        for b_name, b_lims in EEG_BANDS.items():
            tot_b_power = 0.0
            act_cnt = 0
            for ch in range(N_CHANNELS):
                if self.powerdowns[ch] == 0:
                    tot_b_power += compute_band_power(
                        display_data[:, ch], self.sample_rate, b_lims
                    )
                    act_cnt += 1
            band_heights.append(tot_b_power / act_cnt if act_cnt > 0 else 0.0)

        self.band_bars.setOpts(height=band_heights)

    def closeEvent(self, event):
        self.tcp_worker.running = False
        self.udp_worker.running = False
        self.tcp_worker.wait()
        self.udp_worker.wait()
        event.accept()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = NeuroDAQGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
