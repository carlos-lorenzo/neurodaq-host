"""Impedance tab: lead-off engine config + per-channel telemetry."""
from PySide6 import QtWidgets
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.core.config import LEADOFF_THRESHOLD_PCT
from neurodaq_host.dsp.impedance import loop_impedance_kohm
from neurodaq_host.core.config import LEADOFF_CURRENT_MAP, LEADOFF_FREQ_MAP
from neurodaq_host.ui.theme import TOKENS


class ImpedanceTab(TabPlugin):
    id = "impedance"
    title = "Impedance"

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        cfg = QtWidgets.QGroupBox("Lead-off engine")
        gl = QtWidgets.QGridLayout(cfg)
        self.chk_en = QtWidgets.QCheckBox("Enable")
        self.cmb_f = QtWidgets.QComboBox()
        for t in ["DC", "AC 7.8 Hz", "AC 31.2 Hz", "AC fDR/4"]:
            self.cmb_f.addItem(t)
        self.cmb_f.setCurrentIndex(2)
        self.cmb_i = QtWidgets.QComboBox()
        for t in ["6 nA", "24 nA", "6 µA", "24 µA"]:
            self.cmb_i.addItem(t)
        self.cmb_i.setCurrentIndex(3)
        self.cmb_t = QtWidgets.QComboBox()
        for code in range(8):
            self.cmb_t.addItem(f"{LEADOFF_THRESHOLD_PCT[code]:g}%", code)
        self.cmb_t.setCurrentIndex(4)
        btn = QtWidgets.QPushButton("Push config")
        btn.clicked.connect(self._push)
        for i, x in enumerate([self.chk_en, self.cmb_f, self.cmb_i, self.cmb_t, btn]):
            gl.addWidget(x, 0, i)
        lay.addWidget(cfg)
        self.tbl = QtWidgets.QTableWidget(8, 3)
        self.tbl.setHorizontalHeaderLabels(["Ch", "Z (kΩ)", "Tone / flags"])
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for ch in range(8):
            self.tbl.setItem(ch, 0, QtWidgets.QTableWidgetItem(f"Ch{ch+1}"))
            self.tbl.setItem(ch, 1, QtWidgets.QTableWidgetItem("—"))
            self.tbl.setItem(ch, 2, QtWidgets.QTableWidgetItem(""))
        lay.addWidget(self.tbl)
        self.widget = w
        return w

    def _push(self):
        en = self.chk_en.isChecked()
        fc, cc, tc = self.cmb_f.currentIndex(), self.cmb_i.currentIndex(), self.cmb_t.currentData()
        self.ctx.dev.loff_enabled = en
        self.ctx.dev.loff_freq_code = fc
        self.ctx.dev.loff_curr_code = cc
        self.ctx.dev.loff_thresh_code = tc
        self.ctx.tcp.send("config_leadoff", {"enabled": en, "threshold": tc,
                                             "current": cc, "frequency": fc,
                                             "sensp": 255, "sensn": 255, "flip": 0})

    def on_tick(self):
        if not self.widget or not self.widget.isVisible():
            return
        if not self.ctx.dev.loff_enabled:
            return
        sig, _, _ = self.ctx.hub.snapshot(seconds=2.0)
        fs = self.ctx.hub.fs
        fc = self.ctx.dev.loff_freq_code
        f = fs / 4.0 if fc == 3 else LEADOFF_FREQ_MAP[fc]
        cur = LEADOFF_CURRENT_MAP[self.ctx.dev.loff_curr_code]
        if not f:
            return
        for ch in range(8):
            m = loop_impedance_kohm(sig[:, ch], fs, f, cur)
            txt = f"{m['z_kohm']:.2f}" if "LOW_SNR" not in m["flags"] else "low SNR"
            self.tbl.item(ch, 1).setText(txt)
            self.tbl.item(ch, 2).setText(f"{m['tone_uVpk']:.0f} µVpk {' '.join(m['flags'])}")
