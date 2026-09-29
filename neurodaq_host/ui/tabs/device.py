"""Device tab: pure hardware config (no electrode positions — those live in Brain)."""
from PySide6 import QtWidgets
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.core.config import SAMPLE_RATE_MAP, GAIN_MAP

MUX_OPTIONS = ["0 Normal", "1 Shorted", "2 BIAS_MEAS", "3 MVDD",
               "4 Temp", "5 Test", "6 BIAS_DRP", "7 BIAS_DRN"]


class DeviceTab(TabPlugin):
    id = "device"
    title = "Device"

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        # --- global ---
        g = QtWidgets.QGroupBox("Global")
        gl = QtWidgets.QHBoxLayout(g)
        self.cmb_rate = QtWidgets.QComboBox()
        for code, v in SAMPLE_RATE_MAP.items():
            self.cmb_rate.addItem(f"{v:g} Hz", code)
        for i in range(self.cmb_rate.count()):
            if self.cmb_rate.itemData(i) == self.ctx.dev.rate_code:
                self.cmb_rate.setCurrentIndex(i)
        btn_rate = QtWidgets.QPushButton("Apply rate")
        btn_rate.clicked.connect(self._apply_rate)
        self.chk_srb1 = QtWidgets.QCheckBox("SRB1")
        self.chk_srb1.setChecked(True)
        self.chk_srb2 = QtWidgets.QCheckBox("SRB2")
        gl.addWidget(QtWidgets.QLabel("Rate:"))
        gl.addWidget(self.cmb_rate)
        gl.addWidget(btn_rate)
        gl.addWidget(self.chk_srb1)
        gl.addWidget(self.chk_srb2)
        gl.addStretch()
        lay.addWidget(g)

        # --- BIAS drive + sense masks ---
        b = QtWidgets.QGroupBox("BIAS drive & sense")
        bl = QtWidgets.QVBoxLayout(b)
        top = QtWidgets.QHBoxLayout()
        self.chk_bp = QtWidgets.QCheckBox("BIAS P")
        self.chk_bp.setChecked(True)
        self.chk_bn = QtWidgets.QCheckBox("BIAS N")
        self.chk_bn.setChecked(True)
        btn_bias = QtWidgets.QPushButton("Push BIAS")
        btn_bias.clicked.connect(self._apply_bias)
        top.addWidget(self.chk_bp)
        top.addWidget(self.chk_bn)
        top.addStretch()
        top.addWidget(btn_bias)
        bl.addLayout(top)
        sense = QtWidgets.QGridLayout()
        sense.addWidget(QtWidgets.QLabel("Sense P:"), 0, 0)
        sense.addWidget(QtWidgets.QLabel("Sense N:"), 1, 0)
        self._sp, self._sn = [], []
        for ch in range(8):
            p = QtWidgets.QCheckBox(f"{ch+1}")
            p.setChecked(True)
            n = QtWidgets.QCheckBox(f"{ch+1}")
            n.setChecked(True)
            self._sp.append(p)
            self._sn.append(n)
            sense.addWidget(p, 0, ch + 1)
            sense.addWidget(n, 1, ch + 1)
        bl.addLayout(sense)
        lay.addWidget(b)

        # --- per-channel matrix ---
        chg = QtWidgets.QGroupBox("Per-channel matrix")
        cl = QtWidgets.QVBoxLayout(chg)
        self.tbl = QtWidgets.QTableWidget(8, 6)
        self.tbl.setHorizontalHeaderLabels(["Ch", "Visible", "PowerDown", "Gain", "MUX", "Apply"])
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._vis, self._pd, self._gain, self._mux = [], [], [], []
        for ch in range(8):
            cfg = self.ctx.dev.channels[ch]
            self.tbl.setItem(ch, 0, QtWidgets.QTableWidgetItem(f"Ch{ch+1}"))
            vis = QtWidgets.QCheckBox()
            vis.setChecked(cfg.visible)
            vis.toggled.connect(lambda v, c=ch: self._set_visible(c, v))
            self._vis.append(vis)
            self.tbl.setCellWidget(ch, 1, vis)
            pd = QtWidgets.QCheckBox()
            pd.setChecked(bool(cfg.power_down))
            self._pd.append(pd)
            self.tbl.setCellWidget(ch, 2, pd)
            gc = QtWidgets.QComboBox()
            for code, v in GAIN_MAP.items():
                gc.addItem(f"{v:g}x", code)
            for i in range(gc.count()):
                if gc.itemData(i) == cfg.gain_code:
                    gc.setCurrentIndex(i)
                    break
            self._gain.append(gc)
            self.tbl.setCellWidget(ch, 3, gc)
            mx = QtWidgets.QComboBox()
            for i, name in enumerate(MUX_OPTIONS):
                mx.addItem(name, i)
            mx.setCurrentIndex(cfg.mux)
            self._mux.append(mx)
            self.tbl.setCellWidget(ch, 4, mx)
            ab = QtWidgets.QPushButton(f"Apply {ch+1}")
            ab.clicked.connect(lambda _=False, c=ch: self._apply_ch(c))
            self.tbl.setCellWidget(ch, 5, ab)
        cl.addWidget(self.tbl)
        lay.addWidget(chg)

        # --- register poke (debug; ported from tcp_cli.py) ---
        r = QtWidgets.QGroupBox("Register poke (debug)")
        rl = QtWidgets.QHBoxLayout(r)
        self.spn_addr = QtWidgets.QSpinBox(prefix="addr 0x", minimum=0, maximum=255,
                                           displayIntegerBase=16)
        self.spn_val = QtWidgets.QSpinBox(prefix="val 0x", minimum=0, maximum=255,
                                          displayIntegerBase=16)
        btn_read = QtWidgets.QPushButton("Read reg")
        btn_read.clicked.connect(lambda: self.ctx.tcp.send(
            "read_reg", {"address": int(self.spn_addr.value())}))
        btn_write = QtWidgets.QPushButton("Write reg")
        btn_write.clicked.connect(lambda: self.ctx.tcp.send(
            "write_reg", {"address": int(self.spn_addr.value()),
                          "value": int(self.spn_val.value())}))
        btn_dump = QtWidgets.QPushButton("Dump lead-off regs")
        btn_dump.clicked.connect(self._dump_loff)
        btn_standby = QtWidgets.QPushButton("Standby")
        btn_standby.clicked.connect(lambda: self.ctx.tcp.send("standby"))
        btn_wake = QtWidgets.QPushButton("Wakeup")
        btn_wake.clicked.connect(lambda: self.ctx.tcp.send("wakeup"))
        for x in (self.spn_addr, self.spn_val, btn_read, btn_write,
                  btn_dump, btn_standby, btn_wake):
            rl.addWidget(x)
        rl.addStretch()
        lay.addWidget(r)

        self.widget = w
        return w

    def _set_visible(self, ch, v: bool):
        self.ctx.dev.channels[ch].visible = v

    def _sense_mask(self, boxes) -> int:
        m = 0
        for i, b in enumerate(boxes):
            if b.isChecked():
                m |= 1 << i
        return m

    def _apply_rate(self):
        code = self.cmb_rate.currentData()
        self.ctx.dev.rate_code = code
        self.ctx.dev.sample_rate = SAMPLE_RATE_MAP[code]
        self.ctx.hub.set_fs(self.ctx.dev.sample_rate)
        if getattr(self.ctx, "lsl_fwd", None) is not None:
            try:
                self.ctx.lsl_fwd.set_fs(self.ctx.dev.sample_rate)
            except Exception:
                pass
        self.ctx.tcp.send("config_global", {"sample_rate": code,
                                            "srb1_enabled": self.chk_srb1.isChecked(),
                                            "srb2_enabled": self.chk_srb2.isChecked()})

    def _apply_bias(self):
        self.ctx.tcp.send("config_bias", {
            "bias_p_enabled": self.chk_bp.isChecked(),
            "bias_n_enabled": self.chk_bn.isChecked(),
            "sensp": self._sense_mask(self._sp), "sensn": self._sense_mask(self._sn),
            "bias_p": self.chk_bp.isChecked(), "bias_n": self.chk_bn.isChecked()})

    def _apply_ch(self, ch):
        pd = 1 if self._pd[ch].isChecked() else 0
        gc = self._gain[ch].currentData()
        mux = self._mux[ch].currentData()
        cfg = self.ctx.dev.channels[ch]
        cfg.power_down = pd
        cfg.gain = GAIN_MAP[gc]
        cfg.gain_code = gc
        cfg.mux = mux
        if getattr(self.ctx, "udp", None) is not None:
            try:
                self.ctx.udp.set_gain(ch, GAIN_MAP[gc])
            except Exception:
                pass
        self.ctx.tcp.send("config_channel", {"channel": ch + 1, "power_down": pd,
                                             "gain": gc, "mux": mux})

    def _dump_loff(self):
        for addr in (0x04, 0x0F, 0x10, 0x11, 0x17, 0x12, 0x13):
            self.ctx.tcp.send("read_reg", {"address": addr})

    def on_tick(self):
        pass
