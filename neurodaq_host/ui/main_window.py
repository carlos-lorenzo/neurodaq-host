"""Main window: top bar + pluggable tab host + record + tab manager + tick loop."""
import time
from PySide6 import QtCore, QtWidgets
from neurodaq_host.ui.theme import stylesheet, TOKENS
from neurodaq_host.io.protocol import TCP_PORT_DEFAULT
from neurodaq_host.core import profiles, session as S
from neurodaq_host.ui.tabs.scope import ScopeTab
from neurodaq_host.ui.tabs.spectrum import SpectrumTab
from neurodaq_host.ui.tabs.brain import BrainTab
from neurodaq_host.ui.tabs.device import DeviceTab
from neurodaq_host.ui.tabs.impedance import ImpedanceTab
from neurodaq_host.ui.tabs.events import EventsTab
from neurodaq_host.ui.tabs.paradigm import ParadigmTab
from neurodaq_host.ui.tabs.emg import EMGTab
from neurodaq_host.ui.tabs.p300 import P300Tab
from neurodaq_host.ui.tabs.p300_speller import P300SpellerTab

REGISTRY = {"scope": ScopeTab, "spectrum": SpectrumTab, "brain": BrainTab,
            "device": DeviceTab, "impedance": ImpedanceTab, "events": EventsTab,
            "paradigm": ParadigmTab, "emg": EMGTab, "p300": P300Tab,
            "p300_speller": P300SpellerTab}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("NeuroDAQ Master Station")
        self.resize(1440, 900)
        self.setStyleSheet(stylesheet())
        self.tabs: dict[str, object] = {}
        self._last_tick_total = -1
        self._last_tick_rev = -1
        self._rec: list = []
        self._rec_th: list = []
        self._rec_tl: list = []
        self._rec_on = False
        self._build()
        self._load_tabs()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        ctx.tcp.status.connect(self._on_tcp)
        ctx.tcp.response.connect(self._on_tcp_response)

    def _build(self):
        c = QtWidgets.QWidget()
        self.setCentralWidget(c)
        lay = QtWidgets.QVBoxLayout(c)
        bar = QtWidgets.QHBoxLayout()
        self.ip = QtWidgets.QLineEdit("192.168.1.53")
        self.ip.setFixedWidth(120)
        self.port = QtWidgets.QLineEdit(str(TCP_PORT_DEFAULT))
        self.port.setFixedWidth(70)
        self.btn_conn = QtWidgets.QPushButton("Connect")
        self.btn_conn.clicked.connect(self._toggle_tcp)
        # widest of Connect/Disconnect + padding, so the toggle never shifts the bar
        _fw = self.btn_conn.fontMetrics().horizontalAdvance("Disconnect") + 32
        self.btn_conn.setFixedWidth(_fw)
        # Fixed-width status pills: message text is elided and never resizes
        # the top bar (full text always in the tooltip).
        self.pill_tcp = QtWidgets.QLabel("TCP: idle")
        self.pill_tcp.setObjectName("pill")
        self.pill_tcp.setFixedWidth(260)
        self.pill_tcp.setAlignment(QtCore.Qt.AlignCenter)
        self.btn_start = QtWidgets.QPushButton("START")
        self.btn_start.setObjectName("ok")
        self.btn_start.clicked.connect(lambda: self.ctx.tcp.send("start"))
        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(lambda: self.ctx.tcp.send("stop"))
        self.btn_rec = QtWidgets.QPushButton("● Record")
        self.btn_rec.setObjectName("accent")
        self.btn_rec.clicked.connect(self._toggle_rec)
        self.pill_rec = QtWidgets.QLabel("Idle")
        self.pill_rec.setObjectName("pill")
        self.pill_rec.setFixedWidth(220)
        self.pill_rec.setAlignment(QtCore.Qt.AlignCenter)
        # link telemetry: sequence gaps (lost) + checksum failures only
        self.pill_link = QtWidgets.QLabel("lost 0 · crc 0")
        self.pill_link.setObjectName("muted")
        self.pill_link.setFixedWidth(180)
        self.pill_link.setAlignment(QtCore.Qt.AlignCenter)
        self.pill_link.setToolTip("UDP sequence gaps (lost packets) and "
                                  "checksum failures since connect")
        btn_tabs = QtWidgets.QPushButton("Tabs…")
        btn_tabs.clicked.connect(self._tab_manager)
        self.lbl_fs = QtWidgets.QLabel(f"{self.ctx.hub.fs:g} Hz")
        for x in (QtWidgets.QLabel("Host:"), self.ip, QtWidgets.QLabel("Port:"),
                  self.port, self.btn_conn, self.pill_tcp,
                  self.btn_start, self.btn_stop, self.btn_rec, self.pill_rec, btn_tabs):
            bar.addWidget(x)
        bar.addStretch()
        bar.addWidget(self.pill_link)
        bar.addWidget(self.lbl_fs)
        lay.addLayout(bar)
        self.tabw = QtWidgets.QTabWidget()
        lay.addWidget(self.tabw)
        # bottom TCP log with hide toggle
        logrow = QtWidgets.QHBoxLayout()
        self.chk_log = QtWidgets.QCheckBox("Show TCP log")
        self.chk_log.setChecked(True)
        self.chk_log.toggled.connect(self._toggle_log)
        logrow.addWidget(self.chk_log)
        logrow.addStretch()
        lay.addLayout(logrow)
        self.log = QtWidgets.QTextEdit(readOnly=True, maximumHeight=70)
        lay.addWidget(self.log)

    def _toggle_log(self, on: bool):
        self.log.setVisible(on)

    @staticmethod
    def _set_pill(label: QtWidgets.QLabel, text: str):
        """Set pill text elided to its fixed width; full text -> tooltip."""
        label.setToolTip(text)
        fm = label.fontMetrics()
        label.setText(fm.elidedText(text, QtCore.Qt.ElideRight, label.width() - 8))

    def _load_tabs(self):
        order = sorted(self.ctx.profile.get("tabs", []), key=lambda t: t.get("order", 99))
        for t in order:
            if t.get("enabled") and t["id"] in REGISTRY:
                self._add_tab(t["id"])

    def _save_layout(self):
        prof = self.ctx.profile
        en = set(self.tabs)
        for t in prof.get("tabs", []):
            t["enabled"] = t["id"] in en
        panels = prof.setdefault("panels", {})
        for tid, plug in self.tabs.items():
            try:
                panels[tid] = plug.panel_state()
            except Exception:
                pass
        profiles.save_profile(prof)

    def _add_tab(self, tid):
        if tid in self.tabs:
            return
        plug = REGISTRY[tid](self.ctx)
        widget = plug.build()
        try:
            plug.restore_state(self.ctx.profile.get("panels", {}).get(tid, {}))
        except Exception:
            pass
        self.tabw.addTab(widget, plug.title)
        self.tabs[tid] = plug

    def _tab_manager(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Tabs — enable/disable (saved to profile)")
        lay = QtWidgets.QVBoxLayout(dlg)
        checks = {}
        for tid, cls in REGISTRY.items():
            cb = QtWidgets.QCheckBox(cls.title)
            cb.setChecked(tid in self.tabs)
            if tid == "scope":
                cb.setEnabled(False)  # scope is mandatory
            checks[tid] = cb
            lay.addWidget(cb)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        if dlg.exec():
            for tid, cb in checks.items():
                if tid == "scope":
                    continue
                if cb.isChecked():
                    self._add_tab(tid)
                elif tid in self.tabs:
                    idx = self.tabw.indexOf(self.tabs[tid].widget)
                    if idx >= 0:
                        self.tabw.removeTab(idx)
                    self.tabs.pop(tid).teardown()
            self._save_layout()
            self._append_log("Profile saved.")

    # --- slots ---
    def _append_log(self, msg: str):
        if self.chk_log.isChecked():
            self.log.append(msg)

    def _on_tcp_response(self, resp: dict):
        self._append_log(str(resp))
        # surface result in top pill without spamming the log when hidden
        try:
            ok = bool(resp.get("success", False))
            msg = str(resp.get("message", ""))
            self._set_pill(self.pill_tcp, f"TCP: {'OK' if ok else 'ERR'}: {msg}")
        except Exception:
            pass

    def _toggle_tcp(self):
        if self.ctx.tcp.sock is None:
            try:
                self.ctx.tcp.connect_to(self.ip.text().strip(), int(self.port.text()))
            except ValueError:
                self._append_log("Bad port number")
        else:
            self.ctx.tcp.disconnect()

    def _on_tcp(self, ok, msg):
        self._set_pill(self.pill_tcp, f"TCP: {msg}")
        self.btn_conn.setText("Disconnect" if ok else "Connect")

    def _toggle_rec(self):
        import numpy as np
        from dataclasses import asdict
        if not self._rec_on:
            self._rec, self._rec_th, self._rec_tl = [], [], []
            self.ctx.bus.clear()
            self._rec_on = True
            self.btn_rec.setText("■ Stop & save")
            self.ctx.hub.new_samples.connect(self._collect)
        else:
            self._rec_on = False
            try:
                self.ctx.hub.new_samples.disconnect(self._collect)
            except Exception:
                pass
            self.btn_rec.setText("● Record")
            if not self._rec:
                self._set_pill(self.pill_rec, "No data")
                return
            data = np.vstack(self._rec)
            th = np.concatenate(self._rec_th) if self._rec_th else np.zeros(len(data))
            tl = np.concatenate(self._rec_tl) if self._rec_tl else np.zeros(len(data))
            fn = S.default_filename("neurodaq_session")
            S.save_session(fn, data, self.ctx.hub.fs,
                           [f"Ch{i+1}" for i in range(8)], self.ctx.electrode_mapping,
                           th, tl, self.ctx.bus.all(),
                           filter_config=asdict(self.ctx.filt),
                           device_config={"sample_rate": self.ctx.dev.sample_rate,
                                          "gains": [c.gain for c in self.ctx.dev.channels]})
            self._set_pill(self.pill_rec, f"Saved {fn} ({len(data)})")
            self._append_log(f"Saved {fn} ({len(data)} samples, "
                             f"{len(self.ctx.bus.all())} events)")
            self._rec, self._rec_th, self._rec_tl = [], [], []

    def _collect(self, n):
        sig, th, tl = self.ctx.hub.snapshot(seconds=n / max(self.ctx.hub.fs, 1) + 0.01)
        self._rec.append(sig[-n:].copy())
        self._rec_th.append(th[-n:].copy())
        self._rec_tl.append(tl[-n:].copy())
        self._set_pill(self.pill_rec, f"Rec {sum(len(c) for c in self._rec)}")

    def _tick(self):
        udp = getattr(self.ctx, "udp", None)
        if udp is not None:
            self.pill_link.setText(
                f"lost {udp.lost} · crc {self.ctx.hub.checksum_fails}")
            if udp.lost > 0:
                self.pill_link.setStyleSheet("color: #fbbf24; font-weight: bold;")
        self.lbl_fs.setText(f"{self.ctx.hub.fs:g} Hz")
        # Display tabs only repaint on fresh data or filter edits — re-setting
        # identical curves every 50 ms just burns paint budget and judders.
        fresh = (self.ctx.hub.total != self._last_tick_total
                 or self.ctx.filt.rev != self._last_tick_rev)
        self._last_tick_total = self.ctx.hub.total
        self._last_tick_rev = self.ctx.filt.rev
        for p in self.tabs.values():
            if not fresh and not p.tick_always:
                continue
            try:
                p.on_tick()
            except Exception:
                pass

    def closeEvent(self, e):
        try:
            self._save_layout()
        except Exception:
            pass
        self.ctx.tcp.running = False
        if hasattr(self.ctx, "udp"):
            self.ctx.udp.running = False
        e.accept()
