"""Paradigm tab: cued-trial runner (prep -> cue -> record -> relax).

Ports the mi_collector.py state machine into a TabPlugin. Trial data is
captured from the Hub (raw uV), labelled, and saved as an epoch NPZ
compatible with the legacy Schema-B analysis (X: trials×ch×t, y: labels).
Every transition emits a `paradigm` event into the shared EventBus, so
recordings stay aligned with external P300/CV markers.
"""
import random
import time
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.core.events import Event


class ParadigmTab(TabPlugin):
    id = "paradigm"
    title = "Paradigm"
    tick_always = True  # trial timing is wall-clock; must advance without data

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        # --- left: cue display ---
        left = QtWidgets.QVBoxLayout()
        self.lbl_cue = QtWidgets.QLabel("Idle")
        self.lbl_cue.setAlignment(Qt.AlignCenter)
        self.lbl_cue.setStyleSheet("font-size: 42px; font-weight: bold;")
        self.lbl_cue.setMinimumSize(420, 300)
        left.addWidget(self.lbl_cue)
        self.lbl_prog = QtWidgets.QLabel("No session")
        left.addWidget(self.lbl_prog)
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start session")
        self.btn_start.setObjectName("ok")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QtWidgets.QPushButton("Stop & save")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(lambda: self._finish(natural=False))
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left.addLayout(row)
        lay.addLayout(left, stretch=3)
        # --- right: config ---
        cfg = QtWidgets.QGroupBox("Trial config")
        fl = QtWidgets.QFormLayout(cfg)
        self.txt_classes = QtWidgets.QLineEdit("Left,Right,Rest")
        self.spn_trials = QtWidgets.QSpinBox(minimum=1, maximum=200, value=10)
        self.spn_prep = QtWidgets.QDoubleSpinBox(suffix=" s", minimum=0.2,
                                                 maximum=10, value=1.0, singleStep=0.5)
        self.spn_cue = QtWidgets.QDoubleSpinBox(suffix=" s", minimum=0.2,
                                                maximum=10, value=1.0, singleStep=0.5)
        self.spn_rec = QtWidgets.QDoubleSpinBox(suffix=" s", minimum=0.5,
                                                maximum=20, value=4.0, singleStep=0.5)
        self.spn_relax = QtWidgets.QDoubleSpinBox(suffix=" s", minimum=0.2,
                                                  maximum=20, value=2.0, singleStep=0.5)
        fl.addRow("Classes (csv):", self.txt_classes)
        fl.addRow("Trials/class:", self.spn_trials)
        fl.addRow("Prep:", self.spn_prep)
        fl.addRow("Cue:", self.spn_cue)
        fl.addRow("Record:", self.spn_rec)
        fl.addRow("Relax:", self.spn_relax)
        lay.addWidget(cfg, stretch=1)
        # state
        self._reset()
        self.widget = w
        return w

    def _reset(self):
        self._disconnect()
        self.seq: list[str] = []
        self.idx = 0
        self.state = "IDLE"
        self.t0 = 0.0
        self.trial_buf: list[np.ndarray] = []
        self.X: list[np.ndarray] = []
        self.y: list[str] = []
        self._connected = False
        self._rec_start_total = 0
        self._saw_no_data = False

    def _disconnect(self):
        """Detach the hub feed exactly once — never warns when idle."""
        if getattr(self, "_connected", False):
            try:
                self.ctx.hub.new_samples.disconnect(self._collect)
            except (RuntimeError, TypeError):
                pass
            self._connected = False

    def _emit(self, label: str):
        import pylsl
        self.ctx.bus.push(Event(sample_idx=self.ctx.hub.total,
                                t_lsl=pylsl.local_clock(), t_host=time.time(),
                                kind="paradigm", label=label))

    def _start(self):
        classes = [c.strip() for c in self.txt_classes.text().split(",") if c.strip()]
        if not classes:
            return
        self._reset()
        self.seq = classes * int(self.spn_trials.value())
        random.shuffle(self.seq)
        self.ctx.hub.new_samples.connect(self._collect)
        self._connected = True
        self._to("PREP")
        self.btn_start.setEnabled(False)

    def _to(self, s: str):
        self.state = s
        self.t0 = time.time()
        cur = self.seq[self.idx] if self.idx < len(self.seq) else ""
        if s == "PREP":
            self.lbl_cue.setText("+")
            self._emit(f"prep/{cur}")
        elif s == "CUE":
            self.lbl_cue.setText(cur)
            self._emit(f"cue/{cur}")
        elif s == "RECORD":
            self.trial_buf = []
            self._rec_start_total = self.ctx.hub.total
            self.lbl_cue.setText(f"GO:\n{cur}")
            self._emit(f"record-start/{cur}")
        elif s == "RELAX":
            self.lbl_cue.setText("Relax")
            self._emit(f"record-end/{cur}")
            self._store_trial(cur)
        self.lbl_prog.setText(f"Trial {min(self.idx+1, len(self.seq))}/{len(self.seq)} — {s}")

    def _collect(self, n: int):
        if self.state == "RECORD" and n > 0:
            sig, _, _ = self.ctx.hub.snapshot(
                seconds=n / max(self.ctx.hub.fs, 1) + 0.01)
            if len(sig):
                self.trial_buf.append(sig[-n:].copy())

    def _store_trial(self, label: str):
        want = int(self.ctx.hub.fs * float(self.spn_rec.value()))
        if not self.trial_buf:
            # Signal path delivered nothing (e.g. tab hidden is no longer an
            # issue, but a stalled stream still is): fall back to a direct
            # snapshot so one missed callback can't void the trial.
            sig, _, _ = self.ctx.hub.snapshot(seconds=float(self.spn_rec.value()))
            if len(sig):
                self.trial_buf.append(sig[-want:].copy())
        if not self.trial_buf:
            return
        if self.ctx.hub.total <= self._rec_start_total:
            # Hub never advanced during RECORD: device isn't streaming.
            # Skip the all-zero epoch and remember why.
            self._saw_no_data = True
            return
        data = np.vstack(self.trial_buf)
        if len(data) > want:
            data = data[:want]
        elif len(data) < want:
            data = np.vstack([data, np.zeros((want - len(data), data.shape[1]))])
        self.X.append(data.T)  # (ch, t)
        self.y.append(label)

    def _finish(self, natural: bool = False):
        """End the session. On completion AND on early stop, ask where to
        save so no trials are silently dropped. Cancel keeps the trials so
        the user can retry via Stop & save."""
        self.btn_start.setEnabled(True)
        self._disconnect()
        # Stopping mid-RECORD keeps the partial trial instead of voiding it.
        if self.state == "RECORD":
            cur = self.seq[self.idx] if self.idx < len(self.seq) else ""
            self._store_trial(cur + " (partial)" if cur else "partial")
        self.state = "IDLE"
        if not self.X:
            self.lbl_cue.setText("Idle")
            if self._saw_no_data:
                self.lbl_prog.setText(
                    "No trials recorded — no samples arrived during RECORD. "
                    "Is the device streaming? (Connect → START, watch the "
                    "scope tab for live traces.)")
            elif self.idx == 0:
                self.lbl_prog.setText(
                    "No trials recorded — stopped before the first trial finished.")
            else:
                self.lbl_prog.setText("No trials recorded.")
            return
        status = f"Session {'complete' if natural else 'stopped early'}: " \
                 f"{len(self.y)} trials"
        self.lbl_cue.setText(status)
        from datetime import datetime
        default = f"data/paradigm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.widget, "Save paradigm trials", default, "NPZ (*.npz)")
        if not path:
            self.lbl_prog.setText("Save cancelled — trials kept, press Stop & save to retry")
            return
        try:
            import os
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            Xa = np.stack(self.X)[:, None, :, :].astype(np.float32)  # (tr,1,ch,t)
            ya = np.array(self.y)
            np.savez_compressed(path, X=Xa, y=ya, fs=self.ctx.hub.fs,
                                classes=np.array(sorted(set(self.y))))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "Save failed",
                                            f"Could not save trials:\n{e}\n\n"
                                            "Trials are kept — fix the location and press "
                                            "Stop & save to retry.")
            self.lbl_prog.setText(f"Save failed: {e}")
            return
        self.lbl_cue.setText(f"Saved {len(self.y)} trials")
        self.lbl_prog.setText(path)
        self.X, self.y = [], []

    def on_tick(self):
        # NOTE: unlike display tabs, the paradigm state machine must advance
        # even when its tab isn't visible — otherwise switching tabs stalls
        # the trial timing. Labels still update when the user returns.
        if not self.widget:
            return
        if self.state == "IDLE":
            return
        el = time.time() - self.t0
        if self.state == "PREP" and el >= float(self.spn_prep.value()):
            self._to("CUE")
        elif self.state == "CUE" and el >= float(self.spn_cue.value()):
            self._to("RECORD")
        elif self.state == "RECORD" and el >= float(self.spn_rec.value()):
            self._to("RELAX")
        elif self.state == "RELAX" and el >= float(self.spn_relax.value()):
            self.idx += 1
            if self.idx >= len(self.seq):
                self._finish(natural=True)
            else:
                self._to("PREP")
