"""P300 speller tab: row/column oddball flasher for ERP training sessions.

Calibration flow: a target character is cued, then rows/columns of the
matrix flash in random order (N sequences per target). Every flash onset is
pushed to the shared EventBus (kind='p300', label encodes row/col and
target/non-target) with the Hub sample index, so epochs can be cut exactly.

On save, flash-locked epochs [pre_ms, post_ms] are extracted from the
continuous session buffer and stored as a labeled training set:
  X: (n_flashes, 1, n_ch, n_t) float32, y: (n_flashes,) int (1=target row/col)
plus flash meta (target char, row/col, sample_idx) and fs — ready for an
ERP/P300 classifier (e.g. for keyboard typing).
"""
import random
import time
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.core.events import Event

DEFAULT_GRID = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class P300Tab(TabPlugin):
    id = "p300"
    title = "P300"
    tick_always = True  # flash timing is wall-clock; must run when hidden

    COLS = 6

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        # --- left: matrix + cue ---
        left = QtWidgets.QVBoxLayout()
        self.lbl_target = QtWidgets.QLabel("Target: —")
        self.lbl_target.setAlignment(Qt.AlignCenter)
        self.lbl_target.setStyleSheet("font-size: 28px; font-weight: bold;")
        left.addWidget(self.lbl_target)
        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(6)
        self.cells: list[QtWidgets.QLabel] = []
        for i, ch in enumerate(DEFAULT_GRID):
            lab = QtWidgets.QLabel(ch)
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(64, 64)
            lab.setStyleSheet(self._cell_style(False))
            self.grid.addWidget(lab, i // self.COLS, i % self.COLS)
            self.cells.append(lab)
        left.addLayout(self.grid)
        self.lbl_prog = QtWidgets.QLabel("Idle — configure and press Start")
        self.lbl_prog.setAlignment(Qt.AlignCenter)
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
        from neurodaq_host.ui.widgets.sound import SoundCue
        self.snd = SoundCue()
        cfg = QtWidgets.QGroupBox("P300 config")
        fl = QtWidgets.QFormLayout(cfg)
        self.txt_text = QtWidgets.QLineEdit("HELLO")
        self.txt_text.setToolTip("Calibration text: targets are cued in order")
        self.spn_seqs = QtWidgets.QSpinBox(minimum=1, maximum=30, value=8)
        self.spn_seqs.setToolTip("Sequences per target (1 seq = all 6 rows + 6 cols)")
        self.spn_flash = QtWidgets.QSpinBox(suffix=" ms", minimum=30,
                                            maximum=500, value=100, singleStep=10)
        self.spn_isi = QtWidgets.QSpinBox(suffix=" ms", minimum=0,
                                          maximum=1000, value=75, singleStep=25)
        self.spn_cue = QtWidgets.QDoubleSpinBox(suffix=" s", minimum=0.5,
                                                maximum=10, value=2.0, singleStep=0.5)
        self.spn_pre = QtWidgets.QSpinBox(suffix=" ms", minimum=0,
                                          maximum=500, value=200, singleStep=50)
        self.spn_post = QtWidgets.QSpinBox(suffix=" ms", minimum=100,
                                           maximum=2000, value=800, singleStep=100)
        fl.addRow("Text:", self.txt_text)
        fl.addRow("Sequences/target:", self.spn_seqs)
        fl.addRow("Flash dur:", self.spn_flash)
        fl.addRow("ISI (gap):", self.spn_isi)
        fl.addRow("Cue dur:", self.spn_cue)
        fl.addRow("Epoch pre:", self.spn_pre)
        fl.addRow("Epoch post:", self.spn_post)
        self.chk_sound = QtWidgets.QCheckBox("Sound cues")
        self.chk_sound.setChecked(True)
        self.chk_sound.setToolTip("Blip on target flash, chime on letter change")
        self.chk_sound.toggled.connect(lambda on: setattr(self.snd, "enabled", on))
        fl.addRow("Audio:", self.chk_sound)
        lay.addWidget(cfg, stretch=1)
        self._reset()
        self.widget = w
        return w

    @staticmethod
    def _cell_style(lit: bool) -> str:
        if lit:
            return ("font-size: 30px; font-weight: bold; color: #06281f; "
                    "background-color: #2dd4bf; border-radius: 8px;")
        return ("font-size: 30px; font-weight: bold; color: #e6e9ef; "
                "background-color: #252c38; border: 1px solid #2f3744; "
                "border-radius: 8px;")

    # ---------------- session state ----------------
    def _reset(self):
        self._disconnect()
        self.state = "IDLE"
        self.targets: list[str] = []
        self.ti = 0          # target index
        self.seq = 0         # sequence index within target
        self.flashes: list[tuple[str, int]] = []  # (row/col, idx) queue
        self.fi = 0          # flash index within current sequence set
        self.t0 = 0.0
        self.flash_on_until = 0.0
        self.lit: set[int] = set()
        self.buf: list[np.ndarray] = []  # continuous session buffer
        self.buf_start_total = 0
        self.flash_log: list[dict] = []  # {sample_idx, kind, idx, target:bool, char}
        self._connected = False

    def _disconnect(self):
        if getattr(self, "_connected", False):
            try:
                self.ctx.hub.new_samples.disconnect(self._collect)
            except (RuntimeError, TypeError):
                pass
            self._connected = False

    def _emit(self, label: str, value: float = 0.0):
        import pylsl
        self.ctx.bus.push(Event(sample_idx=self.ctx.hub.total,
                                t_lsl=pylsl.local_clock(), t_host=time.time(),
                                kind="p300", label=label, value=value))

    def _start(self):
        text = [c for c in self.txt_text.text().upper() if c in DEFAULT_GRID]
        if not text:
            QtWidgets.QMessageBox.warning(self.widget, "P300",
                                           "Calibration text has no matrix characters.")
            return
        self._reset()
        self.targets = text
        self.buf_start_total = self.ctx.hub.total
        self.ctx.hub.new_samples.connect(self._collect)
        self._connected = True
        self.btn_start.setEnabled(False)
        self._next_target()

    def _collect(self, n: int):
        if n > 0:
            sig, _, _ = self.ctx.hub.snapshot(
                seconds=n / max(self.ctx.hub.fs, 1) + 0.01)
            if len(sig):
                self.buf.append(sig[-n:].copy())

    def _next_target(self):
        if self.ti >= len(self.targets):
            self._finish(natural=True)
            return
        self.char = self.targets[self.ti]
        self.seq = 0
        self.state = "CUE"
        self.t0 = time.time()
        self.lbl_target.setText(f"👁 FOCUS: {self.char}  ({self.ti+1}/{len(self.targets)})")
        self.lbl_target.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #06281f; "
            "background-color: #fbbf24; border-radius: 10px; padding: 6px;")
        self._emit(f"target/{self.char}")
        self.snd.new_letter()

    def _begin_sequences(self):
        self.state = "FLASH"
        self._new_sequence()
        self._emit(f"flash-start/{self.char}")

    def _new_sequence(self):
        order = [(("row", i)) for i in range(self.COLS)] + \
                [(("col", i)) for i in range(self.COLS)]
        random.shuffle(order)
        self.flashes = order
        self.fi = 0
        self.next_flash_at = time.time()

    def _flash_cells(self, kind: str, idx: int) -> set[int]:
        if kind == "row":
            return {idx * self.COLS + c for c in range(self.COLS)}
        return {r * self.COLS + idx for r in range(self.COLS)}

    def _set_lit(self, cells: set[int]):
        for i in self.lit - cells:
            self.cells[i].setStyleSheet(self._cell_style(False))
        for i in cells - self.lit:
            self.cells[i].setStyleSheet(self._cell_style(True))
        self.lit = set(cells)

    def _target_rc(self) -> tuple[int, int]:
        k = DEFAULT_GRID.index(self.char)
        return k // self.COLS, k % self.COLS

    def on_tick(self):
        if not self.widget:
            return
        if self.state == "IDLE":
            return
        now = time.time()
        if self.state == "CUE":
            self.lbl_prog.setText(f"Cue {self.char} — flashing starts soon")
            if now - self.t0 >= float(self.spn_cue.value()):
                self._begin_sequences()
            return
        if self.state == "FLASH":
            # turn off elapsed flash
            if self.lit and now >= self.flash_on_until:
                self._set_lit(set())
            if self.lit:
                return  # flash still on (ISI counted after off below)
            if not hasattr(self, "next_flash_at"):
                self.next_flash_at = now
            if now >= self.next_flash_at:
                if self.fi >= len(self.flashes):
                    self.seq += 1
                    if self.seq >= int(self.spn_seqs.value()):
                        self.ti += 1
                        self._next_target()
                    else:
                        self._new_sequence()
                    return
                kind, idx = self.flashes[self.fi]
                self.fi += 1
                cells = self._flash_cells(kind, idx)
                tr, tc = self._target_rc()
                is_target = (kind == "row" and idx == tr) or \
                            (kind == "col" and idx == tc)
                self._set_lit(cells)
                flash_ms = float(self.spn_flash.value()) / 1000.0
                isi_ms = float(self.spn_isi.value()) / 1000.0
                self.flash_on_until = now + flash_ms
                self.next_flash_at = now + flash_ms + isi_ms
                tag = "target" if is_target else "nontarget"
                self.flash_log.append({"sample_idx": self.ctx.hub.total,
                                       "kind": kind, "idx": idx,
                                       "target": is_target, "char": self.char})
                if is_target:
                    self.snd.target_flash()
                self._emit(f"flash/{kind}-{idx}/{tag}/{self.char}",
                           value=1.0 if is_target else 0.0)
                total_fl = len(self.flash_log)
                self.lbl_prog.setText(
                    f"{self.char} seq {self.seq+1}/{int(self.spn_seqs.value())} "
                    f"— flash {total_fl}")

    # ---------------- saving ----------------
    def _finish(self, natural: bool = False):
        self.btn_start.setEnabled(True)
        self._disconnect()
        self._set_lit(set())
        self.state = "IDLE"
        self.lbl_target.setStyleSheet("font-size: 28px; font-weight: bold;")
        if not self.flash_log or not self.buf:
            self.lbl_target.setText("Target: —")
            if self.flash_log and not self.buf:
                self.lbl_prog.setText(
                    "No samples arrived during flashing — is the device streaming?")
            else:
                self.lbl_prog.setText("No flashes recorded.")
            return
        status = (f"Session complete: {len(self.flash_log)} flashes" if natural
                  else f"Stopped early: {len(self.flash_log)} flashes")
        self.lbl_target.setText(status)
        from datetime import datetime
        default = f"data/p300_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.widget, "Save P300 training set", default, "NPZ (*.npz)")
        if not path:
            self.lbl_prog.setText("Save cancelled — session kept, press Stop & save to retry")
            return
        try:
            data = np.vstack(self.buf)
            fs = self.ctx.hub.fs
            pre = int(fs * float(self.spn_pre.value()) / 1000.0)
            post = int(fs * float(self.spn_post.value()) / 1000.0)
            Xs, ys, meta = [], [], []
            for f in self.flash_log:
                rel = f["sample_idx"] - self.buf_start_total
                a, b = rel - pre, rel + post
                if a < 0 or b > len(data):
                    continue  # flash outside buffered range (e.g. hub reset)
                Xs.append(data[a:b].T)  # (ch, t)
                ys.append(1 if f["target"] else 0)
                meta.append((f["char"], f["kind"], f["idx"], f["sample_idx"]))
            if not Xs:
                raise RuntimeError("no flash epochs fell inside the recorded buffer")
            Xa = np.stack(Xs)[:, None, :, :].astype(np.float32)
            ya = np.array(ys, dtype=np.int64)
            meta_arr = np.array(meta, dtype=[("char", "U4"), ("kind", "U4"),
                                             ("idx", "i8"), ("sample_idx", "i8")])
            import os
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, X=Xa, y=ya, fs=np.float64(fs),
                                pre_ms=int(self.spn_pre.value()),
                                post_ms=int(self.spn_post.value()),
                                flash_meta=meta_arr,
                                grid=np.array(list(DEFAULT_GRID)))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "Save failed",
                                            f"Could not save P300 set:\n{e}\n\n"
                                            "Session is kept — press Stop & save to retry.")
            self.lbl_prog.setText(f"Save failed: {e}")
            return
        n_t = int(np.sum(ya))
        self.lbl_prog.setText(f"Saved {len(ya)} epochs ({n_t} target) → {path}")
        self.buf, self.flash_log = [], []
