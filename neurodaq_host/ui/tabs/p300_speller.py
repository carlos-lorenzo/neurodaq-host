"""P300 live speller tab: free-typing with a trained classifier.

Same 6x6 matrix UI as the P300 calibration tab, but reversed: the user
focuses the letter they want, rows/columns flash for N sequences, and the
trained model (analysis/models/p300/classifier.joblib bundle) scores each
flash epoch. Row/col scores are averaged over sequences; argmax row + argmax
col = decoded letter, appended to the typed text.

Bundle format (saved by analysis/p300_training.ipynb):
  {'pipeline', 'windows', 'fs' (target Hz), 'pre_ms', 'post_ms',
   'hp', 'lp', 'line', ...}
Preprocessing mirrors the notebook exactly:
  DC per epoch -> notch -> bandpass -> baseline [-pre,0] -> downsample.
"""
from __future__ import annotations

import os
import random
import time

import joblib
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.ui.tabs.p300 import DEFAULT_GRID

DEFAULT_MODEL = "analysis/models/p300/classifier.joblib"


class P300SpellerTab(TabPlugin):
    id = "p300_speller"
    title = "P300 Speller"
    tick_always = True

    COLS = 6

    # ---------------- UI ----------------
    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        self.lbl_typed = QtWidgets.QLabel("Typed: ")
        self.lbl_typed.setAlignment(Qt.AlignCenter)
        self.lbl_typed.setWordWrap(True)
        self.lbl_typed.setStyleSheet(
            "font-size: 26px; font-weight: bold; background-color: #1a2030; "
            "border-radius: 10px; padding: 8px;")
        left.addWidget(self.lbl_typed)
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
        self.lbl_prog = QtWidgets.QLabel("Idle — load model and press Spell letter")
        self.lbl_prog.setAlignment(Qt.AlignCenter)
        left.addWidget(self.lbl_prog)
        row = QtWidgets.QHBoxLayout()
        self.btn_spell = QtWidgets.QPushButton("Spell letter")
        self.btn_spell.setObjectName("ok")
        self.btn_spell.clicked.connect(self._start)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_del = QtWidgets.QPushButton("⌫")
        self.btn_del.clicked.connect(self._delete)
        self.btn_space = QtWidgets.QPushButton("Space")
        self.btn_space.clicked.connect(lambda: self._append(" "))
        row.addWidget(self.btn_spell)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_del)
        row.addWidget(self.btn_space)
        left.addLayout(row)
        lay.addLayout(left, stretch=3)
        # --- right: config ---
        cfg = QtWidgets.QGroupBox("Speller config")
        fl = QtWidgets.QFormLayout(cfg)
        self.txt_model = QtWidgets.QLineEdit(DEFAULT_MODEL)
        btn_browse = QtWidgets.QPushButton("…")
        btn_browse.setFixedWidth(36)
        btn_browse.clicked.connect(self._browse)
        mrow = QtWidgets.QHBoxLayout()
        mrow.addWidget(self.txt_model)
        mrow.addWidget(btn_browse)
        mw = QtWidgets.QWidget()
        mw.setLayout(mrow)
        self.spn_seqs = QtWidgets.QSpinBox(minimum=1, maximum=30, value=8)
        self.spn_flash = QtWidgets.QSpinBox(suffix=" ms", minimum=30,
                                            maximum=500, value=100, singleStep=10)
        self.spn_isi = QtWidgets.QSpinBox(suffix=" ms", minimum=0,
                                          maximum=1000, value=75, singleStep=25)
        fl.addRow("Model:", mw)
        fl.addRow("Sequences/letter:", self.spn_seqs)
        fl.addRow("Flash dur:", self.spn_flash)
        fl.addRow("ISI (gap):", self.spn_isi)
        self.lbl_model = QtWidgets.QLabel("No model loaded")
        self.lbl_model.setWordWrap(True)
        fl.addRow("Status:", self.lbl_model)
        lay.addWidget(cfg, stretch=1)
        self._reset()
        self.typed = ""
        self.bundle = None
        self._load_model(silent=True)
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

    # ---------------- model ----------------
    def _browse(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self.widget, "Load P300 classifier", "analysis/models/p300",
            "Joblib (*.joblib)")
        if path:
            self.txt_model.setText(path)
            self._load_model()

    def _load_model(self, silent=False):
        path = self.txt_model.text().strip()
        try:
            self.bundle = joblib.load(path)
            auc = self.bundle.get("cv_auc", float("nan"))
            self.lbl_model.setText(
                f"OK: {os.path.basename(path)} "
                f"(CV-AUC={auc:.3f}, {len(self.bundle.get('windows', []))} windows)")
        except Exception as e:
            self.bundle = None
            self.lbl_model.setText(f"Load failed: {e}")
            if not silent:
                QtWidgets.QMessageBox.critical(self.widget, "Model load failed", str(e))

    # ---------------- session state ----------------
    def _reset(self):
        self._disconnect()
        self.state = "IDLE"
        self.flashes: list[tuple[str, int]] = []
        self.fi = 0
        self.seq = 0
        self.next_flash_at = 0.0
        self.flash_on_until = 0.0
        self.lit: set[int] = set()
        self.buf: list[np.ndarray] = []
        self.buf_start_total = 0
        self.flash_log: list[dict] = []
        self._connected = False
        self._decode_at = 0.0

    def _disconnect(self):
        if getattr(self, "_connected", False):
            try:
                self.ctx.hub.new_samples.disconnect(self._collect)
            except (RuntimeError, TypeError):
                pass
            self._connected = False

    def teardown(self):
        self._disconnect()

    def _start(self):
        self._load_model(silent=True)
        if self.bundle is None:
            QtWidgets.QMessageBox.warning(self.widget, "P300 Speller",
                                           "Load a trained classifier.joblib first.")
            return
        self._reset()
        self.buf_start_total = self.ctx.hub.total
        self.ctx.hub.new_samples.connect(self._collect)
        self._connected = True
        self.state = "FLASH"
        self.btn_spell.setEnabled(False)
        self.lbl_prog.setText("Focus the letter you want… flashing!")
        self._new_sequence()

    def _stop(self):
        self.btn_spell.setEnabled(True)
        self._disconnect()
        self._set_lit(set())
        self.state = "IDLE"
        self.lbl_prog.setText("Stopped.")

    def _collect(self, n: int):
        if n > 0:
            sig, _, _ = self.ctx.hub.snapshot(
                seconds=n / max(self.ctx.hub.fs, 1) + 0.01)
            if len(sig):
                self.buf.append(sig[-n:].copy())

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

    def on_tick(self):
        if not self.widget or self.state == "IDLE":
            return
        now = time.time()
        if self.state == "FLASH":
            if self.lit and now >= self.flash_on_until:
                self._set_lit(set())
            if self.lit:
                return
            if now >= self.next_flash_at:
                if self.fi >= len(self.flashes):
                    self.seq += 1
                    if self.seq >= int(self.spn_seqs.value()):
                        # all sequences done: wait for post-epoch data then decode
                        self.state = "DECODE"
                        post_ms = int(self.bundle.get("post_ms", 800))
                        self._decode_at = now + post_ms / 1000.0 + 0.3
                        self.lbl_prog.setText("Decoding…")
                        return
                    self._new_sequence()
                    return
                kind, idx = self.flashes[self.fi]
                self.fi += 1
                self._set_lit(self._flash_cells(kind, idx))
                flash_ms = float(self.spn_flash.value()) / 1000.0
                isi_ms = float(self.spn_isi.value()) / 1000.0
                self.flash_on_until = now + flash_ms
                self.next_flash_at = now + flash_ms + isi_ms
                self.flash_log.append({"sample_idx": self.ctx.hub.total,
                                       "kind": kind, "idx": idx})
                self.lbl_prog.setText(
                    f"seq {self.seq+1}/{int(self.spn_seqs.value())} "
                    f"— flash {len(self.flash_log)}")
        elif self.state == "DECODE":
            if now >= self._decode_at:
                self._decode()

    # ---------------- decoding ----------------
    def _preprocess(self, X: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Mirror p300_training.ipynb: notch -> bandpass -> baseline -> downsample."""
        b = self.bundle
        Y = X - X.mean(axis=2, keepdims=True)
        line = float(b.get("line", 50.0))
        if 0 < line < fs / 2:
            bb, aa = iirnotch(line, 30.0, fs=fs)
            Y = filtfilt(bb, aa, Y, axis=2)
        sos = butter(4, [float(b.get("hp", 1.0)), float(b.get("lp", 12.0))],
                     btype="bandpass", fs=fs, output="sos")
        Y = sosfiltfilt(sos, Y, axis=2)
        n_pre = int(round(int(b.get("pre_ms", 200)) / 1000 * fs))
        Y = Y - Y[:, :, :n_pre].mean(axis=2, keepdims=True)
        target_fs = float(b.get("fs", 50.0))
        if target_fs and target_fs < fs:
            Y = resample_poly(Y, int(target_fs), int(fs), axis=2)
            fs = target_fs
        pre_ms, post_ms = int(b.get("pre_ms", 200)), int(b.get("post_ms", 800))
        tp = np.linspace(-pre_ms / 1000, post_ms / 1000, Y.shape[2])
        return Y, tp

    def _features(self, Xp: np.ndarray, tp: np.ndarray) -> np.ndarray:
        feats = []
        for lo, hi in self.bundle["windows"]:
            m = (tp >= lo) & (tp <= hi)
            feats.append(Xp[:, :, m].mean(axis=2))
        return np.concatenate(feats, axis=1)

    def _decode(self):
        try:
            data = np.vstack(self.buf)
            fs = self.ctx.hub.fs
            b = self.bundle
            pre = int(fs * int(b.get("pre_ms", 200)) / 1000.0)
            post = int(fs * int(b.get("post_ms", 800)) / 1000.0)
            Xs, keys = [], []
            for f in self.flash_log:
                rel = f["sample_idx"] - self.buf_start_total
                a, bb = rel - pre, rel + post
                if a < 0 or bb > len(data):
                    continue
                Xs.append(data[a:bb].T)
                keys.append((f["kind"], f["idx"]))
            if not Xs:
                raise RuntimeError("no flash epochs in buffer (device streaming?)")
            Xa = np.stack(Xs).astype(np.float64)
            Xp, tp = self._preprocess(Xa, fs)
            scores = b["pipeline"].predict_proba(self._features(Xp, tp))[:, 1]
            # average per (kind, idx); flashes reshuffled each sequence
            r = np.zeros(self.COLS)
            c = np.zeros(self.COLS)
            nr = np.zeros(self.COLS)
            nc = np.zeros(self.COLS)
            for s, (kind, idx) in zip(scores, keys):
                if kind == "row":
                    r[idx] += s
                    nr[idx] += 1
                else:
                    c[idx] += s
                    nc[idx] += 1
            r /= np.maximum(nr, 1)
            c /= np.maximum(nc, 1)
            grid = np.array(list(DEFAULT_GRID)).reshape(6, 6)
            ch = str(grid[int(np.argmax(r)), int(np.argmax(c))])
            self._append(ch)
            self.lbl_prog.setText(
                f"Decoded '{ch}' (row {int(np.argmax(r))} col {int(np.argmax(c))})")
        except Exception as e:
            self.lbl_prog.setText(f"Decode failed: {e}")
        finally:
            self.btn_spell.setEnabled(True)
            self._disconnect()
            self._set_lit(set())
            self.state = "IDLE"

    def _append(self, ch: str):
        self.typed += ch
        self.lbl_typed.setText(f"Typed: {self.typed}")

    def _delete(self):
        self.typed = self.typed[:-1]
        self.lbl_typed.setText(f"Typed: {self.typed}")
