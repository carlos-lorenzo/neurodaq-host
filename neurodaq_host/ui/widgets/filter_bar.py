"""Inline, collapsible DSP filter bar. Lives above plots, not in a popup.

Binds directly to ctx.filt so every consumer (scope, spectrum, brain)
reads the same live config. Emits `changed` after any edit.
"""
from PySide6 import QtCore, QtWidgets


class FilterBar(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self, filt, fs_fn, parent=None, title="Signal conditioning"):
        super().__init__(title, parent)
        self.filt = filt
        self._fs_fn = fs_fn  # callable -> current fs (hub may change rate)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(lambda _on: self.changed.emit())
        self._build()
        self.refresh()

    def _build(self):
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(10, 6, 10, 6)
        grid.setHorizontalSpacing(8)

        self.chk_dc = QtWidgets.QCheckBox("DC remove")
        self.chk_dc.toggled.connect(self._push)
        grid.addWidget(self.chk_dc, 0, 0)

        self.chk_hp = QtWidgets.QCheckBox("HP")
        self.chk_hp.toggled.connect(self._push)
        self.hp_f = QtWidgets.QDoubleSpinBox(suffix=" Hz", minimum=0.1,
                                             maximum=8000, singleStep=0.5)
        self.hp_f.valueChanged.connect(self._push)
        self.hp_o = QtWidgets.QSpinBox(prefix="n=", minimum=1, maximum=8)
        self.hp_o.valueChanged.connect(self._push)
        grid.addWidget(self.chk_hp, 0, 1)
        grid.addWidget(self.hp_f, 0, 2)
        grid.addWidget(self.hp_o, 0, 3)

        self.chk_lp = QtWidgets.QCheckBox("LP")
        self.chk_lp.toggled.connect(self._push)
        self.lp_f = QtWidgets.QDoubleSpinBox(suffix=" Hz", minimum=0.1,
                                             maximum=8000, singleStep=1.0)
        self.lp_f.valueChanged.connect(self._push)
        self.lp_o = QtWidgets.QSpinBox(prefix="n=", minimum=1, maximum=8)
        self.lp_o.valueChanged.connect(self._push)
        grid.addWidget(self.chk_lp, 0, 4)
        grid.addWidget(self.lp_f, 0, 5)
        grid.addWidget(self.lp_o, 0, 6)

        self.chk_n = QtWidgets.QCheckBox("Notch")
        self.chk_n.toggled.connect(self._push)
        self.n_f = QtWidgets.QDoubleSpinBox(suffix=" Hz", minimum=0.1,
                                            maximum=8000, singleStep=1.0)
        self.n_f.valueChanged.connect(self._push)
        self.n_q = QtWidgets.QDoubleSpinBox(prefix="Q ", minimum=1.0,
                                            maximum=100.0, singleStep=1.0)
        self.n_q.valueChanged.connect(self._push)
        grid.addWidget(self.chk_n, 0, 7)
        grid.addWidget(self.n_f, 0, 8)
        grid.addWidget(self.n_q, 0, 9)

        self.chk_view = QtWidgets.QCheckBox("Filtered view")
        self.chk_view.setChecked(True)
        self.chk_view.toggled.connect(lambda _v: self.changed.emit())
        grid.addWidget(self.chk_view, 0, 10)
        grid.setColumnStretch(11, 1)

    def _nyq_max(self) -> float:
        try:
            fs = float(self._fs_fn())
        except Exception:
            fs = 250.0
        return max(0.1, fs / 2.0 - 0.1)

    def refresh(self):
        """Push model -> widgets (block signals to avoid feedback)."""
        f, mx = self.filt, self._nyq_max()
        for w in (self.hp_f, self.lp_f, self.n_f):
            w.blockSignals(True)
            w.setMaximum(mx)
        self.chk_dc.blockSignals(True)
        self.chk_hp.blockSignals(True)
        self.chk_lp.blockSignals(True)
        self.chk_n.blockSignals(True)
        self.hp_o.blockSignals(True)
        self.lp_o.blockSignals(True)
        self.n_q.blockSignals(True)
        try:
            self.chk_dc.setChecked(f.dc_remove)
            self.chk_hp.setChecked(f.hp_enabled)
            self.hp_f.setValue(min(f.hp_freq, mx))
            self.hp_o.setValue(f.hp_order)
            self.chk_lp.setChecked(f.lp_enabled)
            self.lp_f.setValue(min(f.lp_freq, mx))
            self.lp_o.setValue(f.lp_order)
            self.chk_n.setChecked(f.notch_enabled)
            self.n_f.setValue(min(f.notch_freq, mx))
            self.n_q.setValue(f.notch_q)
        finally:
            for w in (self.hp_f, self.lp_f, self.n_f, self.chk_dc,
                      self.chk_hp, self.chk_lp, self.chk_n,
                      self.hp_o, self.lp_o, self.n_q):
                w.blockSignals(False)

    def _push(self, *_a):
        """Push widgets -> model with Nyquist clamping."""
        mx = self._nyq_max()
        f = self.filt
        f.dc_remove = self.chk_dc.isChecked()
        f.hp_enabled = self.chk_hp.isChecked()
        f.hp_freq = min(self.hp_f.value(), mx)
        f.hp_order = self.hp_o.value()
        f.lp_enabled = self.chk_lp.isChecked()
        f.lp_freq = min(self.lp_f.value(), mx)
        f.lp_order = self.lp_o.value()
        f.notch_enabled = self.chk_n.isChecked()
        f.notch_freq = min(self.n_f.value(), mx)
        f.notch_q = self.n_q.value()
        f.bump()
        self.changed.emit()

    @property
    def view_filtered(self) -> bool:
        return self.chk_view.isChecked() and self.isChecked()
