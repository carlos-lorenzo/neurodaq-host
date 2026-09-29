"""Shared DSP filter settings dialog with Nyquist enforcement."""
from PySide6 import QtWidgets
from neurodaq_host.core.config import FilterConfig


class FilterDialog(QtWidgets.QDialog):
    def __init__(self, filt: FilterConfig, fs: float, parent=None):
        super().__init__(parent)
        self.filt = filt
        self.fs = fs
        self.setWindowTitle("Signal conditioning")
        lay = QtWidgets.QFormLayout(self)
        self.chk_dc = QtWidgets.QCheckBox("Remove DC (mean subtraction)")
        self.chk_dc.setChecked(filt.dc_remove)
        lay.addRow(self.chk_dc)
        self.chk_hp = QtWidgets.QCheckBox("Highpass")
        self.chk_hp.setChecked(filt.hp_enabled)
        self.hp_f = QtWidgets.QDoubleSpinBox(suffix=" Hz", minimum=0.1, maximum=1000,
                                             value=filt.hp_freq, singleStep=0.5)
        self.hp_o = QtWidgets.QSpinBox(prefix="order ", minimum=1, maximum=8,
                                       value=filt.hp_order)
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(self.chk_hp); r1.addWidget(self.hp_f); r1.addWidget(self.hp_o)
        lay.addRow(QtWidgets.QWidget())
        lay.addRow("Highpass:", r1)
        self.chk_lp = QtWidgets.QCheckBox("Lowpass")
        self.chk_lp.setChecked(filt.lp_enabled)
        self.lp_f = QtWidgets.QDoubleSpinBox(suffix=" Hz", minimum=0.1, maximum=8000,
                                             value=filt.lp_freq, singleStep=1.0)
        self.lp_o = QtWidgets.QSpinBox(prefix="order ", minimum=1, maximum=8,
                                       value=filt.lp_order)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(self.chk_lp); r2.addWidget(self.lp_f); r2.addWidget(self.lp_o)
        lay.addRow("Lowpass:", r2)
        self.chk_n = QtWidgets.QCheckBox("Notch")
        self.chk_n.setChecked(filt.notch_enabled)
        self.n_f = QtWidgets.QDoubleSpinBox(suffix=" Hz", minimum=0.1, maximum=8000,
                                            value=filt.notch_freq, singleStep=1.0)
        self.n_q = QtWidgets.QDoubleSpinBox(prefix="Q ", minimum=1.0, maximum=100.0,
                                            value=filt.notch_q, singleStep=1.0)
        r3 = QtWidgets.QHBoxLayout()
        r3.addWidget(self.chk_n); r3.addWidget(self.n_f); r3.addWidget(self.n_q)
        lay.addRow("Notch:", r3)
        self._enforce()
        for w in (self.hp_f, self.lp_f, self.n_f):
            w.valueChanged.connect(lambda _v: self._enforce())
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self._apply_and_close)
        btns.rejected.connect(self.reject)
        lay.addRow(btns)

    def _enforce(self):
        nyq = self.fs / 2.0
        mx = max(0.1, nyq - 0.1)
        for w in (self.hp_f, self.lp_f, self.n_f):
            w.setMaximum(mx)
            if w.value() > mx:
                w.setValue(mx)

    def _apply_and_close(self):
        self._enforce()
        f = self.filt
        f.dc_remove = self.chk_dc.isChecked()
        f.hp_enabled = self.chk_hp.isChecked()
        f.hp_freq = self.hp_f.value(); f.hp_order = self.hp_o.value()
        f.lp_enabled = self.chk_lp.isChecked()
        f.lp_freq = self.lp_f.value(); f.lp_order = self.lp_o.value()
        f.notch_enabled = self.chk_n.isChecked()
        f.notch_freq = self.n_f.value(); f.notch_q = self.n_q.value()
        f.bump()
        self.accept()
