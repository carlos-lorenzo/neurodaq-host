"""Scope tab: inline filters + time traces + optional PSD subplot.

The PSD is the shared SpectrumPanel, so scope and spectrum views always
agree. Toggle it with "Show PSD" or collapse the filter bar via its header.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.ui.theme import TOKENS, CH_COLORS
from neurodaq_host.ui.widgets.filter_bar import FilterBar
from neurodaq_host.ui.widgets.spectrum_panel import SpectrumPanel


class ScopeTab(TabPlugin):
    id = "scope"
    title = "Oscilloscope"

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        self.filter_bar = FilterBar(self.ctx.filt, lambda: self.ctx.hub.fs)
        lay.addWidget(self.filter_bar)

        ctl = QtWidgets.QHBoxLayout()
        self.chk_wave = QtWidgets.QCheckBox("Show waveform")
        self.chk_wave.setChecked(True)
        self.chk_wave.toggled.connect(self._toggle_wave)
        self.chk_psd = QtWidgets.QCheckBox("Show PSD")
        self.chk_psd.setChecked(True)
        self.chk_psd.toggled.connect(self._toggle_psd)
        self.spn_off = QtWidgets.QDoubleSpinBox(prefix="Offset: ", suffix=" µV",
                                                minimum=0, maximum=10000, value=100)
        self.spn_sec = QtWidgets.QDoubleSpinBox(prefix="Window: ", suffix=" s",
                                                minimum=0.5, maximum=10, value=3.0,
                                                singleStep=0.5)
        ctl.addWidget(self.chk_wave)
        ctl.addWidget(self.chk_psd)
        ctl.addWidget(self.spn_off)
        ctl.addWidget(self.spn_sec)
        ctl.addStretch()
        lay.addLayout(ctl)

        self.split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.scope_plot = pg.PlotWidget(title="Time domain (µV + offset)")
        self.scope_plot.setBackground(TOKENS["bg"])
        self.scope_plot.showGrid(x=True, y=True, alpha=0.2)
        self.scope_plot.setLabel("left", "Voltage + offset (µV)")
        self.scope_plot.setLabel("bottom", "Samples")
        self.split.addWidget(self.scope_plot)
        self.scope_plot.addLegend(offset=(-8, 8), labelTextSize="9pt",
                                  verSpacing=-4, colCount=2)
        self.curves = [self.scope_plot.plot(pen=pg.mkPen(CH_COLORS[i % 8], width=1.4),
                                            name=f"Ch{i+1}")
                       for i in range(8)]

        self.psd = SpectrumPanel()
        self.split.addWidget(self.psd)
        self.psd.setVisible(self.chk_psd.isChecked())
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        lay.addWidget(self.split)

        self.widget = w
        return w

    def _toggle_psd(self, on: bool):
        self.psd.setVisible(on)

    def _toggle_wave(self, on: bool):
        self.scope_plot.setVisible(on)

    def panel_state(self) -> dict:
        return {"wave": self.chk_wave.isChecked(),
                "psd": self.chk_psd.isChecked(),
                "filters_open": self.filter_bar.isChecked(),
                "view_filtered": self.filter_bar.chk_view.isChecked()}

    def restore_state(self, state: dict):
        if "wave" in state:
            self.chk_wave.setChecked(bool(state["wave"]))
        if "psd" in state:
            self.chk_psd.setChecked(bool(state["psd"]))
        if "filters_open" in state:
            self.filter_bar.setChecked(bool(state["filters_open"]))
        if "view_filtered" in state:
            self.filter_bar.chk_view.setChecked(bool(state["view_filtered"]))

    def _visible(self, ch: int) -> bool:
        cfg = self.ctx.dev.channels[ch]
        return bool(cfg.visible and not cfg.power_down)

    def _display(self) -> np.ndarray:
        return self.ctx.hub.display(float(self.spn_sec.value()), self.ctx.filt,
                                    self.filter_bar.view_filtered)

    def on_tick(self):
        if not self.widget or not self.widget.isVisible():
            return
        if not self.scope_plot.isVisible() and not self.psd.isVisible():
            return
        self.filter_bar.refresh()  # picks up fs changes for Nyquist clamp
        d = self._display()
        off = float(self.spn_off.value())
        if self.scope_plot.isVisible():
            for ch in range(8):
                if self._visible(ch):
                    self.curves[ch].show()
                    self.curves[ch].setData(d[:, ch] + ch * off)
                else:
                    self.curves[ch].hide()
        if self.psd.isVisible():
            self.psd.update(d, self.ctx.hub.fs, self._visible)
