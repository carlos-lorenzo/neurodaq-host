"""EMG tab: envelope + RMS bars. Proves the modality swap story.

No EEG assumptions: no 10-20 map, no bands. Reads the same Hub bytestream
and renders rectified envelopes with per-channel RMS. Enable it (and disable
Brain) for an EMG session — that config is savable as a profile.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.ui.theme import TOKENS, CH_COLORS
from neurodaq_host.ui.widgets.filter_bar import FilterBar


class EMGTab(TabPlugin):
    id = "emg"
    title = "EMG"

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        self.filter_bar = FilterBar(self.ctx.filt, lambda: self.ctx.hub.fs,
                                    title="EMG conditioning (shared DSP chain)")
        lay.addWidget(self.filter_bar)
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Envelope window:"))
        self.spn_win = QtWidgets.QSpinBox(suffix=" ms", minimum=10, maximum=1000,
                                          value=100, singleStep=25)
        ctl.addWidget(self.spn_win)
        ctl.addStretch()
        lay.addLayout(ctl)
        self.env_plot = pg.PlotWidget(title="Envelopes (rectified + smoothed, offset)")
        self.env_plot.setBackground(TOKENS["bg"])
        self.env_plot.showGrid(x=True, y=True, alpha=0.2)
        lay.addWidget(self.env_plot, stretch=3)
        self.env_plot.addLegend(offset=(-8, 8), labelTextSize="9pt",
                                verSpacing=-4, colCount=2)
        self.env_curves = [self.env_plot.plot(pen=pg.mkPen(CH_COLORS[i % 8], width=1.4),
                                              name=f"Ch{i+1}")
                           for i in range(8)]
        self.rms_plot = pg.PlotWidget(title="RMS per channel")
        self.rms_plot.setBackground(TOKENS["bg"])
        self.rms_bars = pg.BarGraphItem(x=list(range(8)), height=[0]*8,
                                        width=0.6, brush=TOKENS["accent"])
        self.rms_plot.addItem(self.rms_bars)
        lay.addWidget(self.rms_plot, stretch=1)
        self.widget = w
        return w

    def on_tick(self):
        if not self.widget or not self.widget.isVisible():
            return
        self.filter_bar.refresh()
        d = self.ctx.hub.display(3.0, self.ctx.filt, self.filter_bar.view_filtered)
        rect = np.abs(d)
        k = max(1, int(self.ctx.hub.fs * float(self.spn_win.value()) / 1000.0))
        kernel = np.ones(k) / k
        env = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, rect)
        span = max(1.0, float(np.max(env)))
        off = span * 1.5
        rms = []
        for ch in range(8):
            if self.ctx.dev.channels[ch].visible and not self.ctx.dev.channels[ch].power_down:
                self.env_curves[ch].show()
                self.env_curves[ch].setData(env[:, ch] + ch * off)
                rms.append(float(np.sqrt(np.mean(d[:, ch] ** 2))))
            else:
                self.env_curves[ch].hide()
                rms.append(0.0)
        self.rms_bars.setOpts(height=rms)
