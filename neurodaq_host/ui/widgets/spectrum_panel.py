"""Reusable PSD panel: FFT magnitude plot + update() from any tab.

Takes already-conditioned display data (use Hub.display) — no filtering here.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
from neurodaq_host.ui.theme import TOKENS, CH_COLORS


class SpectrumPanel(QtWidgets.QWidget):
    """Embeddable widget. Call update(data, fs, filt, visible_fn) per tick."""

    def __init__(self, n_channels: int = 8, parent=None, title="Spectrum (FFT)"):
        super().__init__(parent)
        self.n_channels = n_channels
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(title=title)
        self.plot.setBackground(TOKENS["bg"])
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left", "Magnitude (µV)")
        lay.addWidget(self.plot)
        self.plot.addLegend(offset=(-8, 8), labelTextSize="9pt",
                            verSpacing=-4, colCount=2)
        self.curves = [self.plot.plot(pen=pg.mkPen(CH_COLORS[i % len(CH_COLORS)], width=1.4),
                                      name=f"Ch{i+1}")
                       for i in range(n_channels)]

    def update(self, data: np.ndarray, fs: float, visible_fn):
        n = data.shape[0]
        if n < 32 or fs <= 0:
            return
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        for ch in range(self.n_channels):
            if visible_fn(ch):
                self.curves[ch].show()
                self.curves[ch].setData(freqs, np.abs(np.fft.rfft(data[:, ch])) / n * 2.0)
            else:
                self.curves[ch].hide()
