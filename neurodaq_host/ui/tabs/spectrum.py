"""Spectrum tab: standalone full-size view reusing SpectrumPanel."""
from PySide6 import QtWidgets
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.ui.widgets.filter_bar import FilterBar
from neurodaq_host.ui.widgets.spectrum_panel import SpectrumPanel


class SpectrumTab(TabPlugin):
    id = "spectrum"
    title = "Spectrum"

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        self.filter_bar = FilterBar(self.ctx.filt, lambda: self.ctx.hub.fs)
        lay.addWidget(self.filter_bar)
        self.psd = SpectrumPanel()
        lay.addWidget(self.psd)
        self.widget = w
        return w

    def _visible(self, ch: int) -> bool:
        cfg = self.ctx.dev.channels[ch]
        return bool(cfg.visible and not cfg.power_down)

    def on_tick(self):
        if not self.widget or not self.widget.isVisible():
            return
        self.filter_bar.refresh()
        d = self.ctx.hub.display(3.0, self.ctx.filt, self.filter_bar.view_filtered)
        self.psd.update(d, self.ctx.hub.fs, self._visible)
