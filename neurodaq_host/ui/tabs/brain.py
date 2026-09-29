"""Brain tab: OWNS the 10-20 electrode mapping + RBF scalp heatmap + band bars.

Kept separate from hardware config so EMG (or any modality) simply omits it.
"""
import numpy as np
import pyqtgraph as pg
from scipy.interpolate import Rbf
from PySide6 import QtWidgets
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.ui.theme import TOKENS
from neurodaq_host.devices.profiles import ELECTRODE_1020_POS
from neurodaq_host.dsp.filters import band_power, EEG_BANDS

REGIONS = ["Frontal", "Central", "Parietal", "Occipital", "Temporal"]


def _lut():
    stops = [(0.0, (22, 26, 32)), (0.35, (45, 212, 191)), (0.65, (96, 165, 250)),
             (0.85, (251, 191, 36)), (1.0, (248, 113, 113))]
    lut = np.zeros((256, 4), dtype=np.uint8)
    xp = np.linspace(0, 1, 256)
    for i, c in enumerate("rgb"):
        lut[:, i] = np.interp(xp, [s[0] for s in stops], [s[1][i] for s in stops])
    lut[:, 3] = 215
    return lut


class BrainTab(TabPlugin):
    id = "brain"
    title = "Brain Map"

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        # --- left: electrode mapping + topo ---
        left = QtWidgets.QGroupBox("Scalp map & electrode assignment")
        lv = QtWidgets.QVBoxLayout(left)
        maprow = QtWidgets.QGridLayout()
        self.combos: list[QtWidgets.QComboBox] = []
        for ch in range(8):
            maprow.addWidget(QtWidgets.QLabel(f"Ch{ch+1}"), ch // 4, (ch % 4) * 2)
            cb = QtWidgets.QComboBox()
            for name in sorted(ELECTRODE_1020_POS):
                cb.addItem(name)
            cur = self.ctx.electrode_mapping[ch] if self.ctx.electrode_mapping[ch] else \
                ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"][ch]
            cb.setCurrentText(cur)
            cb.currentTextChanged.connect(lambda t, c=ch: self._set_map(c, t))
            self.combos.append(cb)
            maprow.addWidget(cb, ch // 4, (ch % 4) * 2 + 1)
        lv.addLayout(maprow)
        bandrow = QtWidgets.QHBoxLayout()
        bandrow.addWidget(QtWidgets.QLabel("Band:"))
        self.cmb_band = QtWidgets.QComboBox()
        for b in EEG_BANDS:
            self.cmb_band.addItem(b)
        self.cmb_band.setCurrentText("Alpha")
        bandrow.addWidget(self.cmb_band)
        bandrow.addStretch()
        lv.addLayout(bandrow)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(TOKENS["bg"])
        self.plot.setAspectLocked(True)
        self.plot.setRange(xRange=[-1.25, 1.25], yRange=[-1.25, 1.25])
        self.plot.hideAxis("left"); self.plot.hideAxis("bottom")
        self.rect = __import__("PySide6.QtCore", fromlist=["QRectF"]).QRectF(-1.1, -1.1, 2.2, 2.2)
        self.img = pg.ImageItem()
        self.img.setRect(self.rect)
        self.plot.addItem(self.img)
        th = np.linspace(0, 2 * np.pi, 160)
        self.plot.plot(np.cos(th), np.sin(th), pen=pg.mkPen(TOKENS["text"], width=2))
        self.plot.plot([-0.15, 0, 0.15], [0.98, 1.15, 0.98], pen=pg.mkPen(TOKENS["text"], width=2))
        self.scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(TOKENS["bg"], width=1.5))
        self.plot.addItem(self.scatter)
        lv.addWidget(self.plot)
        lay.addWidget(left, stretch=3)
        # --- right: bars ---
        right = QtWidgets.QGroupBox("Band power")
        rv = QtWidgets.QVBoxLayout(right)
        self.reg_plot = pg.PlotWidget(title="By region")
        self.reg_plot.setBackground(TOKENS["bg"])
        self.reg_bars = pg.BarGraphItem(x=list(range(5)), height=[0]*5, width=0.6,
                                        brush=TOKENS["accent"])
        self.reg_plot.addItem(self.reg_bars)
        self.reg_plot.getAxis("bottom").setTicks([[list(enumerate(REGIONS))][0]])
        self.band_plot = pg.PlotWidget(title="Global bands")
        self.band_plot.setBackground(TOKENS["bg"])
        self.band_bars = pg.BarGraphItem(x=list(range(5)), height=[0]*5, width=0.6,
                                         brush=TOKENS["info"])
        self.band_plot.addItem(self.band_bars)
        self.band_plot.getAxis("bottom").setTicks([[(0, "Delta"), (1, "Theta"),
                                                    (2, "Alpha"), (3, "Beta"), (4, "Gamma")]])
        rv.addWidget(self.reg_plot); rv.addWidget(self.band_plot)
        lay.addWidget(right, stretch=2)
        # grid cache
        self._res = 100
        g = np.linspace(-1.1, 1.1, self._res)
        self._gx, self._gy = np.meshgrid(g, g)
        self._mask = (self._gx**2 + self._gy**2) > 1.0
        self._lut = _lut()
        self._texts: dict[int, object] = {}  # ch -> reused TextItem
        self._tick_n = 0
        self.widget = w
        return w

    def _set_map(self, ch, name):
        self.ctx.electrode_mapping[ch] = name
        self.ctx.dev.channels[ch].electrode = name

    def on_tick(self):
        if not self.widget or not self.widget.isVisible():
            return
        self._tick_n += 1
        # Traces stay at full rate; the RBF heatmap is the expensive part
        # (~10k evals) and scalp power barely changes faster than ~5 Hz.
        do_map = (self._tick_n % 4 == 0)
        d = self.ctx.hub.display(3.0, self.ctx.filt, True)
        band = EEG_BANDS[self.cmb_band.currentText()]
        pw = np.array([band_power(d[:, ch], self.ctx.hub.fs, *band) for ch in range(8)])
        xs, ys, zs, spots = [], [], [], []
        seen = set()
        for ch in range(8):
            if self.ctx.dev.channels[ch].power_down:
                continue
            lab = self.ctx.electrode_mapping[ch]
            x, y, _ = ELECTRODE_1020_POS.get(lab, (0.0, 0.0, ""))
            xs.append(x); ys.append(y); zs.append(pw[ch])
            spots.append({"pos": (x, y), "brush": pg.mkBrush(TOKENS["text"])})
            seen.add(ch)
            tx = self._texts.get(ch)
            if tx is None:
                tx = pg.TextItem("", color=TOKENS["muted"], anchor=(0.5, 0.5))
                self.plot.addItem(tx)
                self._texts[ch] = tx
            tx.setText(f"{lab}\n(Ch{ch+1})")
            tx.setPos(x, y - 0.13)
        for ch in list(self._texts):
            if ch not in seen:
                self.plot.removeItem(self._texts.pop(ch))
        self.scatter.setData(spots)
        if do_map and len(xs) >= 3:
            try:
                z = Rbf(xs, ys, zs, function="multiquadric", smooth=0.05)(self._gx, self._gy)
            except Exception:
                z = np.zeros_like(self._gx)
            rng = z.max() - z.min()
            nz = np.clip((z - z.min()) / rng, 0, 1) if rng > 0 else np.zeros_like(z)
            rgba = self._lut[(nz * 255).astype(np.uint8)]
            rgba[self._mask, 3] = 0
            self.img.setImage(np.transpose(rgba, (1, 0, 2)), levels=(0, 255),
                              rect=self.rect)
        # bars
        reg_sum = {r: [0.0, 0] for r in REGIONS}
        for ch in range(8):
            if self.ctx.dev.channels[ch].power_down:
                continue
            r = ELECTRODE_1020_POS.get(self.ctx.electrode_mapping[ch], (0, 0, "Frontal"))[2]
            reg_sum[r][0] += pw[ch]; reg_sum[r][1] += 1
        self.reg_bars.setOpts(height=[reg_sum[r][0] / max(reg_sum[r][1], 1) for r in REGIONS])

        if do_map:
            bh = []
            for _, lim in EEG_BANDS.items():
                tot = sum(band_power(d[:, ch], self.ctx.hub.fs, *lim)
                          for ch in range(8) if not self.ctx.dev.channels[ch].power_down)
                bh.append(tot / 8)
            self.band_bars.setOpts(height=bh)
