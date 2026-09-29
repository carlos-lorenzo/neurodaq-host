"""Perf regression: shared display cache + stale-tick gating."""
import numpy as np

from neurodaq_host.core.hub import Hub
from neurodaq_host.core.config import FilterConfig


def test_display_cache_shares_filter():
    from neurodaq_host.dsp import filters as F
    hub = Hub()
    hub.push(np.random.randn(1500, 8).astype(np.float32))
    filt = FilterConfig()
    calls = [0]
    orig = F.apply_filters
    try:
        F.apply_filters = lambda *a, **k: (calls.__setitem__(0, calls[0] + 1),
                                           orig(*a, **k))[1]
        import neurodaq_host.core.hub as H
        H.apply_filters = F.apply_filters
        hub.display(3.0, filt, True)
        hub.display(3.0, filt, True)
        hub.display(3.0, filt, True)
        assert calls[0] == 1
        hub.push(np.random.randn(25, 8).astype(np.float32))
        hub.display(3.0, filt, True)
        assert calls[0] == 2
        filt.bump()
        hub.display(3.0, filt, True)
        assert calls[0] == 3
    finally:
        F.apply_filters = orig
        H.apply_filters = orig


def test_tick_skips_stale():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    import sys
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from neurodaq_host.app import build_context
    from neurodaq_host.ui.main_window import MainWindow
    ctx = build_context()
    w = MainWindow(ctx)
    calls = [0]
    scope = w.tabs["scope"]
    orig = scope.on_tick
    scope.on_tick = lambda: (calls.__setitem__(0, calls[0] + 1), orig())[1]
    try:
        w._tick()  # nothing yet... hub empty but total changes? total=0 both -> skip
        n0 = calls[0]
        ctx.hub.push(np.random.randn(25, 8))
        w._tick()
        assert calls[0] == n0 + 1  # fresh data -> ticked
        w._tick()
        assert calls[0] == n0 + 1  # stale -> skipped
    finally:
        scope.on_tick = orig
