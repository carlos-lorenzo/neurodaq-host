"""Offscreen UI smoke: every registered tab builds and ticks."""
import numpy as np


def _ctx():
    from neurodaq_host.app import build_context
    return build_context()


def test_all_tabs_tick():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    import sys
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from neurodaq_host.ui.main_window import MainWindow, REGISTRY
    ctx = _ctx()
    w = MainWindow(ctx)
    w.show()
    for tid in REGISTRY:
        ctx.profile["tabs"] = [{"id": t, "enabled": True, "order": i}
                               for i, t in enumerate(REGISTRY)]
    for tid in REGISTRY:
        w._add_tab(tid)
    ctx.hub.push(np.random.randn(25, 8) * 20)
    for tid, plug in w.tabs.items():
        plug.on_tick()
    assert set(w.tabs) == set(REGISTRY)


def test_scope_panel_state_roundtrip():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    import sys
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from neurodaq_host.ui.main_window import MainWindow
    ctx = _ctx()
    w = MainWindow(ctx)
    scope = w.tabs["scope"]
    scope.chk_psd.setChecked(False)
    st = scope.panel_state()
    assert st["psd"] is False
    scope.chk_psd.setChecked(True)
    scope.restore_state(st)
    assert scope.chk_psd.isChecked() is False
