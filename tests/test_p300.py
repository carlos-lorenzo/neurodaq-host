"""P300 fast calibration session: oddball ratio + epoch shapes."""
import time
import numpy as np


def test_p300_session():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    import sys
    import PySide6.QtWidgets as QW
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from neurodaq_host.app import build_context
    from neurodaq_host.ui.main_window import MainWindow
    ctx = build_context()
    w = MainWindow(ctx)
    w.show()
    w._add_tab("p300")
    p = w.tabs["p300"]
    w.tabw.setCurrentWidget(p.widget)
    fn = "/tmp/t_p300_test.npz"
    QW.QFileDialog.getSaveFileName = lambda *a, **k: (fn, "NPZ (*.npz)")
    p.txt_text.setText("A")
    p.spn_seqs.setValue(1)
    p.spn_flash.setValue(30)
    p.spn_isi.setValue(0)
    p.spn_cue.setValue(0.2)
    p.spn_pre.setValue(100)
    p.spn_post.setValue(300)
    p._start()
    t0 = time.time()
    while p.state != "IDLE" and time.time() - t0 < 15:
        ctx.hub.push(np.random.randn(25, 8) * 20)
        p.on_tick()
        app.processEvents()
        time.sleep(0.01)
    assert p.state == "IDLE"
    z = np.load(fn, allow_pickle=True)
    assert z["X"].shape == (12, 1, 8, 100), z["X"].shape  # 6 rows+6 cols
    assert int(z["y"].sum()) == 2  # target row + target col
    ev = [e for e in ctx.bus.all() if e.kind == "p300"]
    assert len(ev) >= 13  # target cue + flash-start + 12 flashes
