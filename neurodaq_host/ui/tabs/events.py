"""Events tab: LSL/TCP/manual marker monitor + injection (P300/CV sync)."""
import time
from PySide6 import QtWidgets
from neurodaq_host.ui.tabs.base import TabPlugin
from neurodaq_host.core.events import Event


class EventsTab(TabPlugin):
    id = "events"
    title = "Events"

    def build(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.txt = QtWidgets.QLineEdit("stimulus/1")
        self.txt.setPlaceholderText("marker label, e.g. stimulus/row-3")
        btn = QtWidgets.QPushButton("Inject marker")
        btn.setObjectName("accent")
        btn.clicked.connect(self._inject)
        btn_export = QtWidgets.QPushButton("Export CSV")
        btn_export.clicked.connect(self._export)
        row.addWidget(self.txt)
        row.addWidget(btn)
        row.addWidget(btn_export)
        lay.addLayout(row)
        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["t_lsl", "kind", "label", "value"])
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.tbl)
        self.ctx.bus.event_added.connect(self._on_event)
        self.widget = w
        return w

    def _inject(self):
        import pylsl
        label = self.txt.text().strip() or "manual"
        self.ctx.bus.push(Event(sample_idx=self.ctx.hub.total, t_lsl=pylsl.local_clock(),
                                t_host=time.time(), kind="manual", label=label))

    def _on_event(self, ev):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c, v in enumerate([f"{ev.t_lsl:.3f}", ev.kind, ev.label, str(ev.value)]):
            self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.tbl.scrollToBottom()

    def _export(self):
        import csv
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.widget, "Export events", "events.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "t_lsl", "t_host", "kind", "label", "value"])
            for e in self.ctx.bus.all():
                w.writerow([e.sample_idx, e.t_lsl, e.t_host, e.kind, e.label, e.value])

    def on_tick(self):
        pass
