"""App bootstrap: wires Hub/Bus/Config/IO/Context/Window."""
import sys
from PySide6 import QtWidgets
from neurodaq_host.core.hub import Hub
from neurodaq_host.core.events import EventBus
from neurodaq_host.core.config import FilterConfig, DeviceConfig, N_CHANNELS
from neurodaq_host.core.profiles import load_profile
from neurodaq_host.devices.profiles import DEFAULT_EEG_ELECTRODES
from neurodaq_host.io.tcp import TCPClient
from neurodaq_host.io.udp import UDPReceiver
from neurodaq_host.io.lsl import LSLForwarder, MarkerIngest, MarkerPublisher
from neurodaq_host.ui.tabs.base import AppContext
from neurodaq_host.ui.main_window import MainWindow


def build_context() -> AppContext:
    filt = FilterConfig()
    dev = DeviceConfig()
    for i, name in enumerate(DEFAULT_EEG_ELECTRODES):
        dev.channels[i].electrode = name
    hub = Hub(fs=dev.sample_rate)
    bus = EventBus()
    tcp = TCPClient()
    ctx = AppContext(hub, bus, filt, dev, tcp, load_profile())
    ctx.electrode_mapping = [c.electrode for c in dev.channels]
    ctx.udp = UDPReceiver(hub)
    ctx.lsl_fwd = None
    ctx.marker_ingest = None
    return ctx


def main():
    app = QtWidgets.QApplication(sys.argv)
    ctx = build_context()
    win = MainWindow(ctx)
    win.show()
    ctx.tcp.start()
    ctx.udp.start()
    try:
        ctx.lsl_fwd = LSLForwarder(ctx.hub)
    except Exception as e:
        print(f"[lsl] outlet unavailable: {e}")
    try:
        ctx.marker_pub = MarkerPublisher(ctx.bus)
    except Exception as e:
        print(f"[lsl] marker outlet unavailable: {e}")
        ctx.marker_pub = None
    ctx.marker_ingest = MarkerIngest(ctx.bus, ctx.hub)
    ctx.marker_ingest.start()
    rc = app.exec()
    ctx.tcp.running = False
    ctx.udp.running = False
    ctx.marker_ingest.running = False
    ctx.tcp.wait(2000)
    ctx.udp.wait(2000)
    sys.exit(rc)
