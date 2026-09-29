"""LSL outlets/inlets: EEG stream out, marker streams in/out for sync."""
import numpy as np
from PySide6 import QtCore
from neurodaq_host.io import protocol as P


def make_eeg_outlet(fs: float):
    from pylsl import StreamInfo, StreamOutlet
    info = StreamInfo("NeuroDAQ EEG", "EEG", P.N_CHANNELS, fs,
                      "float32", "neurodaq_eeg_stream")
    chns = info.desc().append_child("channels")
    for i in range(P.N_CHANNELS):
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"Ch{i+1}")
        ch.append_child_value("unit", "microvolts")
    return StreamOutlet(info)


def make_marker_outlet():
    from pylsl import StreamInfo, StreamOutlet
    info = StreamInfo("NeuroDAQ Markers", "Markers", 1, 0, "string",
                      "neurodaq_markers")
    return StreamOutlet(info)


class LSLForwarder(QtCore.QObject):
    """Subscribes to Hub, pushes each packet to the EEG outlet."""

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.hub = hub
        self.outlet = make_eeg_outlet(hub.fs)
        self._pending = []
        hub.new_samples.connect(self._on_new)

    def set_fs(self, fs: float):
        self.outlet = make_eeg_outlet(fs)

    def _on_new(self, n: int):
        sig, _, _ = self.hub.snapshot(seconds=n / max(self.hub.fs, 1) + 0.01)
        for row in sig[-n:]:
            self.outlet.push_sample(row.astype(float).tolist())


class MarkerPublisher(QtCore.QObject):
    """Forwards every EventBus event to our LSL marker outlet so external
    apps (P300 stimulators, CV processes, recorders) see our markers too."""

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.outlet = make_marker_outlet()
        bus.event_added.connect(self._on_event)

    def _on_event(self, ev):
        try:
            self.outlet.push_sample([f"{ev.kind}:{ev.label}"])
        except Exception:
            pass

    def inject(self, label: str, kind: str = "manual"):
        """Publish an ad-hoc marker without a bus event (bus push preferred)."""
        try:
            self.outlet.push_sample([f"{kind}:{label}"])
        except Exception:
            pass


class MarkerIngest(QtCore.QThread):
    """Resolves LSL marker streams, pushes arrivals into the EventBus."""

    def __init__(self, bus, hub, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.hub = hub
        self.running = True

    def run(self):
        import pylsl, time
        try:
            streams = pylsl.resolve_byprop("type", "Markers", timeout=2.0)
        except Exception:
            return
        inlets = [pylsl.StreamInlet(s) for s in streams
                  if "NeuroDAQ Markers" not in s.name()]
        while self.running:
            for inl in inlets:
                try:
                    samples, stamps = inl.pull_chunk(timeout=0.0)
                except Exception:
                    continue
                for s, t in zip(samples, stamps):
                    from neurodaq_host.core.events import Event
                    self.bus.push(Event(sample_idx=self.hub.total, t_lsl=float(t),
                                        t_host=time.time(), kind="lsl_marker",
                                        label=str(s[0]) if s else ""))
            self.msleep(20)
