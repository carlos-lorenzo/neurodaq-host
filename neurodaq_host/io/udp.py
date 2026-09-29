"""UDP receiver thread: parses packets, scales to uV, pushes to Hub only."""
import socket
import time
import numpy as np
from PySide6 import QtCore
from neurodaq_host.io import protocol as P
from neurodaq_host.core.config import VREF


class UDPReceiver(QtCore.QThread):
    stats = QtCore.Signal(dict)  # {drops, checksum_fails, total_packets}

    def __init__(self, hub, gains: list[float] | None = None, parent=None,
                 bind_host: str = "0.0.0.0"):
        super().__init__(parent)
        self.hub = hub
        self.gains = list(gains or [24.0] * P.N_CHANNELS)
        self.bind_host = bind_host
        self.running = True
        self._packets = 0
        self._last_seq: int | None = None
        self._lost = 0
        self.last_packet_time: float | None = None

    @property
    def lost(self) -> int:
        return self._lost

    @property
    def packet_age(self) -> float | None:
        """Seconds since the last valid packet, or None if none yet."""
        if self.last_packet_time is None:
            return None
        return time.time() - self.last_packet_time

    def _account_seq(self, seq: int):
        """Gap rule: seq > last + 1  ->  lost += seq - last - 1.

        First packet only sets the baseline (firmware boot counter rarely
        starts at 0, so no loss implied). A backwards seq means device
        restart / 32-bit wrap -> new baseline, cumulative total kept.
        """
        if self._last_seq is None:
            self._last_seq = seq
        elif seq == self._last_seq:
            pass  # duplicate delivery, ignore
        elif seq > self._last_seq:
            if seq > self._last_seq + 1:
                self._lost += seq - self._last_seq - 1
                self.hub.drops = self._lost
            self._last_seq = seq
        else:
            self._last_seq = seq

    def set_gain(self, ch: int, g: float):
        self.gains[ch] = g

    def run(self):
        import pylsl
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, P.UDP_PORT_DATA))
        sock.settimeout(0.5)
        while self.running:
            try:
                raw, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception:
                break
            size_ok, crc_ok, seq, unpacked = P.verify_packet(raw)
            if not size_ok or unpacked is None:
                continue
            self.last_packet_time = time.time()
            if not crc_ok:
                self.hub.checksum_fails += 1
            self._account_seq(seq)
            self._packets += 1
            payload = unpacked[2:-1]
            pkt = np.zeros((P.N_SAMPLES_PER_PACKET, P.N_CHANNELS))
            for s in range(P.N_SAMPLES_PER_PACKET):
                base = s * 12
                raw_ch = payload[base + 3: base + 11]
                for ch in range(P.N_CHANNELS):
                    scale = (2.0 * VREF / self.gains[ch]) / (2**24 - 1) * 1e6
                    pkt[s, ch] = raw_ch[ch] * scale
            self.hub.push(pkt, t_lsl=pylsl.local_clock())
            if self._packets % 20 == 0:
                self.stats.emit({"drops": self.hub.drops,
                                 "checksum_fails": self.hub.checksum_fails,
                                 "packets": self._packets})
        sock.close()
