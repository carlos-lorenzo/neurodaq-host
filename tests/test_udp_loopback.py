"""UDP loopback: real packets through the real receiver thread.

Sends synthetic wire packets (with a sequence gap) over localhost UDP and
asserts the receiver counts the loss and lands the samples in the Hub.
"""
import socket
import struct
import time
import numpy as np

from neurodaq_host.io import protocol as P
from neurodaq_host.io.udp import UDPReceiver
from neurodaq_host.core.hub import Hub


def _packet(seq: int) -> bytes:
    payload = b"".join([
        struct.pack("<3B", 0, 0, 0) + b"\x00"
        + struct.pack("<8i", *([1000] * 8)) + b"\x00" * 4
        + struct.pack("<Q", 0)
        for _ in range(25)
    ])
    raw = struct.pack("<II", P.MAGIC_HEADER, seq) + payload
    assert len(raw) == P.EXPECTED_SIZE - 1
    crc = 0
    for b in raw[8:]:
        crc ^= b
    return raw + struct.pack("<B", crc)


def test_account_seq_rule():
    hub = Hub()
    r = UDPReceiver(hub)
    r._account_seq(500)          # baseline, no loss implied
    assert hub.drops == 0
    r._account_seq(501)          # in order
    assert hub.drops == 0
    r._account_seq(505)          # gap of 3 (502,503,504)
    assert hub.drops == 3, hub.drops
    r._account_seq(505)          # duplicate, ignored
    assert hub.drops == 3
    r._account_seq(7)            # restart/wrap -> re-baseline, total kept
    assert hub.drops == 3
    r._account_seq(8)
    assert hub.drops == 3


def test_loopback_gap():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    import sys
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    hub = Hub()
    rx = UDPReceiver(hub, bind_host="127.0.0.1")  # loopback only: ignores any
    rx.start()                                    # live device traffic on :3333
    try:
        time.sleep(0.2)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for seq in (100, 101, 105):  # 102,103,104 lost on the wire
            sock.sendto(_packet(seq), ("127.0.0.1", P.UDP_PORT_DATA))
        deadline = time.time() + 5.0
        while hub.total < 75 and time.time() < deadline:
            time.sleep(0.05)
        assert hub.total == 75, hub.total
        assert hub.drops == 3, hub.drops
        assert rx.lost == 3
        sig, _, _ = hub.snapshot(seconds=1.0)
        assert np.abs(sig).max() > 0  # samples actually decoded
    finally:
        rx.running = False
        rx.wait(3000)
