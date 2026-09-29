"""Hub: the single raw-bytestream store. Only io/udp writes; all tabs read."""
from __future__ import annotations
import time
import numpy as np
from PySide6 import QtCore
from neurodaq_host.core.config import N_CHANNELS


class Hub(QtCore.QObject):
    """Thread-safe ring buffer of calibrated uV samples + clocks.

    Emits new_samples(n) from the UDP thread; GUI tabs pull via snapshot().
    """

    new_samples = QtCore.Signal(int)

    def __init__(self, fs: float = 250.0, seconds: float = 6.0, parent=None):
        super().__init__(parent)
        self._lock = QtCore.QMutex()
        self._fs = fs
        self._cap = max(64, int(fs * seconds))
        self._buf = np.zeros((self._cap, N_CHANNELS), dtype=np.float32)
        self._t_host = np.zeros(self._cap, dtype=np.float64)
        self._t_lsl = np.zeros(self._cap, dtype=np.float64)
        self._n = 0  # total samples ever pushed
        self.drops = 0
        self.checksum_fails = 0
        self._cache_key = None
        self._cache_data = None

    @property
    def fs(self) -> float:
        return self._fs

    def set_fs(self, fs: float, seconds: float = 6.0):
        locker = QtCore.QMutexLocker(self._lock)
        self._fs = fs
        self._cap = max(64, int(fs * seconds))
        self._buf = np.zeros((self._cap, N_CHANNELS), dtype=np.float32)
        self._t_host = np.zeros(self._cap, dtype=np.float64)
        self._t_lsl = np.zeros(self._cap, dtype=np.float64)
        self._n = 0
        self._cache_key = None
        self._cache_data = None

    def push(self, packet_uV: np.ndarray, t_lsl: float | None = None):
        """Append a (S, C) float packet. Called from the UDP thread."""
        import pylsl
        s = packet_uV.shape[0]
        now_h = time.time()
        now_l = t_lsl if t_lsl is not None else pylsl.local_clock()
        locker = QtCore.QMutexLocker(self._lock)
        if s >= self._cap:
            packet_uV = packet_uV[-self._cap:]
            s = self._cap
        self._buf = np.roll(self._buf, -s, axis=0)
        self._buf[-s:] = packet_uV.astype(np.float32)
        # approximate per-sample clocks across the packet
        dt = 1.0 / self._fs if self._fs > 0 else 0.004
        for i in range(s):
            k = s - 1 - i
            self._t_host[-1 - i] = now_h - k * dt
            self._t_lsl[-1 - i] = now_l - k * dt
        self._n += s
        locker.unlock()
        self.new_samples.emit(s)

    def snapshot(self, seconds: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (signals, t_host, t_lsl) copies of the last `seconds` window."""
        m = min(self._cap, max(16, int(self._fs * seconds)))
        locker = QtCore.QMutexLocker(self._lock)
        return self._buf[-m:].copy(), self._t_host[-m:].copy(), self._t_lsl[-m:].copy()

    # -- shared display cache: all tabs filter the same snapshot once per tick --
    def display(self, seconds: float, filt, filtered_view: bool) -> np.ndarray:
        """Snapshot + optional filtering, cached on (total, filt.rev, window).

        Called by every tab each tick; the expensive sosfiltfilt chain runs at
        most once per data arrival instead of once per tab.
        """
        from neurodaq_host.dsp.filters import apply_filters
        key = (self._n, filt.rev if filtered_view else -1, round(seconds, 3), self._fs)
        if key == self._cache_key and self._cache_data is not None:
            return self._cache_data
        sig, _, _ = self.snapshot(seconds=seconds)
        out = apply_filters(sig, self._fs, filt) if filtered_view else sig
        self._cache_key, self._cache_data = key, out
        return out

    @property
    def total(self) -> int:
        return self._n
