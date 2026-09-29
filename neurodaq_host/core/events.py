"""Unified event/marker bus: P300 stimulators, CV processes, keys, TCP inject."""
from __future__ import annotations
from dataclasses import dataclass
from PySide6 import QtCore


@dataclass
class Event:
    sample_idx: int   # nearest Hub sample at insert time
    t_lsl: float      # LSL local_clock for cross-stream alignment
    t_host: float
    kind: str         # 'lsl_marker' | 'tcp' | 'manual' | 'paradigm'
    label: str
    value: float = 0.0


class EventBus(QtCore.QObject):
    event_added = QtCore.Signal(object)  # Event

    def __init__(self, parent=None):
        super().__init__(parent)
        self._events: list[Event] = []
        self._lock = QtCore.QMutex()

    def push(self, ev: Event):
        locker = QtCore.QMutexLocker(self._lock)
        self._events.append(ev)
        locker.unlock()
        self.event_added.emit(ev)

    def all(self) -> list[Event]:
        locker = QtCore.QMutexLocker(self._lock)
        out = list(self._events)
        locker.unlock()
        return out

    def clear(self):
        locker = QtCore.QMutexLocker(self._lock)
        self._events.clear()
        locker.unlock()
