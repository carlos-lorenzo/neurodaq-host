"""Tab plugin contract: every visualisation is addable/removable/reorderable."""
from __future__ import annotations
from abc import ABC, abstractmethod
from PySide6 import QtWidgets


class TabPlugin(ABC):
    id: str = "base"
    title: str = "Base"
    tick_always: bool = False  # True: tick even with no new data (wall-clock logic)

    def __init__(self, ctx):
        self.ctx = ctx  # AppContext: hub/bus/config/tcp/profile refs
        self.widget: QtWidgets.QWidget | None = None

    @abstractmethod
    def build(self) -> QtWidgets.QWidget:
        ...

    def on_tick(self):
        """Called at ~20 Hz by MainWindow; keep cheap, guard visibility."""

    def panel_state(self) -> dict:
        """Sub-layout state (toggles, splits) saved into the profile."""
        return {}

    def restore_state(self, state: dict):
        """Apply a saved panel_state dict. Called once after build()."""

    def teardown(self):
        pass


class AppContext:
    def __init__(self, hub, bus, filt, dev, tcp, profile: dict):
        self.hub = hub
        self.bus = bus
        self.filt = filt
        self.dev = dev
        self.tcp = tcp
        self.profile = profile
        self.electrode_mapping: list[str] = [c.electrode for c in dev.channels]
