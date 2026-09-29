"""Tiny synthesized sound cues (no audio assets needed).

Two cues: a short high blip for target flashes, a two-tone chime when the
target letter changes. Falls back to QApplication.beep() when QtMultimedia
can't open an audio device (e.g. headless CI).
"""
import math
import os
import struct
import tempfile
import wave

from PySide6 import QtWidgets


def _tone_wav(path: str, freqs: list[float], ms_each: int = 120,
              rate: int = 22050, volume: float = 0.4):
    frames = b""
    for f in freqs:
        n = int(rate * ms_each / 1000.0)
        for i in range(n):
            # 10 ms raised-cosine edges to avoid clicks
            edge = min(i, n - i, int(rate * 0.01)) / max(1, int(rate * 0.01))
            s = volume * edge * math.sin(2 * math.pi * f * i / rate)
            frames += struct.pack("<h", int(32767 * max(-1.0, min(1.0, s))))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


class SoundCue:
    def __init__(self):
        self.enabled = True
        self._ok = False
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect
            d = os.path.join(tempfile.gettempdir(), "neurodaq_sounds")
            os.makedirs(d, exist_ok=True)
            self._blip = os.path.join(d, "target_blip.wav")
            self._chime = os.path.join(d, "new_letter.wav")
            if not os.path.exists(self._blip):
                _tone_wav(self._blip, [880.0], ms_each=120)
            if not os.path.exists(self._chime):
                _tone_wav(self._chime, [660.0, 990.0], ms_each=140)
            self._blip_fx = QSoundEffect()
            self._blip_fx.setSource(QUrl.fromLocalFile(self._blip))
            self._chime_fx = QSoundEffect()
            self._chime_fx.setSource(QUrl.fromLocalFile(self._chime))
            self._ok = True
        except Exception:
            self._ok = False

    def _beep(self):
        try:
            QtWidgets.QApplication.beep()
        except Exception:
            pass

    def target_flash(self):
        """Short blip: the flashed row/column contained the target."""
        if not self.enabled:
            return
        if self._ok:
            self._blip_fx.play()
        else:
            self._beep()

    def new_letter(self):
        """Two-tone chime: focus moves to a different letter."""
        if not self.enabled:
            return
        if self._ok:
            self._chime_fx.play()
        else:
            self._beep()
