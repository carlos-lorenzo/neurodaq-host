"""TCP control client: full command set incl. standby/wakeup/read/write_reg."""
import json
import socket
from PySide6 import QtCore

COMMANDS = ["start", "stop", "reset", "standby", "wakeup", "read_reg",
            "write_reg", "config_global", "config_leadoff", "config_bias",
            "config_channel"]


class TCPClient(QtCore.QThread):
    response = QtCore.Signal(dict)
    status = QtCore.Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sock: socket.socket | None = None
        self.running = True
        self._queue: list = []
        self._lock = QtCore.QMutex()
        self._req_id = 1

    def connect_to(self, host: str, port: int):
        self.disconnect()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((host, port))
            self.status.emit(True, f"Connected to {host}:{port}")
        except Exception as e:
            self.status.emit(False, f"Connection failed: {e}")
            self.sock = None

    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self.status.emit(False, "Disconnected")

    def send(self, cmd: str, params: dict | None = None):
        if cmd not in COMMANDS:
            raise ValueError(f"unknown command {cmd}")
        locker = QtCore.QMutexLocker(self._lock)
        self._queue.append((cmd, params))
        locker.unlock()

    def run(self):
        while self.running:
            if self.sock is None:
                self.msleep(100)
                continue
            locker = QtCore.QMutexLocker(self._lock)
            item = self._queue.pop(0) if self._queue else None
            locker.unlock()
            if not item:
                self.msleep(20)
                continue
            cmd, params = item
            req = {"id": self._req_id, "cmd": cmd}
            if params:
                req["params"] = params
            self._req_id += 1
            try:
                self.sock.sendall((json.dumps(req) + "\n").encode())
                buf = ""
                while self.running:
                    ch = self.sock.recv(1).decode()
                    if not ch:
                        raise ConnectionError("lost")
                    buf += ch
                    if ch == "\n":
                        break
                self.response.emit(json.loads(buf))
            except Exception as e:
                self.status.emit(False, f"TCP error: {e}")
                self.sock = None
