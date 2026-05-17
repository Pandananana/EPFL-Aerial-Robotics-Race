"""UDP video stream from the AI-deck.

Sends START_MAGIC to the AI-deck to kick off streaming, reads CPX-wrapped
JPEG frames, decodes them, and emits Frame messages via the `frame_ready`
Qt signal. The signal is auto-queued across thread boundaries, so any
QObject slot can subscribe safely.

The receiver is a QThread so it doesn't block the Qt event loop. If the
laptop OS already has LOCAL_PORT bound (e.g. previous run still alive),
the bind here will fail noisily — kill the old process or pick a new port
in config/default.yaml.
"""

from __future__ import annotations

import contextlib
import os
import socket
import struct
import time

import cv2
import numpy as np
from PyQt6 import QtCore

from src.messages import Frame


@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


class UdpVideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)  # Frame

    CPX_HEADER_SIZE = 4
    IMG_HEADER_MAGIC = 0xBC
    IMG_HEADER_SIZE = 11

    def __init__(
        self,
        *,
        aideck_ip: str,
        aideck_port: int,
        local_port: int,
        start_magic: bytes,
        width: int,
        height: int,
        min_jpeg_bytes: int,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._aideck_ip = aideck_ip
        self._aideck_port = aideck_port
        self._local_port = local_port
        self._start_magic = start_magic
        self._width = width
        self._height = height
        self._min_jpeg_bytes = min_jpeg_bytes
        self._seq = 0

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(("0.0.0.0", self._local_port))
        sock.sendto(self._start_magic, (self._aideck_ip, self._aideck_port))

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while True:
            data, _ = sock.recvfrom(2048)
            if len(data) < self.CPX_HEADER_SIZE:
                continue
            payload = data[self.CPX_HEADER_SIZE :]

            if (
                len(payload) >= self.IMG_HEADER_SIZE
                and payload[0] == self.IMG_HEADER_MAGIC
            ):
                _, w, h, _, _, size = struct.unpack(
                    "<BHHBBI", payload[: self.IMG_HEADER_SIZE]
                )
                if w == self._width and h == self._height and 0 < size < 65536:
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                self._decode_and_emit(buffer)
                receiving = False

    def _decode_and_emit(self, buffer: bytearray) -> None:
        soi = buffer.find(b"\xff\xd8")
        eoi = buffer.rfind(b"\xff\xd9")
        if soi < 0 or eoi <= soi:
            return
        jpeg_len = eoi + 2 - soi
        if jpeg_len < self._min_jpeg_bytes:
            return
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (self._height, self._width):
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._seq += 1
        self.frame_ready.emit(Frame(timestamp=time.time(), seq=self._seq, image=img))
