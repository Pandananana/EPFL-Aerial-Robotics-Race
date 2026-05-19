#!/usr/bin/env python3
"""
Crazyflie FPV example.

What happens when this script runs:
  * Fly commands are sent to the Crazyflie with cflib over Crazyradio using URI.
    Modify URI below to match your Crazyflie setup.
  * Vision runs separately over WiFi: the laptop must be connected to the
    AI-deck WiFi. The AI-deck sends camera frames as a UDP stream to LOCAL_PORT.
  * Each frame is received as grayscale JPEG data, decoded with OpenCV, and
    shown in a PyQt window. This example expects 324 x 244 images.
  * The UDP socket and Qt event queue can briefly cache received data. If the
    network or GUI falls behind, displayed frames may be delayed instead of
    always showing the newest camera frame immediately. The images may also be
    slightly distorted, though they looked fine in our experimental environment.
  * If the script cannot start because LOCAL_PORT is already in use, a previous
    run may still be alive. See the FAQ for how to free the port on your OS.

Keys:  arrows = pitch/roll,  A/D = yaw,  W/S = up/down,  Space = stop.
"""
import contextlib
import csv
import datetime
import os
import socket
import struct
import sys
import threading
import time

import cflib.crtp
import cv2
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from PyQt6 import QtCore, QtGui, QtWidgets


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

# --- Configure these for your setup ---
URI         = 'radio://0/80/2M/E7E7E7E718'
AIDECK_IP   = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT  = 5001
START_MAGIC = b'FER'
SPEED       = 0.6

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
IMG_WIDTH        = 324
IMG_HEIGHT       = 244
MIN_JPEG_BYTES   = 5000


class UdpVideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while True:
            data, _ = sock.recvfrom(2048)
            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack(
                    '<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
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

    def _decode_and_emit(self, buffer):
        soi = buffer.find(b'\xff\xd8')
        eoi = buffer.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(img)


class FPVWindow(QtWidgets.QWidget):
    _connected_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crazyflie FPV')

        self.image_label = QtWidgets.QLabel('Waiting for video...')
        self.status_label = QtWidgets.QLabel(f'Connecting to {URI}...')
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)

        # Recording setup
        self._frame_count = 0
        self._state_lock = threading.Lock()
        self._latest_state = {'x': 0.0, 'y': 0.0, 'z': 0.0,
                              'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_dir = os.path.join('recordings', run_id)
        os.makedirs(self._save_dir, exist_ok=True)
        self._csv_file = open(os.path.join(self._save_dir, 'measurements.csv'), 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(['timestamp', 'image', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])

        self.video = UdpVideoThread(self)
        self.video.frame_ready.connect(self._show_frame)
        self.video.start()

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(rw_cache='cache')
        self._connected_signal.connect(self._on_connected)
        self.cf.connected.add_callback(self._connected_signal.emit)
        self.cf.open_link(URI)

    def _on_connected(self, uri):
        self.status_label.setText(f'Connected to {uri}')
        self._start_logging()

    def _start_logging(self):
        lg = LogConfig(name='State', period_in_ms=10)
        lg.add_variable('stateEstimate.x', 'float')
        lg.add_variable('stateEstimate.y', 'float')
        lg.add_variable('stateEstimate.z', 'float')
        lg.add_variable('stabilizer.roll', 'float')
        lg.add_variable('stabilizer.pitch', 'float')
        lg.add_variable('stabilizer.yaw', 'float')
        self.cf.log.add_config(lg)
        lg.data_received_cb.add_callback(self._log_callback)
        lg.start()

    def _log_callback(self, timestamp, data, logconf):
        with self._state_lock:
            self._latest_state['x'] = data['stateEstimate.x']
            self._latest_state['y'] = data['stateEstimate.y']
            self._latest_state['z'] = data['stateEstimate.z']
            self._latest_state['roll'] = data['stabilizer.roll']
            self._latest_state['pitch'] = data['stabilizer.pitch']
            self._latest_state['yaw'] = data['stabilizer.yaw']

    def _show_frame(self, img):
        self._save_frame(img)
        if img.ndim == 2:
            h, w = img.shape
            qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = img.shape
            qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.scaled(w * 2, h * 2)))

    def _save_frame(self, img):
        self._frame_count += 1
        filename = f'img_{self._frame_count:06d}.png'
        path = os.path.join(self._save_dir, filename)
        cv2.imwrite(path, img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with self._state_lock:
            s = dict(self._latest_state)
        self._csv.writerow([time.time(), filename, s['x'], s['y'], s['z'],
                            s['roll'], s['pitch'], s['yaw']])
        self._csv_file.flush()

    def closeEvent(self, event):
        self.cf.close_link()
        self._csv_file.close()
        print(f'Saved {self._frame_count} frames to {self._save_dir}')
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    sys.exit(app.exec())