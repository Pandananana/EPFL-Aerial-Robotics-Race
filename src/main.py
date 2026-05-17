"""Live entry point: instantiate modules, wire them via Qt signals, and run.

Topology (signal -> slot):

   UdpVideoThread.frame_ready ----+--> Recorder.on_frame
                                  +--> GateDetector.on_frame
                                  +--> FpvWindow._show_frame

   CrazyflieLink.pose_ready ------+--> Recorder.on_pose
                                  +--> WaypointPlanner.on_pose
                                  +--> Controller.on_pose

   GateDetector.detection_ready  ---> PoseEstimator.on_detection
   PoseEstimator.gate_ready      ---> WaypointPlanner.on_gate
   WaypointPlanner.waypoint_ready --> Controller.on_waypoint
   Controller.setpoint_ready     ---> CrazyflieLink.set_setpoint

The FPV window (keyboard control + image display) lives here for now —
move it out once `Controller` is real and manual control is no longer the
primary input path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PyQt6 import QtCore, QtGui, QtWidgets

from src.io.crazyflie_link import CrazyflieLink
from src.io.recorder import Recorder
from src.io.video_stream import UdpVideoThread
from src.messages import Frame, Setpoint
from src.perception.gate_detector import GateDetector
from src.perception.pose_estimator import PoseEstimator
from src.planning.controller import Controller
from src.planning.waypoint_planner import WaypointPlanner


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_dir: Path | None = None) -> tuple[dict, dict]:
    config_dir = config_dir or (REPO_ROOT / "config")
    cfg = yaml.safe_load((config_dir / "default.yaml").read_text())
    cal = yaml.safe_load((config_dir / "calibration.yaml").read_text())
    return cfg, cal


class FpvWindow(QtWidgets.QWidget):
    """Live FPV display plus keyboard manual-control fallback.

    Arrow keys = pitch/roll, A/D = yaw, W/S = height, Space = stop. Manual
    keypresses publish a Setpoint directly to CrazyflieLink, bypassing the
    Controller — handy while the real controller is still being built.
    """

    SPEED = 0.6
    YAW_RATE = 70.0

    def __init__(self, link: CrazyflieLink):
        super().__init__()
        self.setWindowTitle("Crazyflie FPV")
        self._link = link

        self.image_label = QtWidgets.QLabel("Waiting for video...")
        self.status_label = QtWidgets.QLabel("Initialising...")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)

        self._hover = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "height": 0.3}
        self._held: set[int] = set()
        self._link.connected.connect(
            lambda uri: self.status_label.setText(f"Connected to {uri}")
        )

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        img = frame.image
        h, w = img.shape[:2]
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.scaled(w * 2, h * 2)))

    def _publish_setpoint(self) -> None:
        self._link.set_setpoint(Setpoint(
            vx=self._hover["vx"],
            vy=self._hover["vy"],
            yaw_rate=self._hover["yaw_rate"],
            height=self._hover["height"],
        ))

    def _update_velocity(self) -> None:
        K = QtCore.Qt.Key
        vx = (K.Key_Up in self._held) * self.SPEED - (K.Key_Down in self._held) * self.SPEED
        vy = (K.Key_Left in self._held) * self.SPEED - (K.Key_Right in self._held) * self.SPEED
        yaw_rate = (K.Key_D in self._held) * self.YAW_RATE - (K.Key_A in self._held) * self.YAW_RATE
        self._hover["vx"], self._hover["vy"], self._hover["yaw_rate"] = vx, vy, yaw_rate
        self._publish_setpoint()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Space:
            self._link.cf.commander.send_stop_setpoint()
            return
        if k == QtCore.Qt.Key.Key_W:
            self._hover["height"] += 0.1
            self._publish_setpoint()
            return
        if k == QtCore.Qt.Key.Key_S:
            self._hover["height"] -= 0.1
            self._publish_setpoint()
            return
        self._held.add(k)
        self._update_velocity()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        self._held.discard(event.key())
        self._update_velocity()


def build_system(cfg: dict, cal: dict) -> dict:
    """Instantiate and wire every module. Returns the bag of objects so the
    caller (main / a test / a script) can start them and keep them alive."""
    video = UdpVideoThread(
        aideck_ip=cfg["network"]["aideck_ip"],
        aideck_port=cfg["network"]["aideck_port"],
        local_port=cfg["network"]["local_port"],
        start_magic=cfg["network"]["start_magic"].encode(),
        width=cfg["video"]["width"],
        height=cfg["video"]["height"],
        min_jpeg_bytes=cfg["video"]["min_jpeg_bytes"],
    )
    link = CrazyflieLink(
        uri=cfg["crazyflie"]["uri"],
        cache_dir=cfg["crazyflie"]["cache_dir"],
        setpoint_rate_hz=cfg["control"]["setpoint_rate_hz"],
    )
    recorder = Recorder(base_dir=cfg["recording"]["base_dir"])
    detector = GateDetector(model_name=cfg["perception"]["detector"])
    estimator = PoseEstimator(
        camera_matrix=np.array(cal["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(cal["dist_coeffs"], dtype=np.float64),
        gate_height_m=cfg["perception"]["gate_height_m"],
        width_search=tuple(cfg["perception"]["gate_width_search_m"]),
    )
    planner = WaypointPlanner()
    controller = Controller(default_height_m=cfg["control"]["default_height_m"])

    # Wiring.
    video.frame_ready.connect(recorder.on_frame)
    video.frame_ready.connect(detector.on_frame)
    link.pose_ready.connect(recorder.on_pose)
    link.pose_ready.connect(planner.on_pose)
    link.pose_ready.connect(controller.on_pose)
    detector.detection_ready.connect(estimator.on_detection)
    estimator.gate_ready.connect(planner.on_gate)
    planner.waypoint_ready.connect(controller.on_waypoint)
    controller.setpoint_ready.connect(link.set_setpoint)

    return {
        "video": video,
        "link": link,
        "recorder": recorder,
        "detector": detector,
        "estimator": estimator,
        "planner": planner,
        "controller": controller,
    }


def main() -> int:
    cfg, cal = load_config()
    app = QtWidgets.QApplication(sys.argv)
    sys_ = build_system(cfg, cal)
    win = FpvWindow(sys_["link"])
    sys_["video"].frame_ready.connect(win.on_frame)
    win.show()
    sys_["video"].start()
    sys_["link"].open()
    try:
        return app.exec()
    finally:
        sys_["link"].close()
        sys_["recorder"].close()


if __name__ == "__main__":
    raise SystemExit(main())
