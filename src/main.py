"""Live entry point: instantiate modules, wire them via Qt signals, and run.

Topology (signal -> slot):

   UdpVideoThread.frame_ready ----+--> Recorder.on_frame
                                  +--> GateDetector.on_frame
                                  +--> FpvWindow.on_frame

   CrazyflieLink.pose_ready ------+--> Recorder.on_pose
                                  +--> Planner.on_pose
                                  +--> Controller.on_pose
   CrazyflieLink.connected         -> (status text update)

   GateDetector.detection_ready  ---> PoseEstimator.on_detection
   PoseEstimator.gate_ready      ---> Planner.on_gate
   Planner.waypoint_ready        ---> Controller.on_waypoint

   Controller.setpoint_ready ---+--> CrazyflieLink.set_setpoint
   ManualControl.setpoint_ready -+
   ManualControl.stop_requested ---> CrazyflieLink.send_stop

   FpvWindow key events -> ManualControl.handle_key_press / handle_key_release

Two setpoint sources (Controller and ManualControl) both feed the link's
Latest[Setpoint] latch; whichever wrote last wins on the next radio tick.
Arbitration is the controls team's call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PyQt6 import QtWidgets

from src.control.controller import Controller
from src.control.manual import ManualControl
from src.control.planner import Planner
from src.io.crazyflie_link import CrazyflieLink
from src.io.recorder import Recorder
from src.io.video_stream import UdpVideoThread
from src.perception.gate_detector import GateDetector
from src.perception.pose_estimator import PoseEstimator
from src.ui.fpv_window import FpvWindow


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_dir: Path | None = None) -> tuple[dict, dict]:
    config_dir = config_dir or (REPO_ROOT / "config")
    cfg = yaml.safe_load((config_dir / "default.yaml").read_text())
    cal = yaml.safe_load((config_dir / "calibration.yaml").read_text())
    return cfg, cal


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
    planner = Planner()
    controller = Controller(default_height_m=cfg["control"]["default_height_m"])
    manual = ManualControl(
        speed_mps=cfg["control"]["speed_mps"],
        yaw_rate_dps=cfg["control"]["yaw_rate_dps"],
        default_height_m=cfg["control"]["default_height_m"],
    )

    # IO -> consumers
    video.frame_ready.connect(recorder.on_frame)
    video.frame_ready.connect(detector.on_frame)
    link.pose_ready.connect(recorder.on_pose)
    link.pose_ready.connect(planner.on_pose)
    link.pose_ready.connect(controller.on_pose)

    # Perception chain
    detector.detection_ready.connect(estimator.on_detection)
    estimator.gate_ready.connect(planner.on_gate)

    # Control chain
    planner.waypoint_ready.connect(controller.on_waypoint)
    controller.setpoint_ready.connect(link.set_setpoint)
    manual.setpoint_ready.connect(link.set_setpoint)
    manual.stop_requested.connect(link.send_stop)

    return {
        "video": video,
        "link": link,
        "recorder": recorder,
        "detector": detector,
        "estimator": estimator,
        "planner": planner,
        "controller": controller,
        "manual": manual,
    }


def main() -> int:
    cfg, cal = load_config()
    app = QtWidgets.QApplication(sys.argv)
    sys_ = build_system(cfg, cal)

    win = FpvWindow(sys_["manual"])
    sys_["video"].frame_ready.connect(win.on_frame)
    sys_["link"].connected.connect(lambda uri: win.set_status(f"Connected to {uri}"))
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
