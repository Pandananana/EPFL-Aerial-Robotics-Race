"""Live entry point: instantiate modules, wire them via Qt signals, and run.

The IO backends are pluggable — see src/io/sources.py for the VideoSource
and DroneLink protocols. `build_system` takes them as arguments; `main`
picks one based on the --source flag.

Topology (signal -> slot):

   video.frame_ready -------------+--> Recorder.on_frame  (live only)
                                  +--> GateDetector.on_frame
                                  +--> FpvWindow.on_frame

   link.pose_ready ---------------+--> Recorder.on_pose   (live only)
                                  +--> Planner.on_pose
                                  +--> Controller.on_pose
   link.connected ----------------+--> FpvWindow.set_status

   GateDetector.detection_ready --+--> PoseEstimator.on_detection
                                  +--> FpvWindow.on_detection
   PoseEstimator.gate_ready      ---> Planner.on_gate
   Planner.waypoint_ready        ---> Controller.on_waypoint

   Controller.setpoint_ready ---+--> link.set_setpoint
   ManualControl.setpoint_ready -+
   ManualControl.stop_requested ---> link.send_stop

   FpvWindow key events -> ManualControl.handle_key_press / handle_key_release

Two setpoint sources (Controller and ManualControl) both feed the link's
sink; whichever wrote last wins on the next radio tick. Arbitration is
the controls team's call. In replay mode the link's set_setpoint /
send_stop are no-ops — there is no drone to command.
"""

from __future__ import annotations

import argparse
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
from src.io.replay import ReplayThread
from src.io.sources import DroneLink, VideoSource
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


def build_live_backends(cfg: dict) -> tuple[VideoSource, DroneLink]:
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
    return video, link


def build_replay_backend(recording: Path, speed: float) -> ReplayThread:
    """Single ReplayThread serves as both VideoSource and DroneLink."""
    return ReplayThread(recording, speed=speed)


def build_system(
    cfg: dict,
    cal: dict,
    *,
    video: VideoSource,
    link: DroneLink,
    record: bool = True,
) -> dict:
    """Instantiate and wire every module. Returns the bag of objects so
    the caller can start them and keep them alive."""
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
    video.frame_ready.connect(detector.on_frame)
    link.pose_ready.connect(planner.on_pose)
    link.pose_ready.connect(controller.on_pose)

    recorder: Recorder | None = None
    if record:
        recorder = Recorder(base_dir=cfg["recording"]["base_dir"])
        video.frame_ready.connect(recorder.on_frame)
        link.pose_ready.connect(recorder.on_pose)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the integrated drone-race system.")
    ap.add_argument(
        "--source", choices=["live", "replay"], default="live",
        help="IO backend: 'live' connects to the AI-deck and Crazyflie; "
             "'replay' plays back a recording (controller setpoints are dropped).",
    )
    ap.add_argument(
        "--recording", type=Path, default=None,
        help="Recording directory for --source replay (e.g. data/recordings/<id>).",
    )
    ap.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed multiplier (default 1.0).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg, cal = load_config()
    app = QtWidgets.QApplication(sys.argv[:1])

    if args.source == "live":
        video, link = build_live_backends(cfg)
        record = True
    else:
        if args.recording is None:
            raise SystemExit("--source replay requires --recording <dir>")
        replay = build_replay_backend(args.recording, args.speed)
        video, link = replay, replay
        record = False

    sys_ = build_system(cfg, cal, video=video, link=link, record=record)

    win = FpvWindow(sys_["manual"])
    sys_["video"].frame_ready.connect(win.on_frame)
    sys_["detector"].detection_ready.connect(win.on_detection)
    sys_["link"].connected.connect(lambda s: win.set_status(f"Connected to {s}"))
    win.show()

    sys_["video"].start()
    sys_["link"].open()
    try:
        return app.exec()
    finally:
        sys_["link"].close()
        if sys_["recorder"] is not None:
            sys_["recorder"].close()


if __name__ == "__main__":
    raise SystemExit(main())
