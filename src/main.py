"""Entry point: instantiate modules, wire them via Qt signals, and run.

    uv run python -m src.main --source live
    uv run python -m src.main --source webots --ibvs --autostart
    uv run python -m src.main --source replay --recording data/recordings/<id>

Each backend lives under src/io/<mode>/ and exposes a `build_<mode>(cfg)`
helper that returns objects satisfying the VideoSource and DroneLink
protocols in src/io/sources.py. `main` picks one based on --source.

There are two interchangeable control stacks; `--ibvs` selects the second.

Default (3D estimator -> Planner -> Controller):

   video.frame_ready -------------+--> GateDetector.on_frame -> FpvWindow / Recorder
   GateDetector.detection_ready  ---> PoseEstimator.on_detection
   PoseEstimator.gate_ready      ---> Planner.on_gate
   link.pose_ready ---------------+--> Planner.on_pose / Controller.on_pose
   Planner.waypoint_ready        ---> Controller.on_waypoint
   Controller.setpoint_ready    ---> link.set_setpoint

IBVS (`--ibvs`, image-plane only, no PoseEstimator or Controller):

   video.frame_ready             ---> IBVSPlanner.on_frame      (image shape)
   GateDetector.detection_ready  ---> IBVSPlanner.on_detection  (FSM + control law)
   link.pose_ready               ---> IBVSPlanner.on_pose       (takeoff only)
   IBVSPlanner.setpoint_ready    ---> link.set_setpoint

ManualControl always shares the link's setpoint sink with whichever planner
is active; whichever wrote last wins on the next radio tick. In replay mode
the link's set_setpoint / send_stop are no-ops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PyQt6 import QtCore, QtWidgets

from src.control.controller import Controller
from src.control.ibvs import IBVSPlanner
from src.control.manual import ManualControl
from src.control.planner import Planner
from src.bus import Latest
from src.control.states.gate_tracker import GateTracker, camera_corners_to_world
from src.io.live import build_live
from src.io.recorder import Recorder
from src.io.replay import build_replay
from src.io.sources import DroneLink, VideoSource
from src.io.webots import build_webots
from src.perception.gate_detector import GateDetector
from src.perception.pose_estimator import PoseEstimator
from src.ui.fpv_window import FpvWindow

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_dir: Path | None = None) -> tuple[dict, dict]:
    config_dir = config_dir or (REPO_ROOT / "config")
    cfg = yaml.safe_load((config_dir / "default.yaml").read_text())
    cal = yaml.safe_load((config_dir / "calibration.yaml").read_text())
    return cfg, cal


def build_system(
    cfg: dict,
    cal: dict,
    *,
    video: VideoSource,
    link: DroneLink,
    record: bool = True,
    ibvs: bool = False,
    n_gates: int = Planner.DEFAULT_GATE_COUNT,
) -> dict:
    """Instantiate and wire every module. Returns the bag of objects so
    the caller can start them and keep them alive.

    `ibvs=True` swaps the 3D planner + controller chain for the image-based
    visual servoing planner, which consumes 2D detections directly and
    emits Setpoint without ever lifting corners to world coordinates.
    """
    detector = GateDetector(model_name=cfg["perception"]["detector"])
    # Heavy YOLO inference (~65 ms) runs on its own thread so it doesn't
    # block the control loop. Qt makes the cross-thread signal connection
    # queued automatically.
    detector_thread = QtCore.QThread()
    detector_thread.setObjectName("GateDetectorThread")
    detector.moveToThread(detector_thread)
    detector_thread.start()

    manual = ManualControl(
        speed_mps=cfg["control"]["speed_mps"],
        yaw_rate_dps=cfg["control"]["yaw_rate_dps"],
        default_height_m=cfg["control"]["default_height_m"],
    )

    video.frame_ready.connect(detector.on_frame)

    recorder: Recorder | None = None
    if record:
        recorder = Recorder(
            base_dir=cfg["recording"]["base_dir"],
            pose_log_every_n=cfg["recording"].get("pose_log_every_n", 10),
        )
        video.frame_ready.connect(recorder.on_frame)
        link.pose_ready.connect(recorder.on_pose)
        link.connected.connect(recorder.on_connected)
        detector.detection_ready.connect(recorder.on_detection)
        manual.setpoint_ready.connect(recorder.on_setpoint)

    manual.setpoint_ready.connect(link.set_setpoint)
    manual.stop_requested.connect(link.send_stop)

    out: dict = {
        "video": video,
        "link": link,
        "recorder": recorder,
        "detector": detector,
        "detector_thread": detector_thread,
        "manual": manual,
    }

    if ibvs:
        ibvs_planner = IBVSPlanner(
            default_height_m=cfg["control"]["default_height_m"],
            n_gates=n_gates,
        )
        video.frame_ready.connect(ibvs_planner.on_frame)
        link.pose_ready.connect(ibvs_planner.on_pose)
        detector.detection_ready.connect(ibvs_planner.on_detection)
        ibvs_planner.setpoint_ready.connect(link.set_setpoint)
        if recorder is not None:
            ibvs_planner.setpoint_ready.connect(recorder.on_setpoint)
            ibvs_planner.phase_changed.connect(recorder.on_state_changed)
        out["planner"] = ibvs_planner
        out["estimator"] = None
        out["controller"] = None
        return out

    estimator = PoseEstimator(
        camera_matrix=np.array(cal["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(cal["dist_coeffs"], dtype=np.float64),
        gate_height_m=cfg["perception"]["gate_height_m"],
        width_search=tuple(cfg["perception"]["gate_width_search_m"]),
    )
    planner = Planner(default_height_m=cfg["control"]["default_height_m"])
    controller = Controller(default_height_m=cfg["control"]["default_height_m"])

    link.pose_ready.connect(planner.on_pose)
    link.pose_ready.connect(controller.on_pose)

    if recorder is not None:
        estimator.gate_ready.connect(recorder.on_gate)
        planner.state_changed.connect(recorder.on_state_changed)
        planner.waypoint_ready.connect(recorder.on_waypoint)
        controller.setpoint_ready.connect(recorder.on_setpoint)

    # Perception chain
    detector.detection_ready.connect(estimator.on_detection)
    estimator.gate_ready.connect(planner.on_gate)

    # Control chain
    planner.waypoint_ready.connect(controller.on_waypoint)
    controller.setpoint_ready.connect(link.set_setpoint)

    out["estimator"] = estimator
    out["planner"] = planner
    out["controller"] = controller
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the integrated drone-race system.")
    ap.add_argument(
        "--source", choices=["live", "replay", "webots"], default="live",
        help="IO backend: 'live' connects to the AI-deck and Crazyflie; "
             "'replay' plays back a recording (controller setpoints are dropped); "
             "'webots' attaches to a running Webots simulation as an extern controller.",
    )
    ap.add_argument(
        "--recording", type=Path, default=None,
        help="Recording directory for --source replay (e.g. data/recordings/<id>).",
    )
    ap.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed multiplier (default 1.0).",
    )
    ap.add_argument(
        "--autostart", action="store_true",
        help="Kick off the autonomous mission (takeoff -> recon -> race -> land) "
             "as soon as the drone link is connected. Off by default so manual "
             "control stays in charge.",
    )
    ap.add_argument(
        "--no-fly", action="store_true",
        help="Connect to the AI-deck and Crazyflie for video + pose but never "
             "arm or send setpoints. Use this when recording calibration / "
             "training frames so the drone stays inert in your hand.",
    )
    ap.add_argument(
        "--ibvs", action="store_true",
        help="Use image-based visual servoing instead of the 3D-estimator "
             "planner+controller chain. Setpoints are derived directly from "
             "the locked gate's pixel-space corners.",
    )
    ap.add_argument(
        "--n-gates", type=int, default=Planner.DEFAULT_GATE_COUNT,
        help="Number of gates the mission expects (used by both planners).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg, cal = load_config()
    app = QtWidgets.QApplication(sys.argv[:1])

    if args.source == "live":
        video, link = build_live(cfg, no_fly=args.no_fly)
        record = True
        if args.no_fly:
            print("[main] --no-fly: arming + setpoints disabled; video/pose only.")
            if args.autostart:
                print("[main] --no-fly overrides --autostart; mission will not run.")
                args.autostart = False
    elif args.source == "webots":
        backend = build_webots(cfg)
        video, link = backend, backend
        record = True
        # The Webots assignment world has emissive pink-panel gates; the
        # HSV-based pink detector is purpose-built for them and avoids the
        # domain gap that trips up the AI-deck-trained YOLO models.
        cfg["perception"]["detector"] = "pink"
    else:
        if args.recording is None:
            raise SystemExit("--source replay requires --recording <dir>")
        replay = build_replay(args.recording, args.speed)
        video, link = replay, replay
        record = False

    sys_ = build_system(
        cfg, cal, video=video, link=link, record=record,
        ibvs=args.ibvs, n_gates=args.n_gates,
    )

    _latest_pose = Latest()
    sys_["link"].pose_ready.connect(lambda p: _latest_pose.set(p))

    if sys_["estimator"] is not None:
        def print_gate3d(g):
            if not g.corners_cam_m:
                print(f"[GATE3D] frame={g.frame_seq} no valid 3D gates", flush=True)
                return

            pose = _latest_pose.get()
            for i, corners in enumerate(g.corners_cam_m):
                if pose is not None:
                    world_pts = camera_corners_to_world(corners, pose)
                    center = np.mean(world_pts, axis=0)
                    print(
                        f"[GATE3D] frame={g.frame_seq} gate={i} "
                        f"world_center=[{center[0]:+.2f}, {center[1]:+.2f}, {center[2]:+.2f}]m "
                        f"width={g.widths_m[i]:.2f}m err={g.reprojection_errors_px[i]:.1f}px",
                        flush=True,
                    )
                else:
                    cam_center = corners.mean(axis=0)
                    print(
                        f"[GATE3D] frame={g.frame_seq} gate={i} "
                        f"cam_center=[{cam_center[0]:+.2f}, {cam_center[1]:+.2f}, {cam_center[2]:+.2f}]m "
                        f"width={g.widths_m[i]:.2f}m err={g.reprojection_errors_px[i]:.1f}px",
                        flush=True,
                    )

        sys_["estimator"].gate_ready.connect(print_gate3d)

        if args.source == "replay":
            debug_tracker = GateTracker()
            debug_poses = {}
            sys_["link"].pose_ready.connect(lambda p: debug_poses.__setitem__(p.timestamp, p))

            def update_debug_tracker(g):
                pose = debug_poses.get(g.timestamp)
                if pose is not None:
                    print(
                        f"[POSE_DEBUG] frame={g.frame_seq} "
                        f"pos=[{pose.x:+.2f}, {pose.y:+.2f}, {pose.z:+.2f}]m "
                        f"rpy=[{pose.roll:+.1f}, {pose.pitch:+.1f}, {pose.yaw:+.1f}]deg",
                        flush=True,
                    )
                    debug_tracker.update(g, pose)

            sys_["estimator"].gate_ready.connect(update_debug_tracker)

    if args.ibvs:
        sys_["planner"].phase_changed.connect(
            lambda name: print(f"[IBVS] -> {name}", flush=True)
        )
        sys_["planner"].locked_track_changed.connect(
            lambda tid: print(f"[IBVS] locked track = {tid}", flush=True)
        )
    else:
        sys_["planner"].state_changed.connect(
            lambda name: print(f"[FSM] -> {name}", flush=True)
        )
    sys_["planner"].mission_done.connect(
        lambda: print("[FSM] mission done", flush=True)
    )

    if args.autostart:
        sys_["link"].connected.connect(lambda _s: sys_["planner"].start())
    # Cut motors once the FSM reaches the terminal state — regardless of
    # whether the mission was autostarted or kicked off manually later.
    sys_["planner"].mission_done.connect(sys_["link"].send_stop)

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
        sys_["detector_thread"].quit()
        sys_["detector_thread"].wait()


if __name__ == "__main__":
    raise SystemExit(main())
