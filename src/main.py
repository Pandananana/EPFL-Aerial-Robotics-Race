"""Entry point: instantiate modules, wire them via Qt signals, and run.

    uv run python -m src.main --source live
    uv run python -m src.main --source webots
    uv run python -m src.main --source replay --recording data/recordings/<id>

Each backend lives under src/io/<mode>/ and exposes a `build_<mode>(cfg)`
helper that returns objects satisfying the VideoSource and DroneLink
protocols in src/io/sources.py. `main` picks one based on --source.

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
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PyQt6 import QtCore, QtWidgets

from src.bus import Latest
from src.control.controller import Controller
from src.control.gates_csv import load_gates_csv
from src.control.manual import ManualControl
from src.control.planner import Planner
from src.control.states.gate_tracker import GateTracker, camera_corners_to_world
from src.io.live import build_live
from src.io.recorder import Recorder
from src.io.replay import build_replay
from src.io.sources import DroneLink, VideoSource
from src.io.webots import build_webots
from src.perception.gate_detector import GateDetector
from src.perception.pose_estimator import PoseEstimator
from src.ui.fpv_window import FpvWindow
from src.ui.gate_debug_plot import GateDebugPlotter

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
    preloaded_gates=None,
    gates_save_path: Path | None = None,
) -> dict:
    """Instantiate and wire every module. Returns the bag of objects so
    the caller can start them and keep them alive."""
    detector = GateDetector(model_name=cfg["perception"]["detector"])
    # Heavy YOLO inference (~65 ms) runs on its own thread so it doesn't
    # block the control loop. Qt makes the cross-thread signal connection
    # queued automatically.
    detector_thread = QtCore.QThread()
    detector_thread.setObjectName("GateDetectorThread")
    detector.moveToThread(detector_thread)
    detector_thread.start()

    estimator = PoseEstimator(
        camera_matrix=np.array(cal["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(cal["dist_coeffs"], dtype=np.float64),
        gate_height_m=cfg["perception"]["gate_height_m"],
        width_search=tuple(cfg["perception"]["gate_width_search_m"]),
    )
    planner = Planner(
        default_height_m=cfg["control"]["default_height_m"],
        preloaded_gates=preloaded_gates,
        gates_save_path=gates_save_path,
    )
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
        recorder = Recorder(
            base_dir=cfg["recording"]["base_dir"],
            pose_log_every_n=cfg["recording"].get("pose_log_every_n", 10),
        )
        video.frame_ready.connect(recorder.on_frame)
        link.pose_ready.connect(recorder.on_pose)
        link.connected.connect(recorder.on_connected)
        detector.detection_ready.connect(recorder.on_detection)
        estimator.gate_ready.connect(recorder.on_gate)
        planner.state_changed.connect(recorder.on_state_changed)
        planner.waypoint_ready.connect(recorder.on_waypoint)
        planner.gate_estimated.connect(recorder.on_gate_estimated)
        planner.measurement_accepted.connect(recorder.on_measurement_accepted)
        controller.setpoint_ready.connect(recorder.on_setpoint)
        manual.setpoint_ready.connect(recorder.on_setpoint)

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
        "detector_thread": detector_thread,
        "estimator": estimator,
        "planner": planner,
        "controller": controller,
        "manual": manual,
    }


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
        "--replay-step", action="store_true",
        help="In replay mode, wait for a keypress in the FPV window before "
             "emitting each recorded pose/frame row.",
    )
    ap.add_argument(
        "--start-frame", type=int, default=1,
        help="Replay only: start playback from this 1-based frame/row number.",
    )
    ap.add_argument(
        "--no-fly", action="store_true",
        help="Connect to the AI-deck and Crazyflie for video + pose but never "
             "arm or send setpoints. Use this when recording calibration / "
             "training frames so the drone stays inert in your hand. Also "
             "suppresses the autonomous mission.",
    )
    ap.add_argument(
        "--true-gates", type=Path, default=None,
        help="Ground-truth gates.csv. Feeds the 3D debug plot in replay, and — "
             "with --race-only — is also loaded as the preloaded gate set. "
             "Replay falls back to <recording>/gates.csv for the debug plot "
             "when this is omitted. In --source webots, the truth gates are "
             "always data/gates/sim_gates.csv (also placed in the sim) and "
             "this flag is ignored.",
    )
    ap.add_argument(
        "--race-only", action="store_true",
        help="Skip the recon lap and drop straight from takeoff into the racing "
             "trajectory. In --source live/replay, requires --true-gates. In "
             "--source webots, uses data/gates/sim_gates.csv automatically.",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Open the 3D gate-debug viewer (true vs. measured gates). Needs a "
             "truth source — either --true-gates or the replay/webots fallback.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg, cal = load_config()
    app = QtWidgets.QApplication(sys.argv[:1])

    if args.race_only and args.true_gates is None and args.source != "webots":
        raise SystemExit("--race-only requires --true-gates <csv>")

    webots_sim_gates: list | None = None
    webots_sim_gates_path: Path | None = None
    if args.source == "live":
        video, link = build_live(cfg, no_fly=args.no_fly)
        record = True
        if args.no_fly:
            print("[main] --no-fly: arming + setpoints disabled; video/pose only.")
    elif args.source == "webots":
        backend, webots_sim_gates, webots_sim_gates_path = build_webots(cfg)
        video, link = backend, backend
        record = False
        # The Webots assignment world has emissive pink-panel gates; the
        # HSV-based pink detector is purpose-built for them and avoids the
        # domain gap that trips up the AI-deck-trained YOLO models.
        cfg["perception"]["detector"] = "pink"
        if args.true_gates is not None:
            print(
                f"[main] --source webots ignores --true-gates; "
                f"using {webots_sim_gates_path} as the truth source.",
                flush=True,
            )
    else:
        if args.recording is None:
            raise SystemExit("--source replay requires --recording <dir>")
        replay = build_replay(
            args.recording,
            args.speed,
            step=args.replay_step,
            start_frame=args.start_frame,
        )
        video, link = replay, replay
        record = False

    preloaded_gates = None
    if args.race_only:
        if args.source == "webots":
            preloaded_gates = webots_sim_gates
            preloaded_src = webots_sim_gates_path
        else:
            preloaded_gates = load_gates_csv(args.true_gates)
            preloaded_src = args.true_gates
        print(
            f"[main] race-only mode: loaded {len(preloaded_gates)} gates from "
            f"{preloaded_src}",
            flush=True,
        )

    gates_save_path = (
        REPO_ROOT / "data" / "gates" / "estimated"
        / f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    )

    active_cal = cal.get(args.source, cal)
    sys_ = build_system(
        cfg, active_cal,
        video=video, link=link, record=record,
        preloaded_gates=preloaded_gates,
        gates_save_path=gates_save_path,
    )

    _latest_pose = Latest()
    sys_["link"].pose_ready.connect(lambda p: _latest_pose.set(p))

    if args.source == "webots":
        debug_truth_csv = webots_sim_gates_path
    else:
        debug_truth_csv = args.true_gates
        if debug_truth_csv is None and args.source == "replay" and args.recording is not None:
            candidate = args.recording / "gates.csv"
            if candidate.exists():
                debug_truth_csv = candidate

    gate_debug_plotter = None
    if args.debug and debug_truth_csv is not None:
        gate_debug_plotter = GateDebugPlotter(truth_csv=debug_truth_csv)
        sys_["link"].pose_ready.connect(gate_debug_plotter.on_pose)
        sys_["estimator"].gate_ready.connect(gate_debug_plotter.on_gate)
        sys_["planner"].gate_estimate_ready.connect(gate_debug_plotter.on_gate_estimate)
        sys_["planner"].race_trajectory_ready.connect(gate_debug_plotter.on_race_trajectory)
        sys_["planner"].state_changed.connect(gate_debug_plotter.on_state_changed)
        print(f"[GATE_DEBUG] plotting true gates from {debug_truth_csv}", flush=True)
        if args.source == "webots":
            print("[GATE_DEBUG] using Webots world frame: x forward, y left, z up", flush=True)
    elif args.debug:
        if args.source == "replay" and args.recording is not None:
            print(
                f"[GATE_DEBUG] disabled: no truth gates found. Pass "
                f"--true-gates <csv>, or add {args.recording / 'gates.csv'}.",
                flush=True,
            )
        else:
            print(
                "[GATE_DEBUG] disabled: pass --true-gates <csv> to open the 3D plotter.",
                flush=True,
            )

    def print_gate3d(g):
        pose = _latest_pose.get()
        bs = ""
        if args.debug:
            count = pose.lighthouse_bs_visible if pose is not None else None
            bs = f" lighthouse_bs={count if count is not None else '?'}"
        if not g.corners_cam_m:
            print(f"[GATE3D] frame={g.frame_seq}{bs} no valid 3D gates", flush=True)
            return

        for i, corners in enumerate(g.corners_cam_m):
            if pose is not None:
                world_pts = camera_corners_to_world(corners, pose)
                center = np.mean(world_pts, axis=0)
                print(
                    f"[GATE3D] frame={g.frame_seq}{bs} gate={i} "
                    f"world_center=[{center[0]:+.2f}, {center[1]:+.2f}, {center[2]:+.2f}]m "
                    f"width={g.widths_m[i]:.2f}m err={g.reprojection_errors_px[i]:.1f}px",
                    flush=True,
                )
            else:
                cam_center = corners.mean(axis=0)
                print(
                    f"[GATE3D] frame={g.frame_seq}{bs} gate={i} "
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

    sys_["planner"].state_changed.connect(
        lambda name: print(f"[FSM] -> {name}", flush=True)
    )
    sys_["planner"].mission_done.connect(
        lambda: print("[FSM] mission done", flush=True)
    )

    # Autonomous mission starts once BOTH the drone link is up and the first
    # camera frame has arrived — otherwise live mode would take off while
    # the AI-deck is still offline / mis-wired. --no-fly is the opt-out,
    # used when the drone is being held for calibration / recording.
    # --race-only skips perception entirely, so the camera feed isn't needed.
    if not (args.source == "live" and args.no_fly):
        ready = {"link": False, "video": args.race_only}

        def _try_start() -> None:
            if ready["link"] and ready["video"]:
                sys_["planner"].start()

        def _on_link_connected(_s) -> None:
            if ready["link"]:
                return
            ready["link"] = True
            if not ready["video"]:
                print("[main] link up; waiting for first camera frame before takeoff.", flush=True)
            _try_start()

        def _on_first_frame(_f) -> None:
            if ready["video"]:
                return
            ready["video"] = True
            if not ready["link"]:
                print("[main] camera up; waiting for drone link before takeoff.", flush=True)
            _try_start()

        sys_["link"].connected.connect(_on_link_connected)
        if not args.race_only:
            sys_["video"].frame_ready.connect(_on_first_frame)
    sys_["planner"].mission_done.connect(sys_["link"].send_stop)

    win = FpvWindow(sys_["manual"])
    if args.source == "replay" and args.replay_step:
        win.key_pressed.connect(sys_["link"].advance)
        print("[REPLAY] step mode: focus the FPV window and press any key to advance.", flush=True)
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
