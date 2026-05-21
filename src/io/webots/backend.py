"""Webots extern-controller backend.

A single class that implements both `VideoSource` and `DroneLink` (see
src/io/sources.py), so it drops into src/main.py the same way `ReplayThread`
does. The drone in the .wbt world must have `controller "<extern>"` so
Webots waits for this Python process to attach.

Per-tick (`basicTimeStep`, 8 ms = 125 Hz in the bundled world):

  * Read GPS + IMU + gyro. Emit DronePose throttled to `pose_rate_hz`
    (rpy in degrees, like the real Crazyflie log stream).
  * Read the simulated camera (BGRA at the camera's configured size) and
    forward it to perception as native BGR — we no longer convert to
    grayscale or resize to the AI-deck shape, because the sim doesn't
    emulate the real camera anyway and the pink-panel detector needs the
    colour signal.
  * If a hover Setpoint has been pushed (vx, vy, yaw_rate, height — the
    cflib hover-commander surface), run the cascaded controller in
    `pid.py` to get motor PWMs and apply them. Without a setpoint or
    after `send_stop`, motors are held at zero (drone sits on its pad).

The emission rates match the real Crazyflie (AI-deck JPEG stream ~3 fps,
log block 100 Hz). The PID still runs every tick so attitude control stays
stable; only what is forwarded into the Qt pipeline is throttled. Emitting
at the full 125 Hz floods the detector and hangs the sim.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6 import QtCore

from src.bus import Latest
from src.control.gates_csv import RecordedGate
from src.io.webots.gate_placer import place_gates
from src.io.webots.launcher import ROBOT_NAME
from src.io.webots.pid import HoverController, _Sensors
from src.messages import DronePose, Frame, Setpoint


class WebotsBackend(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)  # Frame
    pose_ready = QtCore.pyqtSignal(object)  # DronePose
    connected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        camera_fps: float,
        pose_rate_hz: float,
        sim_gates: list[RecordedGate],
        sim_gates_path: Path | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._camera_period = 1.0 / camera_fps
        self._pose_period = 1.0 / pose_rate_hz
        self._sim_gates = sim_gates
        self._sim_gates_path = sim_gates_path
        self._setpoint: Latest[Setpoint] = Latest()
        self._stopped = False
        self._seq = 0

    # --- DroneLink interface (open is a no-op; the QThread `start()` from
    #     VideoSource drives everything). ---

    def open(self) -> None:
        pass

    def close(self) -> None:
        self.requestInterruption()
        self.wait()

    @QtCore.pyqtSlot(object)
    def set_setpoint(self, sp: Setpoint) -> None:
        self._stopped = False
        self._setpoint.set(sp)

    @QtCore.pyqtSlot()
    def send_stop(self) -> None:
        self._stopped = True

    # --- QThread main loop ---

    def run(self) -> None:
        # Lazy import: only fails if --source webots is actually used. Use
        # Supervisor (not Robot) so we can move the gate nodes to match the
        # ground-truth CSV before flight starts. Requires `supervisor TRUE`
        # on the Crazyflie node in race.wbt.
        from controller import Supervisor  # type: ignore[import-not-found]

        robot = Supervisor()
        timestep = int(robot.getBasicTimeStep())
        dt = timestep / 1000.0

        place_gates(robot, self._sim_gates)
        src = self._sim_gates_path if self._sim_gates_path is not None else "<provided>"
        print(
            f"[webots] placed {len(self._sim_gates)} gates from {src}",
            flush=True,
        )

        # Camera sampling period is rounded up to a multiple of timestep so
        # Webots doesn't re-render in between ticks we'll forward anyway.
        camera_period_ms = max(timestep, int(round(self._camera_period * 1000.0)))
        camera = robot.getDevice("cf_camera")
        camera.enable(camera_period_ms)
        gps = robot.getDevice("gps")
        gps.enable(timestep)
        imu = robot.getDevice("inertial unit")
        imu.enable(timestep)
        gyro = robot.getDevice("gyro")
        gyro.enable(timestep)

        motors = [robot.getDevice(f"m{i}_motor") for i in (1, 2, 3, 4)]
        # Motor directions matching the Crazyflie proto's rotor mixing.
        motor_signs = (-1, 1, -1, 1)
        for m, sign in zip(motors, motor_signs):
            m.setPosition(float("inf"))
            m.setVelocity(sign)

        # Prime sensors with one step so getValues() returns real data.
        if robot.step(timestep) == -1:
            return
        last_xyz = np.array(gps.getValues(), dtype=np.float64)
        self.connected.emit(f"webots:{ROBOT_NAME}")

        controller = HoverController()
        next_pose_t = 0.0
        next_frame_t = 0.0

        while not self.isInterruptionRequested():
            if robot.step(timestep) == -1:
                return

            xyz = np.array(gps.getValues(), dtype=np.float64)
            vel = (xyz - last_xyz) / dt
            last_xyz = xyz
            rpy = imu.getRollPitchYaw()
            quat = imu.getQuaternion()
            rates = gyro.getValues()

            # Use sim time to throttle emissions; the loop still ticks at
            # `timestep` so the PID below runs every iteration.
            sim_t = robot.getTime()
            now = time.time()
            if sim_t >= next_pose_t:
                self.pose_ready.emit(DronePose(
                    timestamp=now,
                    x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                    roll=float(np.degrees(rpy[0])),
                    pitch=float(np.degrees(rpy[1])),
                    yaw=float(np.degrees(rpy[2])),
                ))
                next_pose_t = sim_t + self._pose_period

            if sim_t >= next_frame_t:
                self._emit_frame(camera, now)
                next_frame_t = sim_t + self._camera_period

            sp = None if self._stopped else self._setpoint.get()
            if sp is None:
                pwm = [0.0, 0.0, 0.0, 0.0]
            else:
                sensors = _Sensors(
                    x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                    vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                    roll=float(rpy[0]), pitch=float(rpy[1]), yaw=float(rpy[2]),
                    rate_roll=float(rates[0]), rate_pitch=float(rates[1]),
                    rate_yaw=float(rates[2]),
                    quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )
                pwm = controller.step(
                    vx_body=sp.vx, vy_body=sp.vy,
                    yaw_rate_rad=float(np.radians(sp.yaw_rate)),
                    height=sp.height, dt=dt, sensors=sensors,
                )
            for m, sign, p in zip(motors, motor_signs, pwm):
                m.setVelocity(sign * p)

    def _emit_frame(self, camera, timestamp: float) -> None:
        raw = camera.getImage()
        if not raw:
            return
        h, w = camera.getHeight(), camera.getWidth()
        bgra = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        self._seq += 1
        self.frame_ready.emit(Frame(timestamp=timestamp, seq=self._seq, image=bgr))
