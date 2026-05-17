"""Webots extern-controller backend.

A single class that implements both `VideoSource` and `DroneLink` (see
src/io/sources.py), so it drops into src/main.py the same way `ReplayThread`
does. The drone in the .wbt world must have `controller "<extern>"` so
Webots waits for this Python process to attach.

Per-tick (`basicTimeStep`, 8 ms = 125 Hz in the bundled world):

  * Read GPS + IMU + gyro -> emit DronePose (rpy in degrees, like the real
    Crazyflie log stream).
  * Read the simulated camera (BGRA at sim_camera_width x sim_camera_height),
    convert to grayscale and resize to video.width x video.height so it
    matches the AI-deck format -> emit Frame.
  * If a hover Setpoint has been pushed (vx, vy, yaw_rate, height — the
    cflib hover-commander surface), run the cascaded controller in
    `_webots_pid` to get motor PWMs and apply them. Without a setpoint or
    after `send_stop`, motors are held at zero (drone sits on its pad).

The Webots `controller` Python module is only importable if `WEBOTS_HOME` is
set and the platform-specific library path includes Webots' lib/controller.
`scripts/sim_viewer.py` handles that before importing this module.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt6 import QtCore

from src.bus import Latest
from src.io._webots_pid import HoverController, _Sensors
from src.messages import DronePose, Frame, Setpoint


class WebotsBackend(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)  # Frame
    pose_ready = QtCore.pyqtSignal(object)  # DronePose
    connected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        robot_name: str,
        out_width: int,
        out_height: int,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._robot_name = robot_name
        self._out_width = out_width
        self._out_height = out_height
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
        # Lazy import: only fails if --source webots is actually used.
        from controller import Robot  # type: ignore[import-not-found]

        robot = Robot()
        timestep = int(robot.getBasicTimeStep())
        dt = timestep / 1000.0

        camera = robot.getDevice("cf_camera")
        camera.enable(timestep)
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
        self.connected.emit(f"webots:{self._robot_name}")

        controller = HoverController()

        while not self.isInterruptionRequested():
            if robot.step(timestep) == -1:
                return

            xyz = np.array(gps.getValues(), dtype=np.float64)
            vel = (xyz - last_xyz) / dt
            last_xyz = xyz
            rpy = imu.getRollPitchYaw()
            quat = imu.getQuaternion()
            rates = gyro.getValues()

            now = time.time()
            self.pose_ready.emit(DronePose(
                timestamp=now,
                x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                roll=float(np.degrees(rpy[0])),
                pitch=float(np.degrees(rpy[1])),
                yaw=float(np.degrees(rpy[2])),
            ))

            self._emit_frame(camera, now)

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
        gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
        if (w, h) != (self._out_width, self._out_height):
            gray = cv2.resize(gray, (self._out_width, self._out_height),
                              interpolation=cv2.INTER_AREA)
        self._seq += 1
        self.frame_ready.emit(Frame(timestamp=timestamp, seq=self._seq, image=gray))
