"""Gate Kalman filter and camera->world geometry for the recon FSM.

Detections arrive as 3D corner sets in the OpenCV camera frame (x right,
y down, z forward). We rotate them into the drone body frame (x forward,
y left, z up), then into the world frame via the drone pose. The Kalman
filter itself is a static-target model on the 12D corners vector — for a
stationary gate that just averages noisy detections with a shrinking
covariance, identical to the aerial-robotics version.
"""

from __future__ import annotations

import math

import numpy as np

from src.control.gates_csv import RecordedGate
from src.messages import DronePose, Gate3D


# OpenCV camera frame -> drone body frame.
#   cam +x (right)   -> body -y  (body +y is left)
#   cam +y (down)    -> body -z  (body +z is up)
#   cam +z (forward) -> body +x
R_BODY_FROM_CAM = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def rotation_world_from_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler rotation matrix mapping body-frame vectors into the world frame."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,    cp * sr,                cp * cr],
    ])


def camera_corners_to_world(
    corners_cam: np.ndarray, pose: DronePose
) -> list[np.ndarray]:
    """Lift a (4, 3) camera-frame corner array to a list of (3,) world points."""
    R = rotation_world_from_body(
        math.radians(pose.roll),
        math.radians(pose.pitch),
        math.radians(pose.yaw),
    )
    origin = np.array([pose.x, pose.y, pose.z])
    return [R @ (R_BODY_FROM_CAM @ c) + origin for c in corners_cam]


def gate_normal(corners: list[np.ndarray]) -> np.ndarray | None:
    """Unit normal to the gate plane from four ordered (TL, TR, BR, BL) corners."""
    edge1 = corners[1] - corners[0]
    edge2 = corners[3] - corners[0]
    n = np.cross(edge1, edge2)
    mag = float(np.linalg.norm(n))
    if mag < 1e-6:
        return None
    return n / mag


class GateKalman:
    """Static-target Kalman filter on the 12D gate-corner state vector."""

    DIM = 12

    def __init__(
        self,
        corners: list[np.ndarray],
        initial_uncertainty: float = 0.5,
        process_noise: float = 0.001,
        measurement_noise: float = 0.1,
    ) -> None:
        self.x = np.concatenate(corners).astype(float)
        self.P = np.eye(self.DIM) * initial_uncertainty
        self.Q = np.eye(self.DIM) * process_noise
        self.R = np.eye(self.DIM) * measurement_noise

    def update(self, corners: list[np.ndarray], measurement_noise: float | None = None) -> None:
        self.P = self.P + self.Q
        z = np.concatenate(corners).astype(float)
        y = z - self.x
        R = self.R
        if measurement_noise is not None:
            R = np.eye(self.DIM) * measurement_noise
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.DIM) - K) @ self.P

    def corners(self) -> list[np.ndarray]:
        return [self.x[i * 3:(i + 1) * 3].copy() for i in range(4)]

    def center(self) -> np.ndarray:
        return self.x.reshape(4, 3).mean(axis=0)


class GateTracker:
    """Per-gate Kalman tracker with the chosen approach-side normal."""

    BASE_MEASUREMENT_NOISE = 0.1

    def __init__(self) -> None:
        self.kalman: GateKalman | None = None
        # Unit world-frame normal pointing toward the side from which the drone
        # is approaching. Set when the drone first picks a side; used to keep
        # subsequent normal computations from flipping orientation mid-flight.
        self.approach_normal: np.ndarray | None = None
        # Snapshot of each gate after its measurement state holds it still.
        # Survives `reset()` between gates so the planner can hand the list off
        # to the racing trajectory and the CSV writer at end-of-recon.
        self.recorded_gates: list[RecordedGate] = []

    @property
    def has_estimate(self) -> bool:
        return self.kalman is not None

    def update(self, gate: Gate3D, pose: DronePose) -> None:
        """Pick the best of `gate.corners_cam_m` and feed it into the filter.

        The detector returns every gate in frame. The "best" candidate is the
        one closest to the current Kalman estimate (so we stay locked on the
        same gate across frames) or, if there's no estimate yet, the one
        closest to the drone (the next one we'd actually fly toward).
        """
        if not gate.corners_cam_m:
            return
        measurement_noise = self._measurement_noise_for_pose(pose)
        if measurement_noise is None:
            return

        candidates = [camera_corners_to_world(c, pose) for c in gate.corners_cam_m]
        drone_pos = np.array([pose.x, pose.y, pose.z])

        if self.kalman is not None:
            est = self.kalman.center()
            best = min(
                candidates,
                key=lambda cs: np.linalg.norm(np.mean(cs, axis=0) - est),
            )
        else:
            best = min(
                candidates,
                key=lambda cs: np.linalg.norm(np.mean(cs, axis=0) - drone_pos),
            )

        if self.kalman is None:
            self.kalman = GateKalman(best, measurement_noise=measurement_noise)
        else:
            self.kalman.update(best, measurement_noise=measurement_noise)

    def reset(self) -> None:
        self.kalman = None
        self.approach_normal = None

    def reset_filter_only(self) -> None:
        """Drop the filter but keep the approach side fixed (used at MEASURE entry)."""
        self.kalman = None

    def record_current_gate(self) -> RecordedGate | None:
        """Snapshot the current filter estimate into `recorded_gates`.

        Called by MeasureState after the hold has converged. Returns the
        appended record (or None when the filter / approach normal aren't
        ready).
        """
        if self.kalman is None or self.approach_normal is None:
            return None
        corners = self.kalman.corners()
        tl, tr, br, bl = corners
        width = (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) / 2.0
        height = (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) / 2.0
        rec = RecordedGate(
            center=self.kalman.center().copy(),
            normal=self.approach_normal.copy(),
            width_m=width,
            height_m=height,
        )
        self.recorded_gates.append(rec)
        return rec

    def _measurement_noise_for_pose(self, pose: DronePose) -> float | None:
        count = pose.lighthouse_bs_visible
        if count is None:
            return self.BASE_MEASUREMENT_NOISE
        if count <= 1:
            print(
                f"[GATE_REJECT] lighthouse base stations visible={count}; need >=2",
                flush=True,
            )
            return None
        if count >= 3:
            return self.BASE_MEASUREMENT_NOISE / 5.0
        return self.BASE_MEASUREMENT_NOISE

    def oriented_normal(self) -> np.ndarray | None:
        """Gate normal flipped to align with `approach_normal` when one is set."""
        if self.kalman is None:
            return None
        n = gate_normal(self.kalman.corners())
        if n is None:
            return None
        if self.approach_normal is not None and float(np.dot(n, self.approach_normal)) < 0:
            n = -n
        return n
