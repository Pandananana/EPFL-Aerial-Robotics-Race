"""Shared message types passed between modules over Qt signals or Latest latches.

These are the contract between IO, perception, and planning. If you add a
field here, every subscriber sees it. If you rename one, every subscriber
breaks — which is the point.

Timestamps are wall-clock seconds (time.time()) so messages can be matched
across modules and also written to the recording CSV in the existing format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One camera frame, uint8. Grayscale (H, W) from the real AI-deck;
    BGR (H, W, 3) from the Webots backend, which forwards the simulator's
    native colour image without any down-conversion."""
    timestamp: float
    seq: int
    image: np.ndarray


@dataclass(frozen=True)
class DronePose:
    """Latest state estimate from the Crazyflie (x/y/z metres, rpy degrees)."""
    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    lighthouse_bs_visible: int | None = None


@dataclass(frozen=True)
class GateDetection2D:
    """Gates detected in image coordinates for a single frame.

    Each entry in `corners_px` is a (4, 2) float array in TL, TR, BR, BL order.
    Empty list means the detector ran and found nothing.
    """
    timestamp: float
    frame_seq: int
    corners_px: list[np.ndarray] = field(default_factory=list)
    image_shape_hw: tuple[int, int] | None = None


@dataclass(frozen=True)
class Gate3D:
    """3D pose of every gate detected in one frame, in OpenCV camera frame
    (x right, y down, z forward), metres.

    Each entry in `corners_cam_m` is a (4, 3) array in TL, TR, BR, BL order.
    Lists are index-aligned across the four fields.
    """
    timestamp: float
    frame_seq: int
    corners_cam_m: list[np.ndarray] = field(default_factory=list)
    widths_m: list[float] = field(default_factory=list)
    reprojection_errors_px: list[float] = field(default_factory=list)
    near_image_edge: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class Waypoint:
    """Target the controller should fly toward, in world frame.

    x, y, z metres; yaw degrees. `max_speed_mps` caps how fast the controller
    may drive toward this waypoint — used to differentiate cautious recon
    flying from fast race laps. `vx_ff, vy_ff` are an optional world-frame
    velocity feedforward (m/s); set by trajectory-following states (race) so
    the controller doesn't lag and cut curves. Stationary targets leave them
    zero.
    """
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float
    max_speed_mps: float
    vx_ff: float = 0.0
    vy_ff: float = 0.0


@dataclass(frozen=True)
class GateEstimate:
    """Final Kalman-filtered estimate for one gate, in world frame.

    Matches the gates.csv schema so estimates can be saved directly.
    theta_rad is the gate-normal yaw (atan2 of the oriented normal).
    width_m / height_m are derived from the filtered corner positions.
    """
    gate_num: int
    x: float
    y: float
    z: float
    theta_rad: float
    width_m: float
    height_m: float


@dataclass(frozen=True)
class Setpoint:
    """Desired hover-commander setpoint, in the cflib send_hover_setpoint format.

    vx/vy are body-frame velocities (m/s), yaw_rate is deg/s, height is absolute
    z (m). The Crazyflie link reads the latest one at its setpoint timer rate.
    """
    vx: float
    vy: float
    yaw_rate: float
    height: float
