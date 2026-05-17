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
    """One grayscale camera frame, 324x244, uint8."""
    timestamp: float
    seq: int
    image: np.ndarray  # (H, W) uint8


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


@dataclass(frozen=True)
class GateDetection2D:
    """Gates detected in image coordinates for a single frame.

    Each entry in `corners_px` is a (4, 2) float array in TL, TR, BR, BL order.
    Empty list means the detector ran and found nothing.
    """
    timestamp: float
    frame_seq: int
    corners_px: list[np.ndarray] = field(default_factory=list)


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
