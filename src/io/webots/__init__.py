"""Webots backend: in-process extern controller for sim/worlds/race.wbt.

`build_webots(cfg)` launches Webots headless and returns a WebotsBackend that
serves as both VideoSource and DroneLink. The PID cascade in `pid.py` converts
hover Setpoints into rotor PWMs so the rest of the pipeline can treat the sim
exactly like the live Crazyflie.
"""

from __future__ import annotations

from src.io.webots.backend import WebotsBackend
from src.io.webots.launcher import launch_webots


def build_webots(cfg: dict) -> WebotsBackend:
    launch_webots(cfg["webots"])
    return WebotsBackend(
        camera_fps=cfg["webots"]["camera_fps"],
        pose_rate_hz=cfg["webots"]["pose_rate_hz"],
        pose_position_noise_std_m=cfg["webots"].get("pose_position_noise_std_m", 0.0),
        pose_attitude_noise_std_deg=cfg["webots"].get("pose_attitude_noise_std_deg", 0.0),
    )


__all__ = ["build_webots", "WebotsBackend"]
