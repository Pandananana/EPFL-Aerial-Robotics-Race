"""Live backend: AI-deck UDP video + Crazyflie radio link.

Real hardware path. `build_live(cfg)` constructs both halves of the
(VideoSource, DroneLink) pair that src/main.py wires into the pipeline.
"""

from __future__ import annotations

from src.io.live.link import CrazyflieLink
from src.io.live.video import UdpVideoThread
from src.io.sources import DroneLink, VideoSource


def build_live(cfg: dict, *, no_fly: bool = False) -> tuple[VideoSource, DroneLink]:
    video = UdpVideoThread(
        aideck_ip=cfg["network"]["aideck_ip"],
        aideck_port=cfg["network"]["aideck_port"],
        local_port=cfg["network"]["local_port"],
    )
    link = CrazyflieLink(
        uri=cfg["crazyflie"]["uri"],
        cache_dir=cfg["crazyflie"]["cache_dir"],
        setpoint_rate_hz=cfg["control"]["setpoint_rate_hz"],
        disable_flight=no_fly,
    )
    return video, link


__all__ = ["build_live", "CrazyflieLink", "UdpVideoThread"]
