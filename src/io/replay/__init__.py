"""Replay backend: plays a recording directory as both VideoSource and DroneLink.

No hardware needed. set_setpoint / send_stop are no-ops — controller output
is dropped because there is no drone to command.
"""

from __future__ import annotations

from pathlib import Path

from src.io.replay.backend import ReplayThread


def build_replay(
    recording: Path,
    speed: float = 1.0,
    *,
    step: bool = False,
    start_frame: int = 1,
) -> ReplayThread:
    return ReplayThread(recording, speed=speed, step=step, start_frame=start_frame)


__all__ = ["build_replay", "ReplayThread"]
