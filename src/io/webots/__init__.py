"""Webots backend: in-process extern controller for sim/worlds/race.wbt.

`build_webots(cfg)` launches Webots headless and returns a WebotsBackend that
serves as both VideoSource and DroneLink. The PID cascade in `pid.py` converts
hover Setpoints into rotor PWMs so the rest of the pipeline can treat the sim
exactly like the live Crazyflie.

Gate ground truth is fixed: the backend always reads
`data/gates/sim_gates.csv` and places each `GATE{i}` node accordingly via the
Supervisor API. The same list is exposed to callers (main.py) as the truth
source for race-only / debug plotting, so the sim, planner prior, and viewer
can never disagree.
"""

from __future__ import annotations

from pathlib import Path

from src.control.gates_csv import RecordedGate, load_gates_csv
from src.io.webots.backend import WebotsBackend
from src.io.webots.launcher import REPO, launch_webots

SIM_GATES_CSV = REPO / "data" / "gates" / "sim_gates.csv"


def build_webots(cfg: dict) -> tuple[WebotsBackend, list[RecordedGate], Path]:
    """Launch Webots and build the backend. Returns (backend, gates, csv_path)."""
    sim_gates = load_gates_csv(SIM_GATES_CSV)
    launch_webots(cfg["webots"])
    backend = WebotsBackend(
        camera_fps=cfg["webots"]["camera_fps"],
        pose_rate_hz=cfg["webots"]["pose_rate_hz"],
        sim_gates=sim_gates,
        sim_gates_path=SIM_GATES_CSV,
    )
    return backend, sim_gates, SIM_GATES_CSV


__all__ = ["build_webots", "WebotsBackend", "SIM_GATES_CSV"]
