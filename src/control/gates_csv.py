"""Per-gate measurement record + read/write for the gates.csv format.

Format (matches the ground-truth files under data/recordings/<id>/gates.csv):

    Gate,x,y,z,theta,width,height
    1,0.345,0.7725,1.155,1.5375,0.29,0.40
    ...

`x, y, z` is the gate centre in world metres. `theta` is the yaw of the gate's
approach normal in the world XY plane, radians. `width` / `height` are the
average horizontal / vertical edge lengths of the LED frame in metres.

Read order is preserved: the row order in the CSV is the race-mode flight
order, so the writer just stamps sequential gate IDs starting at 1.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RecordedGate:
    """One gate measurement captured during a recon lap (or loaded from CSV)."""
    center: np.ndarray   # (3,) world XYZ in metres
    normal: np.ndarray   # (3,) unit vector pointing toward the approach side
    width_m: float
    height_m: float


def gate_yaw(normal: np.ndarray) -> float:
    """Yaw angle (radians) of the gate's approach normal in the world XY plane."""
    return float(math.atan2(float(normal[1]), float(normal[0])))


def normal_from_theta(theta: float) -> np.ndarray:
    """Inverse of `gate_yaw` — rebuild a horizontal unit normal from a yaw angle."""
    return np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)


def save_gates_csv(path: Path, gates: list[RecordedGate]) -> None:
    """Write gates in flight order (IDs 1..N). Creates parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gate", "x", "y", "z", "theta", "width", "height"])
        for i, g in enumerate(gates, start=1):
            cx, cy, cz = (float(g.center[0]), float(g.center[1]), float(g.center[2]))
            w.writerow([i, cx, cy, cz, gate_yaw(g.normal), g.width_m, g.height_m])


def load_gates_csv(path: Path) -> list[RecordedGate]:
    """Read gates in file order. Trailing blank rows are skipped."""
    path = Path(path)
    out: list[RecordedGate] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("x"):
                continue
            center = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            theta = float(row["theta"])
            out.append(RecordedGate(
                center=center,
                normal=normal_from_theta(theta),
                width_m=float(row["width"]),
                height_m=float(row["height"]),
            ))
    return out
