"""Load a recording's pose log and gate layout, filter the drone path with a
constant-velocity Kalman filter, and pop up an interactive 3D viewer.

Usage:
    uv run python visualize_flight_3d.py [recording_dir]

Default recording is recordings/20260513_115203. The lighthouse takes a few
frames to acquire a fix at the start of a run, so rows with all-zero pose are
dropped before filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def load_measurements(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, positions) with the leading no-fix rows dropped."""
    ts: list[float] = []
    xyz: list[tuple[float, float, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp"]))
            xyz.append((float(row["x"]), float(row["y"]), float(row["z"])))
    ts_arr = np.array(ts)
    pos_arr = np.array(xyz)

    # Drop rows before the lighthouse has a fix (exact zeros across xyz).
    valid = ~np.all(pos_arr == 0.0, axis=1)
    if valid.any():
        first = np.argmax(valid)
        ts_arr = ts_arr[first:]
        pos_arr = pos_arr[first:]
    return ts_arr, pos_arr


def load_gates(path: Path) -> dict[str, np.ndarray]:
    """Return {gate_id: (4,3) array of corners in CORNER_ORDER}."""
    with path.open() as f:
        raw = json.load(f)
    return {
        gid: np.array([data["corners"][k] for k in CORNER_ORDER], dtype=float)
        for gid, data in raw.items()
    }


def kalman_smooth(
    ts: np.ndarray,
    meas: np.ndarray,
    meas_std: float = 0.20,
    accel_std: float = 0.5,
) -> np.ndarray:
    """Constant-velocity Kalman filter over 3D position.

    meas_std: assumed std-dev of each position measurement (m).
    accel_std: std-dev of unmodelled acceleration (m/s^2) — drives the process
    noise. Larger values let the filter track aggressive flight.
    """
    n = len(ts)
    state = np.zeros(6)
    state[:3] = meas[0]
    P = np.eye(6) * 1.0

    H = np.zeros((3, 6))
    H[:, :3] = np.eye(3)
    R = np.eye(3) * meas_std**2

    out = np.zeros((n, 3))
    out[0] = meas[0]

    for i in range(1, n):
        dt = float(ts[i] - ts[i - 1])
        if dt <= 0:
            out[i] = state[:3]
            continue

        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Discrete-time white-noise-acceleration process noise (per axis).
        q = accel_std**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        block = np.array([[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]]) * q
        Q = np.zeros((6, 6))
        for k in range(3):
            Q[k, k] = block[0, 0]
            Q[k, k + 3] = block[0, 1]
            Q[k + 3, k] = block[1, 0]
            Q[k + 3, k + 3] = block[1, 1]

        state = F @ state
        P = F @ P @ F.T + Q

        y = meas[i] - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ y
        P = (np.eye(6) - K @ H) @ P

        out[i] = state[:3]

    return out


def draw_gate(ax, corners: np.ndarray, label: str) -> None:
    face = Poly3DCollection(
        [corners],
        facecolors=(1.0, 0.55, 0.0, 0.25),
        edgecolors=(1.0, 0.35, 0.0, 1.0),
        linewidths=2.5,
    )
    ax.add_collection3d(face)

    # Close the loop for the outline scatter.
    loop = np.vstack([corners, corners[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#ff6a00", linewidth=2.5)

    centroid = corners.mean(axis=0)
    ax.text(
        centroid[0],
        centroid[1],
        centroid[2] + 0.05,
        label,
        color="#aa3300",
        fontsize=9,
        ha="center",
    )


def set_axes_equal(ax) -> None:
    """Force equal aspect ratio on a 3D axes (matplotlib doesn't do this)."""
    x_lim, y_lim, z_lim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    spans = np.array([np.ptp(x_lim), np.ptp(y_lim), np.ptp(z_lim)])
    centers = np.array([np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)])
    radius = spans.max() / 2.0
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "recording",
        nargs="?",
        type=Path,
        default=Path("recordings/20260513_115203"),
    )
    ap.add_argument("--meas-std", type=float, default=0.20)
    ap.add_argument("--accel-std", type=float, default=0.5)
    args = ap.parse_args()

    ts, raw = load_measurements(args.recording / "measurements.csv")
    gates = load_gates(args.recording / "gate_positions.json")
    filtered = kalman_smooth(ts, raw, args.meas_std, args.accel_std)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for gid, corners in gates.items():
        draw_gate(ax, corners, gid.replace("gate_", "G"))

    ax.plot(
        raw[:, 0],
        raw[:, 1],
        raw[:, 2],
        color="#888",
        linewidth=0.8,
        alpha=0.5,
        label="raw pose",
    )
    ax.plot(
        filtered[:, 0],
        filtered[:, 1],
        filtered[:, 2],
        color="#1f77b4",
        linewidth=2.0,
        label="kalman filtered",
    )
    ax.scatter(
        filtered[0, 0],
        filtered[0, 1],
        filtered[0, 2],
        color="#2ca02c",
        s=60,
        label="start",
    )
    ax.scatter(
        filtered[-1, 0],
        filtered[-1, 1],
        filtered[-1, 2],
        color="#d62728",
        s=60,
        label="end",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{args.recording.name} — {len(ts)} poses, {len(gates)} gates")
    ax.legend(loc="upper left")
    set_axes_equal(ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
