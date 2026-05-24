"""Plot drone trajectory with estimated and actual gate positions.

Reads from a recording directory:
  run_log.csv         - pose trajectory
  gate_estimates.csv  - accepted measurements + final Kalman estimates (written by recorder)
  gates.csv           - ground-truth gate positions

Usage:
    uv run python scripts/plot_recording.py data/recordings/<id>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_trajectory(recording_dir: Path) -> pd.DataFrame:
    log = pd.read_csv(recording_dir / "run_log.csv")
    df = log[log["event"] == "pose"][["timestamp", "x", "y", "z"]].dropna().copy()
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df.reset_index(drop=True)


def load_gate_estimates(recording_dir: Path) -> tuple[list[dict], list[dict]]:
    """Parse gate_estimates.csv, return (raw_measurements, final_estimates).

    Handles two formats:
    - New: two sections separated by '# Accepted measurements' / '# Final Kalman estimates'
    - Old: single section with header 'Gate,x,y,z,theta,width,height'
    """
    path = recording_dir / "gate_estimates.csv"
    if not path.exists():
        return [], []

    with open(path, newline="") as f:
        text = f.read()

    # Detect format by presence of section markers
    if "# Accepted" in text or "# Final" in text:
        raw: list[dict] = []
        final: list[dict] = []
        section = None
        for row in csv.reader(text.splitlines()):
            if not row or row[0].startswith("# Accepted"):
                section = "raw"
                continue
            if row[0].startswith("# Final"):
                section = "final"
                continue
            if row[0].lower() in ("gate", ""):
                continue
            try:
                if section == "raw":
                    raw.append({"gate": int(row[0]), "x": float(row[1]),
                                 "y": float(row[2]), "z": float(row[3])})
                elif section == "final":
                    final.append({"gate": int(row[0]), "x": float(row[1]),
                                   "y": float(row[2]), "z": float(row[3])})
            except (ValueError, IndexError):
                continue
        return raw, final
    else:
        # Old format: Gate,x,y,z,theta,width,height
        final = []
        for row in csv.reader(text.splitlines()):
            if not row or row[0].lower() in ("gate", ""):
                continue
            try:
                final.append({"gate": int(row[0]), "x": float(row[1]),
                               "y": float(row[2]), "z": float(row[3])})
            except (ValueError, IndexError):
                continue
        return [], final


def load_ground_truth(recording_dir: Path) -> pd.DataFrame | None:
    for name in ("gates.csv", "gate_positions.csv"):
        p = recording_dir / name
        if p.exists():
            return pd.read_csv(p)
    return None


def plot(recording_dir: Path) -> None:
    traj = load_trajectory(recording_dir)
    raw_meas, final_ests = load_gate_estimates(recording_dir)
    gt = load_ground_truth(recording_dir)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(recording_dir.name, fontsize=12)

    ax.plot(traj["x"], traj["y"], linewidth=1, color="steelblue", label="trajectory")
    ax.plot(traj["x"].iloc[0], traj["y"].iloc[0], "go", markersize=8, label="start")
    ax.plot(traj["x"].iloc[-1], traj["y"].iloc[-1], "rs", markersize=8, label="end")

    # Raw accepted measurements — small dots per gate
    if raw_meas:
        raw_df = pd.DataFrame(raw_meas)
        for gate_num, grp in raw_df.groupby("gate"):
            ax.scatter(grp["x"], grp["y"], s=15, alpha=0.4, zorder=3,
                       label=f"measurements G{gate_num}" if gate_num == raw_df["gate"].min() else None)

    # Final Kalman estimates
    for est in final_ests:
        ax.plot(est["x"], est["y"], "b^", markersize=11, zorder=5,
                label="estimated" if est == final_ests[0] else None)
        ax.annotate(f"G{est['gate']} est\nz={est['z']:.2f}m",
                    xy=(est["x"], est["y"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=7, color="blue")

    # Ground truth
    if gt is not None:
        for _, row in gt.iterrows():
            ax.plot(row["x"], row["y"], "kD", markersize=10, zorder=6,
                    label="ground truth" if row.name == 0 else None)
            ax.annotate(f"G{int(row['Gate'])}\nz={row['z']:.2f}m",
                        xy=(row["x"], row["y"]), xytext=(6, -14),
                        textcoords="offset points", fontsize=7)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path, help="Recording directory")
    args = ap.parse_args()
    plot(args.recording)


if __name__ == "__main__":
    main()
