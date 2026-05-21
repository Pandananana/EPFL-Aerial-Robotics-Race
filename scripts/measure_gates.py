"""Capture gate corner positions from the Crazyflie's state estimate.

Walk the drone (held by hand, or on a stick) to each gate corner and tap SPACE
to record the current `stateEstimate.{x,y,z}`. Captures are gated on having at
least two lighthouse base stations visible — otherwise the pose is unreliable
and we refuse to save the point.

Capture order: gate 0 TL, TR, BR, BL, gate 1 TL, ..., gate 4 BL — 20 points.

Output goes to `data/gates/<YYYY-MM-DD_HH-MM>.csv` in the format
`src/control/gates_csv.py` reads (gate IDs start at 1, matching existing files).

Keys:
    SPACE  capture the current pose for the next corner
    U      undo the last capture
    Q/Esc  quit (Ctrl-C also works)

Run:
    uv run python scripts/measure_gates.py
"""

from __future__ import annotations

import argparse
import datetime
import math
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cflib.crtp  # noqa: E402
from cflib.crazyflie import Crazyflie  # noqa: E402
from cflib.crazyflie.log import LogConfig  # noqa: E402

from src.control.gates_csv import RecordedGate, save_gates_csv  # noqa: E402

NUM_GATES = 5
CORNERS_PER_GATE = 4
CORNER_LABELS = ["TL", "TR", "BR", "BL"]
TOTAL_POINTS = NUM_GATES * CORNERS_PER_GATE
MIN_BASE_STATIONS = 2

LIGHTHOUSE_BS_AVAILABLE = "lighthouse.bsAvailable"


class PoseStream:
    """Latches the latest pose + base-station count from a cflib log config."""

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.bs_visible: int | None = None
        self.last_update: float = 0.0
        self._has_lighthouse_var = False

    def attach(self, cf: Crazyflie) -> None:
        lg = LogConfig(name="MeasureGates", period_in_ms=20)
        lg.add_variable("stateEstimate.x", "float")
        lg.add_variable("stateEstimate.y", "float")
        lg.add_variable("stateEstimate.z", "float")
        toc = cf.log.toc
        group_vars = toc.toc.get("lighthouse", {}) if toc else {}
        if "bsAvailable" in group_vars:
            lg.add_variable(LIGHTHOUSE_BS_AVAILABLE, "uint16_t")
            self._has_lighthouse_var = True
        else:
            print(
                f"[warn] {LIGHTHOUSE_BS_AVAILABLE} not in log TOC; "
                "capture will not be gated on lighthouse visibility.",
                file=sys.stderr,
            )
        cf.log.add_config(lg)
        lg.data_received_cb.add_callback(self._on_data)
        lg.start()

    def _on_data(self, _ts_ms, data, _conf) -> None:
        self.x = float(data["stateEstimate.x"])
        self.y = float(data["stateEstimate.y"])
        self.z = float(data["stateEstimate.z"])
        if self._has_lighthouse_var:
            self.bs_visible = int(data[LIGHTHOUSE_BS_AVAILABLE]).bit_count()
        self.last_update = time.time()

    def fresh(self) -> bool:
        return self.last_update > 0.0 and (time.time() - self.last_update) < 0.5

    def can_capture(self) -> bool:
        if not self.fresh():
            return False
        if not self._has_lighthouse_var:
            return True
        return self.bs_visible is not None and self.bs_visible >= MIN_BASE_STATIONS


def _connect(uri: str, cache_dir: str) -> Crazyflie:
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache=cache_dir)
    connected = {"done": False, "failed": False, "reason": ""}

    def _on_connected(_uri: str) -> None:
        connected["done"] = True

    def _on_failed(_uri: str, msg: str) -> None:
        connected["failed"] = True
        connected["reason"] = msg

    cf.connected.add_callback(_on_connected)
    cf.connection_failed.add_callback(_on_failed)
    cf.connection_lost.add_callback(_on_failed)

    print(f"Connecting to {uri} ...", flush=True)
    cf.open_link(uri)
    deadline = time.time() + 10.0
    while not connected["done"] and not connected["failed"] and time.time() < deadline:
        time.sleep(0.05)
    if connected["failed"]:
        raise RuntimeError(f"Crazyflie connection failed: {connected['reason']}")
    if not connected["done"]:
        raise RuntimeError("Crazyflie connection timed out")
    print("Connected.", flush=True)
    return cf


def _gate_label(idx: int) -> str:
    g = idx // CORNERS_PER_GATE
    c = idx % CORNERS_PER_GATE
    return f"Gate {g} {CORNER_LABELS[c]}"


def _render_status(idx: int, stream: PoseStream, last_msg: str) -> str:
    if idx >= TOTAL_POINTS:
        target = "all gates captured"
    else:
        target = f"next: {_gate_label(idx)}  ({idx + 1}/{TOTAL_POINTS})"
    bs = stream.bs_visible if stream.bs_visible is not None else "?"
    ok = "OK" if stream.can_capture() else "BLOCKED"
    pose = f"x={stream.x:+.3f} y={stream.y:+.3f} z={stream.z:+.3f}"
    line = f"[{ok}] bs={bs} | {pose} | {target}"
    if last_msg:
        line = f"{line}   {last_msg}"
    return line


def _gate_from_corners(corners: np.ndarray) -> RecordedGate:
    """corners: (4, 3) world-frame XYZ in TL, TR, BR, BL order."""
    tl, tr, br, bl = corners
    center = corners.mean(axis=0)
    width_m = 0.5 * (
        float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))
    )
    height_m = 0.5 * (
        float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))
    )
    # The "normal" field in RecordedGate is, by convention in this codebase,
    # the gate's width axis projected onto world XY (see gate_debug_plot.py).
    # That is what theta = atan2(normal.y, normal.x) decodes back to.
    width_vec_xy = np.array([tr[0] - tl[0], tr[1] - tl[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(width_vec_xy[:2]))
    if n < 1e-9:
        raise ValueError("Gate width vector has zero XY length; corners look wrong.")
    normal = width_vec_xy / n
    return RecordedGate(
        center=center, normal=normal, width_m=width_m, height_m=height_m,
    )


class RawTerminal:
    """Context manager that puts stdin into cbreak (single-char) mode."""

    def __enter__(self) -> "RawTerminal":
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        print()  # newline after the live status line

    def read_key(self, timeout: float) -> str | None:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        return sys.stdin.read(1)


def _capture_loop(stream: PoseStream) -> list[np.ndarray] | None:
    """Returns the list of (20, 3) captured points, or None if user quit early."""
    points: list[np.ndarray] = []
    last_msg = ""

    print()
    print("Press SPACE to capture, U to undo, Q to quit.")
    print(f"Capture gated on >= {MIN_BASE_STATIONS} lighthouse base stations.")
    print()

    with RawTerminal() as term:
        while True:
            idx = len(points)
            line = _render_status(idx, stream, last_msg)
            # \r to return to line start, ESC[K to clear to end-of-line.
            sys.stdout.write(f"\r\033[K{line}")
            sys.stdout.flush()

            if idx >= TOTAL_POINTS:
                return points

            key = term.read_key(0.1)
            if key is None:
                continue
            if key in ("q", "\x1b", "\x03"):  # q, Esc, Ctrl-C
                return None
            if key == "u":
                if points:
                    dropped = points.pop()
                    last_msg = (
                        f"undid {_gate_label(len(points))} "
                        f"({dropped[0]:+.3f},{dropped[1]:+.3f},{dropped[2]:+.3f})"
                    )
                else:
                    last_msg = "nothing to undo"
                continue
            if key == " ":
                if not stream.fresh():
                    last_msg = "no pose yet — check radio link"
                    continue
                if not stream.can_capture():
                    bs = stream.bs_visible
                    last_msg = (
                        f"refused: only {bs} base station(s) visible "
                        f"(need >= {MIN_BASE_STATIONS})"
                    )
                    continue
                p = np.array([stream.x, stream.y, stream.z], dtype=np.float64)
                points.append(p)
                last_msg = (
                    f"captured {_gate_label(idx)} "
                    f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
                )


def _print_summary(points: list[np.ndarray], gates: list[RecordedGate]) -> None:
    print()
    print("Captured corners (world frame, metres):")
    for i in range(NUM_GATES):
        chunk = points[i * 4 : (i + 1) * 4]
        formatted = ", ".join(
            f"{lab}=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})"
            for lab, p in zip(CORNER_LABELS, chunk)
        )
        print(f"  gate {i}: {formatted}")

    print()
    print("Derived gates (CSV rows, 1-indexed):")
    print("  Gate,x,y,z,theta,width,height")
    for i, g in enumerate(gates, start=1):
        theta = math.atan2(float(g.normal[1]), float(g.normal[0]))
        cx, cy, cz = (float(g.center[0]), float(g.center[1]), float(g.center[2]))
        print(f"  {i},{cx:.4f},{cy:.4f},{cz:.4f},{theta:.4f},{g.width_m:.3f},{g.height_m:.3f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--config", type=Path, default=REPO_ROOT / "config" / "default.yaml",
        help="Path to default.yaml (for crazyflie.uri / cache_dir).",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "gates",
        help="Where to write the CSV. Filename is <YYYY-MM-DD_HH-MM>.csv.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    uri = cfg["crazyflie"]["uri"]
    cache_dir = cfg["crazyflie"]["cache_dir"]

    # cflib's connection_lost is async; if the radio drops mid-session we want
    # to bail out cleanly rather than hang on the next sleep.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    cf = _connect(uri, cache_dir)
    stream = PoseStream()
    try:
        stream.attach(cf)
        # Give the first log packets a moment to arrive so the first status
        # render isn't blank.
        deadline = time.time() + 2.0
        while not stream.fresh() and time.time() < deadline:
            time.sleep(0.05)

        try:
            points = _capture_loop(stream)
        except KeyboardInterrupt:
            points = None
    finally:
        cf.close_link()

    if points is None:
        print("Aborted before all gates captured. Nothing written.")
        return 1
    if len(points) != TOTAL_POINTS:
        print(f"Captured {len(points)}/{TOTAL_POINTS} points. Nothing written.")
        return 1

    gates: list[RecordedGate] = []
    for i in range(NUM_GATES):
        chunk = np.stack(points[i * 4 : (i + 1) * 4])  # (4, 3)
        gates.append(_gate_from_corners(chunk))

    _print_summary(points, gates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = args.output_dir / f"{stamp}.csv"
    save_gates_csv(out_path, gates)
    print()
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
