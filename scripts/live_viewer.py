"""Run the integrated system with the FPV window.

Live backend (default — needs the AI-deck and Crazyflie powered on):

    uv run python scripts/live_viewer.py

Replay backend (no hardware needed — plays a recording through the same
perception + UI pipeline; controller / manual setpoints are dropped):

    uv run python scripts/live_viewer.py \\
        --source replay --recording data/recordings/<id> --speed 1.0

Backend protocols live in src/io/sources.py; add Webots etc. there.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
