"""Live entry point — runs the full integrated system with the FPV window.

Equivalent to `python -m src.main`, but with a friendlier path for the
`scripts/` convention. Connects to the AI-deck (video) and the Crazyflie
(radio) using values from config/default.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
