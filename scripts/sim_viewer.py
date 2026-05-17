"""Launch Webots in the background and run the integrated stack against it.

Webots runs headless (--no-rendering --minimize --batch) in real-time mode,
so the FPV window in this process is the only thing the user sees. The drone
in sim/worlds/race.wbt has controller "<extern>", meaning Webots blocks
until this Python process attaches via the controller library.

Equivalent to the live stack — the only difference is `--source webots`:

    uv run python scripts/sim_viewer.py
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CFG = yaml.safe_load((REPO / "config" / "default.yaml").read_text())["webots"]
WEBOTS = Path(os.environ.get("WEBOTS", CFG["binary"]))
WORLD = REPO / CFG["world"]

if not WEBOTS.exists():
    raise SystemExit(
        f"Webots binary not found at {WEBOTS}. Install Webots or set "
        f"WEBOTS=<path> (or update webots.binary in config/default.yaml)."
    )


def _find_webots_home(binary: Path) -> Path:
    """Locate the Webots install root. On macOS this is the .app bundle (libs
    live under Contents/lib/controller); on Linux/Windows it's the dir that
    directly contains lib/controller/python."""
    # macOS .app bundle: wb.py joins WEBOTS_HOME with Contents/lib/...,
    # so WEBOTS_HOME must be the .app dir, not Contents/. Check this first
    # because Contents/lib/controller/python also matches the Linux pattern.
    if sys.platform == "darwin":
        for parent in [binary.parent, *binary.parents]:
            if parent.suffix == ".app" and (
                parent / "Contents" / "lib" / "controller" / "python"
            ).is_dir():
                return parent
    for parent in [binary.parent, *binary.parents]:
        if (parent / "lib" / "controller" / "python").is_dir():
            return parent
    raise SystemExit(f"Could not locate Webots install root from {binary}")


HOME = _find_webots_home(WEBOTS)
os.environ["WEBOTS_HOME"] = str(HOME)

_controller_root = HOME / "Contents" if sys.platform == "darwin" else HOME
sys.path.insert(0, str(_controller_root / "lib" / "controller" / "python"))

# Webots' Python controller binds to a native library; tell the loader where.
_lib = str(_controller_root / "lib" / "controller")
if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = f"{_lib}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
elif sys.platform.startswith("linux"):
    os.environ["LD_LIBRARY_PATH"] = f"{_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# IPC handshake: Webots listens on --port=N, the extern controller dials in
# via WEBOTS_CONTROLLER_URL. Use a non-default port so a casually-running
# Webots elsewhere on the machine doesn't collide.
PORT = os.environ.get("WEBOTS_PORT", "1234")
os.environ["WEBOTS_CONTROLLER_URL"] = f"ipc://{PORT}/{CFG['robot_name']}"

proc = subprocess.Popen([
    str(WEBOTS),
    f"--port={PORT}",
    "--mode=realtime",
    "--no-rendering",
    "--minimize",
    "--batch",
    "--stdout",
    "--stderr",
    str(WORLD),
])


@atexit.register
def _cleanup() -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


from src.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--source", "webots", "--autostart"]))
