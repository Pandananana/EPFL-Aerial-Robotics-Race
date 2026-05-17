"""Launch Webots in the background and set up the extern-controller env.

`launch_webots(cfg)` must run before anything imports `controller` from the
Webots Python bindings — it locates the Webots install, prepends the controller
library to sys.path / DYLD_LIBRARY_PATH, and dials in via WEBOTS_CONTROLLER_URL.

The simulator runs headless (--no-rendering --minimize --batch) in real-time
mode; the FPV window owned by src/main.py is the only thing the user sees.
The drone in the .wbt world has controller "<extern>", so Webots blocks until
WebotsBackend's QThread imports `controller` and attaches.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


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


def launch_webots(cfg: dict) -> subprocess.Popen:
    webots = Path(os.environ.get("WEBOTS", cfg["binary"]))
    world = REPO / cfg["world"]

    if not webots.exists():
        raise SystemExit(
            f"Webots binary not found at {webots}. Install Webots or set "
            f"WEBOTS=<path> (or update webots.binary in config/default.yaml)."
        )

    home = _find_webots_home(webots)
    os.environ["WEBOTS_HOME"] = str(home)

    controller_root = home / "Contents" if sys.platform == "darwin" else home
    sys.path.insert(0, str(controller_root / "lib" / "controller" / "python"))

    lib = str(controller_root / "lib" / "controller")
    if sys.platform == "darwin":
        os.environ["DYLD_LIBRARY_PATH"] = f"{lib}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
    elif sys.platform.startswith("linux"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # IPC handshake: Webots listens on --port=N, the extern controller dials in
    # via WEBOTS_CONTROLLER_URL. Use a non-default port so a casually-running
    # Webots elsewhere on the machine doesn't collide.
    port = os.environ.get("WEBOTS_PORT", "1234")
    os.environ["WEBOTS_CONTROLLER_URL"] = f"ipc://{port}/{cfg['robot_name']}"

    proc = subprocess.Popen([
        str(webots),
        f"--port={port}",
        "--mode=realtime",
        "--no-rendering",
        "--minimize",
        "--batch",
        "--stdout",
        "--stderr",
        str(world),
    ])

    @atexit.register
    def _cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    return proc
