"""Protocols for swappable IO backends.

The integrated system in src/main.py is driven by two backends:

- A `VideoSource` emits camera frames on `frame_ready` and is started once.
- A `DroneLink` emits drone pose on `pose_ready`, accepts setpoints via
  `set_setpoint` / `send_stop`, and fires `connected` once when the link
  is ready so the UI can update its status.

Implementations:

- `live`   : UdpVideoThread (video) + CrazyflieLink (link). Real hardware.
- `replay` : a single ReplayThread serves both roles; set_setpoint and
             send_stop are no-ops because there is no drone to command —
             setpoints from the controller / manual control are dropped
             on the floor.
- `webots` : (planned) a Webots-backed object that exposes a sim camera
             as the video source and pose+setpoint sink as the link.

Backends are duck-typed; anything with the right Qt signals and methods
works. These classes exist as documentation and for static type checkers.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.messages import Setpoint


class VideoSource(Protocol):
    """Emits Frame messages on `frame_ready`. Started via `start()`."""

    frame_ready: Any  # pyqtSignal(Frame)

    def start(self) -> None: ...


class DroneLink(Protocol):
    """Pose + control interface.

    `pose_ready` emits DronePose. `connected` fires once when the link is
    usable (UI listens for status). `set_setpoint` / `send_stop` are slots
    the autonomous controller and manual control write into.
    """

    pose_ready: Any  # pyqtSignal(DronePose)
    connected: Any  # pyqtSignal(str)

    def open(self) -> None: ...
    def close(self) -> None: ...
    def set_setpoint(self, sp: Setpoint) -> None: ...
    def send_stop(self) -> None: ...
