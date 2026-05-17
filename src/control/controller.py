"""Maps the current waypoint + drone pose to a Setpoint.

STUB — owned by the controls team. The current implementation caches
the latest DronePose and emits nothing; CrazyflieLink will not move the
drone until `setpoint_ready` actually fires. While this is unwired,
ManualControl is the only thing publishing setpoints.

Output Setpoint follows the cflib send_hover_setpoint format:
  vx, vy  (body-frame velocities, m/s)
  yaw_rate (deg/s)
  height   (absolute z, m)
"""

from __future__ import annotations

from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Setpoint


class Controller(QtCore.QObject):
    setpoint_ready = QtCore.pyqtSignal(object)  # Setpoint

    def __init__(self, *, default_height_m: float, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._pose: Latest[DronePose] = Latest()
        self._default_height = default_height_m

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)

    @QtCore.pyqtSlot(object)
    def on_waypoint(self, waypoint: object) -> None:
        # Fill in once Waypoint is defined and a real controller is implemented.
        # Example:
        #   sp = Setpoint(vx=..., vy=..., yaw_rate=..., height=...)
        #   self.setpoint_ready.emit(sp)
        del waypoint
