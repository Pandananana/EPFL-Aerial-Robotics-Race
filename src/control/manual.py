"""Keyboard manual control — keypresses -> Setpoint.

Lives alongside the autonomous Controller; both connect to
CrazyflieLink.set_setpoint and the last writer wins on the radio's
setpoint timer. Arbitration (e.g. "ignore controller while a key is
held") is the controls team's call — wire it however you want in
src/main.py.

Doesn't capture keyboard events directly; the UI hands them in via
handle_key_press / handle_key_release. Keeps this module Qt-widget-free
so it can be unit tested without a display.

Keys: arrows = pitch/roll, A/D = yaw, W/S = up/down, Space = stop.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui

from src.messages import Setpoint


class ManualControl(QtCore.QObject):
    setpoint_ready = QtCore.pyqtSignal(object)  # Setpoint
    stop_requested = QtCore.pyqtSignal()

    HEIGHT_STEP_M = 0.1

    def __init__(
        self,
        *,
        speed_mps: float,
        yaw_rate_dps: float,
        default_height_m: float,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._speed = speed_mps
        self._yaw_rate = yaw_rate_dps
        self._held: set[int] = set()
        self._hover = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "height": default_height_m}

    def handle_key_press(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        k = event.key()
        K = QtCore.Qt.Key
        if k == K.Key_Space:
            self.stop_requested.emit()
            return
        if k == K.Key_W:
            self._hover["height"] += self.HEIGHT_STEP_M
            self._publish()
            return
        if k == K.Key_S:
            self._hover["height"] -= self.HEIGHT_STEP_M
            self._publish()
            return
        self._held.add(k)
        self._update_velocity()

    def handle_key_release(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        self._held.discard(event.key())
        self._update_velocity()

    def _update_velocity(self) -> None:
        K = QtCore.Qt.Key
        vx = (K.Key_Up in self._held) * self._speed - (K.Key_Down in self._held) * self._speed
        vy = (K.Key_Left in self._held) * self._speed - (K.Key_Right in self._held) * self._speed
        yaw_rate = (K.Key_D in self._held) * self._yaw_rate - (K.Key_A in self._held) * self._yaw_rate
        self._hover["vx"], self._hover["vy"], self._hover["yaw_rate"] = vx, vy, yaw_rate
        self._publish()

    def _publish(self) -> None:
        self.setpoint_ready.emit(Setpoint(
            vx=self._hover["vx"],
            vy=self._hover["vy"],
            yaw_rate=self._hover["yaw_rate"],
            height=self._hover["height"],
        ))
