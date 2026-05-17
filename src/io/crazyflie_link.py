"""Crazyflie radio link.

  - Connects over Crazyradio (cflib) to the configured URI.
  - Subscribes to a 10 ms log config and emits DronePose on `pose_ready`.
  - A QTimer at setpoint_rate_hz reads the latest desired Setpoint and
    sends it via send_hover_setpoint. If no Setpoint has been set yet,
    nothing is sent (motors stay disarmed).

cflib callbacks arrive on cflib's own thread; Qt auto-queues `pose_ready`
emissions to whatever thread the receiver QObject lives on. The setpoint
timer runs on whatever thread this object lives on (typically main).
"""

from __future__ import annotations

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Setpoint


class CrazyflieLink(QtCore.QObject):
    pose_ready = QtCore.pyqtSignal(object)  # DronePose
    connected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        uri: str,
        cache_dir: str,
        setpoint_rate_hz: float = 10.0,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._uri = uri
        self._setpoint: Latest[Setpoint] = Latest()

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(rw_cache=cache_dir)
        # Bounce cflib's connected callback (fires on cflib thread) through a
        # Qt signal so the rest of setup runs on this object's thread.
        self.cf.connected.add_callback(self.connected.emit)
        self.connected.connect(self._on_connected)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._send_setpoint)
        self._timer.setInterval(int(1000.0 / setpoint_rate_hz))

    @QtCore.pyqtSlot(object)
    def set_setpoint(self, sp: Setpoint) -> None:
        """Push a desired setpoint. The radio timer will send the latest
        one on its next tick."""
        self._setpoint.set(sp)

    @QtCore.pyqtSlot()
    def send_stop(self) -> None:
        """Immediately send a stop setpoint (motors off). Use for E-stop."""
        self.cf.commander.send_stop_setpoint()

    def open(self) -> None:
        self.cf.open_link(self._uri)

    def close(self) -> None:
        self._timer.stop()
        try:
            self.cf.commander.send_stop_setpoint()
        finally:
            self.cf.close_link()

    def _on_connected(self, uri: str) -> None:
        self.cf.supervisor.send_arming_request(True)
        self._timer.start()
        self._start_logging()

    def _start_logging(self) -> None:
        lg = LogConfig(name="State", period_in_ms=10)
        lg.add_variable("stateEstimate.x", "float")
        lg.add_variable("stateEstimate.y", "float")
        lg.add_variable("stateEstimate.z", "float")
        lg.add_variable("stabilizer.roll", "float")
        lg.add_variable("stabilizer.pitch", "float")
        lg.add_variable("stabilizer.yaw", "float")
        self.cf.log.add_config(lg)
        lg.data_received_cb.add_callback(self._on_log)
        lg.start()

    def _on_log(self, timestamp_ms, data, _logconf) -> None:
        import time
        pose = DronePose(
            timestamp=time.time(),
            x=data["stateEstimate.x"],
            y=data["stateEstimate.y"],
            z=data["stateEstimate.z"],
            roll=data["stabilizer.roll"],
            pitch=data["stabilizer.pitch"],
            yaw=data["stabilizer.yaw"],
        )
        self.pose_ready.emit(pose)

    def _send_setpoint(self) -> None:
        sp = self._setpoint.get()
        if sp is None:
            return
        self.cf.commander.send_hover_setpoint(sp.vx, sp.vy, sp.yaw_rate, sp.height)
