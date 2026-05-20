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

LIGHTHOUSE_BS_AVAILABLE = "lighthouse.bsAvailable"


class CrazyflieLink(QtCore.QObject):
    pose_ready = QtCore.pyqtSignal(object)  # DronePose
    connected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        uri: str,
        cache_dir: str,
        setpoint_rate_hz: float = 10.0,
        disable_flight: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._uri = uri
        self._setpoint: Latest[Setpoint] = Latest()
        self._disable_flight = disable_flight
        self._lighthouse_available_var: str | None = None

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
        if self._disable_flight:
            return
        self._setpoint.set(sp)

    @QtCore.pyqtSlot()
    def send_stop(self) -> None:
        """Immediately send a stop setpoint (motors off). Use for E-stop."""
        if self._disable_flight:
            return
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
        # In disable_flight mode we still want pose logging (useful while
        # recording), but we must not arm or run the setpoint loop.
        if not self._disable_flight:
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
        if self._has_log_variable(LIGHTHOUSE_BS_AVAILABLE):
            self._lighthouse_available_var = LIGHTHOUSE_BS_AVAILABLE
            lg.add_variable(LIGHTHOUSE_BS_AVAILABLE, "uint16_t")
            print(f"[LIGHTHOUSE] logging {LIGHTHOUSE_BS_AVAILABLE}", flush=True)
        else:
            toc = self.cf.log.toc
            available = sorted(toc.toc.get("lighthouse", {}).keys()) if toc else []
            print(
                "[LIGHTHOUSE] lighthouse.bsAvailable not in log TOC; "
                f"available lighthouse logs: {available}",
                flush=True,
            )
        self.cf.log.add_config(lg)
        lg.data_received_cb.add_callback(self._on_log)
        lg.start()

    def _on_log(self, timestamp_ms, data, _logconf) -> None:
        import time
        bs_visible = None
        if self._lighthouse_available_var is not None:
            bs_visible = int(data[self._lighthouse_available_var]).bit_count()
        pose = DronePose(
            timestamp=time.time(),
            x=data["stateEstimate.x"],
            y=data["stateEstimate.y"],
            z=data["stateEstimate.z"],
            roll=data["stabilizer.roll"],
            pitch=data["stabilizer.pitch"],
            yaw=data["stabilizer.yaw"],
            lighthouse_bs_visible=bs_visible,
        )
        self.pose_ready.emit(pose)

    def _has_log_variable(self, name: str) -> bool:
        if self.cf.log.toc is None:
            return False
        group, var = name.split(".", 1)
        return self.cf.log.toc.get_element(group, var) is not None

    def _send_setpoint(self) -> None:
        sp = self._setpoint.get()
        if sp is None:
            return
        self.cf.commander.send_hover_setpoint(sp.vx, sp.vy, sp.yaw_rate, sp.height)
