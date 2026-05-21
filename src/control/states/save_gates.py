"""Persist the recon-lap gate measurements to disk, then hand off to takeoff.

This sits between the recon-phase LandState and the second TakeoffState (the
one that leads into RaceState). On entry we serialize the recorded gates to
the configured CSV path so a subsequent race-only run can consume it. Then
we hold the drone on the ground for HOLD_S so the operator sees the pause,
and pass control to the next state in the chain.

The motors stay armed throughout — LandState only fires `notify_mission_done`
when it has no follow-on, and the recon-phase Land was instantiated with
`then=SaveGatesState(...)`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.control.gates_csv import RecordedGate, save_gates_csv
from src.control.states.base import Context, State

logger = logging.getLogger(__name__)


class SaveGatesState(State):
    HOLD_S = 1.5
    GROUND_SPEED_MPS = 0.1

    def __init__(
        self,
        *,
        gates: list[RecordedGate],
        save_path: Path | None,
        then: State,
    ) -> None:
        self._gates = gates
        self._save_path = Path(save_path) if save_path is not None else None
        self._then = then
        self._saved = False
        self._start_t: float | None = None
        self._hold_z: float | None = None

    def tick(self, ctx: Context) -> State | None:
        if not self._saved:
            if self._save_path is None:
                logger.warning("No gates_save_path configured; skipping CSV write")
            else:
                save_gates_csv(self._save_path, self._gates)
                logger.info("Wrote %d gates to %s", len(self._gates), self._save_path)
            self._saved = True
            self._start_t = ctx.pose.timestamp
            self._hold_z = ctx.pose.z

        # Hold on the ground briefly so the operator can see the pause.
        assert self._start_t is not None and self._hold_z is not None
        ctx.emit(
            ctx.start_x, ctx.start_y, self._hold_z,
            ctx.start_yaw_rad, self.GROUND_SPEED_MPS,
        )

        if ctx.pose.timestamp - self._start_t >= self.HOLD_S:
            logger.info("Hold complete; -> %s", type(self._then).__name__)
            return self._then
        return None
