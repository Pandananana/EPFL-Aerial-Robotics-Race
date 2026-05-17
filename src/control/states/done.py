"""Terminal state. No further waypoints are emitted.

The link has already been told to stop, so the drone is either on the pad
(sim) or coasting down its descent (real hardware). Staying here is a
no-op rather than a guarded "do nothing" branch elsewhere.
"""

from __future__ import annotations

from src.control.states.base import Context, State


class DoneState(State):
    def tick(self, ctx: Context) -> State | None:
        return None
