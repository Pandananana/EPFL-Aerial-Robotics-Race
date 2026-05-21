"""Place RacingGate nodes in the running Webots world from a gates.csv.

The DEF names in `sim/worlds/race.wbt` are `GATE0`..`GATE{N-1}`, in the same
order as the rows of the CSV (which is itself the flight order). Each row's
xyz/theta/width/height fully determines the gate's pose and beam dimensions
— the initial values inside the .wbt file are placeholders that get
overwritten the moment the extern controller attaches.

Conversion between CSV theta and the wbt rotation angle around Z follows the
convention already baked into the world file: `rotation_z = theta - pi/2`.
The dependent beam scales/positions match the formulas in the EPFL
aerial-robotics `set_goal_fields` helper.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.control.gates_csv import RecordedGate

if TYPE_CHECKING:  # pragma: no cover
    from controller import Supervisor

# Beam cross-section (matches the originals in race.wbt / the EPFL helper).
_BEAM_W = 0.06  # depth of each frame beam
_BEAM_H = 0.04  # thickness of each frame beam


def place_gates(supervisor: "Supervisor", gates: list[RecordedGate]) -> None:
    """Move every GATE{i} node to match the corresponding gates[i] row."""
    for i, gate in enumerate(gates):
        node = supervisor.getFromDef(f"GATE{i}")
        if node is None:
            raise RuntimeError(
                f"GATE{i} node not found in the Webots world — "
                f"sim_gates.csv has {len(gates)} gates but the world is missing DEFs"
            )
        _set_gate_fields(node, gate)


def _set_gate_fields(node, gate: RecordedGate) -> None:
    x, y, z = float(gate.center[0]), float(gate.center[1]), float(gate.center[2])
    width = float(gate.width_m)
    opening_h = float(gate.height_m)
    theta = math.atan2(float(gate.normal[1]), float(gate.normal[0]))
    rotation_z = theta - math.pi / 2

    w, h = _BEAM_W, _BEAM_H
    leg_len = z + opening_h / 2 + w - h  # vertical beam length from the floor

    node.getField("translation").setSFVec3f([x, y, z])
    node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, rotation_z])
    node.getField("goalSize").setSFVec3f([h, width, opening_h])

    node.getField("topBeamScale").setSFVec3f([width, w, h])
    node.getField("topBeamTranslation").setSFVec3f([0.0, 0.0, opening_h / 2 + w / 2])
    node.getField("bottomBeamScale").setSFVec3f([width, w, h])
    node.getField("bottomBeamTranslation").setSFVec3f([0.0, 0.0, -opening_h / 2 - w / 2])

    node.getField("leftBeamScale").setSFVec3f([leg_len, w, h])
    node.getField("leftBeamTranslation").setSFVec3f(
        [0.0, width / 2 + w / 2, h + leg_len / 2 - z]
    )
    node.getField("rightBeamScale").setSFVec3f([leg_len, w, h])
    node.getField("rightBeamTranslation").setSFVec3f(
        [0.0, -width / 2 - w / 2, h + leg_len / 2 - z]
    )

    node.getField("leftLegTranslation").setSFVec3f(
        [0.0, width / 2 + w / 2, h / 2 - z]
    )
    node.getField("rightLegTranslation").setSFVec3f(
        [0.0, -width / 2 - w / 2, h / 2 - z]
    )
