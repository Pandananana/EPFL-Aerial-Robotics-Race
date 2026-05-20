"""Image-Based Visual Servoing planner.

The camera image is the source of truth: control errors live in pixels, not
metres. No PnP, no 3D estimation, no reliance on the drone pose for gate-
related decisions. As long as the locked target's corners look reasonable,
the proportional control law produces sensible hover Setpoints.

State machine:

    TAKEOFF  -> rise to the configured altitude
    SEARCH   -> yaw slowly until a confident track persists
    APPROACH -> IBVS on the locked track
    COMMIT   -> brief open-loop forward burst when the gate fills the frame
                (corners straddle opposite edges, i.e. drone is inside it)
    DONE     -> motors stopped after the configured number of gates

The tracker is a tiny in-module greedy centroid matcher — it works for any
detector that emits `GateDetection2D`. When the YOLO pose model is the
detector, `model.track` would produce nicer IDs, but the control law only
needs *some* persistent ID and a small detector is enough.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass

import numpy as np
from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Frame, GateDetection2D, Setpoint

logger = logging.getLogger(__name__)


@dataclass
class _Track:
    id: int
    centroid: np.ndarray       # (2,) pixel xy
    corners: np.ndarray        # (4, 2)
    area: float
    age: int = 1               # frames matched (including this one)
    missed: int = 0            # consecutive frames missed
    last_seen_t: float = 0.0


def _polygon_area(corners: np.ndarray) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _trapezoidal_asymmetry(corners: np.ndarray) -> float:
    """Normalised difference between the two vertical edges, TL,TR,BR,BL.

    Positive when the left edge is longer than the right (drone is to the
    right of the gate's optical axis, so the right side is farther away).
    Head-on returns ~0.
    """
    left = float(np.linalg.norm(corners[3] - corners[0]))   # TL -> BL
    right = float(np.linalg.norm(corners[2] - corners[1]))  # TR -> BR
    return (left - right) / (0.5 * (left + right) + 1e-6)


def _aspect_ratio(corners: np.ndarray) -> float:
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bot = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    h = 0.5 * (left + right) + 1e-6
    w = 0.5 * (top + bot) + 1e-6
    return w / h


class _TrackManager:
    """Greedy centroid-distance tracker. One pass per detection."""

    MATCH_DIST_PX = 80.0
    MAX_MISSED = 6
    CENTROID_SMOOTH = 0.4   # 0 = ignore measurement, 1 = no smoothing
    CORNER_SMOOTH = 0.5

    def __init__(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._next_id: int = 1

    def update(self, det: GateDetection2D) -> list[_Track]:
        new_centroids = np.array(
            [c.mean(axis=0) for c in det.corners_px], dtype=np.float32
        ) if det.corners_px else np.zeros((0, 2), dtype=np.float32)

        assigned = [False] * len(det.corners_px)
        # Match older / longer-lived tracks first so a new noisy detection
        # can't steal an established ID.
        for t in sorted(self._tracks.values(), key=lambda x: -x.age):
            if not det.corners_px:
                t.missed += 1
                continue
            dists = np.linalg.norm(new_centroids - t.centroid, axis=1)
            order = np.argsort(dists)
            picked = False
            for j in order:
                if assigned[j]:
                    continue
                if dists[j] > self.MATCH_DIST_PX:
                    break
                corners = det.corners_px[j].astype(np.float32)
                t.centroid = (
                    (1 - self.CENTROID_SMOOTH) * t.centroid
                    + self.CENTROID_SMOOTH * new_centroids[j]
                )
                t.corners = (
                    (1 - self.CORNER_SMOOTH) * t.corners
                    + self.CORNER_SMOOTH * corners
                )
                t.area = _polygon_area(t.corners)
                t.age += 1
                t.missed = 0
                t.last_seen_t = det.timestamp
                assigned[j] = True
                picked = True
                break
            if not picked:
                t.missed += 1

        for j, hit in enumerate(assigned):
            if hit:
                continue
            corners = det.corners_px[j].astype(np.float32)
            new_id = self._next_id
            self._next_id += 1
            self._tracks[new_id] = _Track(
                id=new_id,
                centroid=new_centroids[j].astype(np.float32),
                corners=corners.copy(),
                area=_polygon_area(corners),
                last_seen_t=det.timestamp,
            )

        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.missed <= self.MAX_MISSED
        }
        return list(self._tracks.values())

    def get(self, track_id: int | None) -> _Track | None:
        if track_id is None:
            return None
        return self._tracks.get(track_id)

    def reset(self) -> None:
        self._tracks.clear()


class _Phase(enum.Enum):
    TAKEOFF = "TAKEOFF"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    COMMIT = "COMMIT"
    DONE = "DONE"


class IBVSPlanner(QtCore.QObject):
    """Self-contained IBVS planner; emits Setpoint directly (no Controller)."""

    setpoint_ready = QtCore.pyqtSignal(object)  # Setpoint
    phase_changed = QtCore.pyqtSignal(str)
    mission_done = QtCore.pyqtSignal()
    locked_track_changed = QtCore.pyqtSignal(object)  # int | None

    # --- Gains ---
    YAW_RATE_GAIN_DPS_PER_PX = 0.7        # deg/s per px of horizontal centroid error
    HEIGHT_GAIN_M_PER_PX_S = 0.006        # m/s per px of vertical centroid error
    LATERAL_GAIN_MPS = 0.6                # m/s per unit of trapezoidal asymmetry
    FORWARD_BASE_MPS = 0.45               # baseline forward velocity when on-target
    FORWARD_OFFCENTER_PENALTY = 1.4       # scaling factor on the off-center penalty

    # --- Limits ---
    MAX_YAW_RATE_DPS = 90.0
    MAX_LATERAL_MPS = 0.5
    MAX_FORWARD_MPS = 0.6
    MIN_HEIGHT_M = 0.2
    MAX_HEIGHT_M = 2.0

    # --- Phase thresholds ---
    SEARCH_YAW_RATE_DPS = 35.0
    SEARCH_MIN_AGE = 3                    # frames a track must persist to lock
    COMMIT_AREA_FRAC = 0.30               # gate must fill this much of the frame
    COMMIT_DURATION_S = 1.0
    COMMIT_FORWARD_MPS = 0.7

    LOST_HOLD_S = 0.5
    LOST_SLOW_S = 1.2
    LOST_RESET_S = 2.5

    # --- Takeoff ---
    TAKEOFF_TOLERANCE_M = 0.08
    TAKEOFF_SETTLE_S = 1.0

    def __init__(
        self,
        *,
        default_height_m: float,
        n_gates: int,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_height = default_height_m
        self._height_target = default_height_m
        self._n_gates = n_gates

        self._tracker = _TrackManager()
        self._latest_pose: Latest[DronePose] = Latest()
        self._frame_shape: tuple[int, int] | None = None

        self._phase: _Phase | None = None
        self._locked_id: int | None = None
        self._gates_done = 0
        self._commit_until_t: float | None = None
        self._takeoff_settled_at: float | None = None
        self._last_seen_t: float | None = None
        self._last_setpoint: Setpoint | None = None
        self._last_tick_t: float | None = None
        self._last_big_centered: bool = False

    # --- Lifecycle ---

    @QtCore.pyqtSlot()
    def start(self) -> None:
        if self._phase is not None:
            return
        self._phase = _Phase.TAKEOFF
        self._gates_done = 0
        self._locked_id = None
        self._height_target = self._default_height
        self._takeoff_settled_at = None
        self._tracker.reset()
        logger.info("IBVS mission started (target %d gates)", self._n_gates)
        self.phase_changed.emit(self._phase.value)

    # --- Inputs ---

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._latest_pose.set(pose)
        if self._phase is _Phase.TAKEOFF:
            self._tick_takeoff(pose)

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        h, w = frame.image.shape[:2]
        self._frame_shape = (h, w)

    @QtCore.pyqtSlot(object)
    def on_detection(self, det: GateDetection2D) -> None:
        tracks = self._tracker.update(det)
        if self._phase is None or self._frame_shape is None:
            return
        if self._phase is _Phase.TAKEOFF or self._phase is _Phase.DONE:
            return
        h, w = self._frame_shape
        now = det.timestamp

        if self._phase is _Phase.SEARCH:
            self._tick_search(tracks, w, h, now)
        elif self._phase is _Phase.APPROACH:
            self._tick_approach(tracks, w, h, now)
        elif self._phase is _Phase.COMMIT:
            self._tick_commit(now)

    # --- Phases ---

    def _tick_takeoff(self, pose: DronePose) -> None:
        self._emit(Setpoint(
            vx=0.0, vy=0.0, yaw_rate=0.0,
            height=self._height_target,
        ))
        if abs(pose.z - self._default_height) > self.TAKEOFF_TOLERANCE_M:
            self._takeoff_settled_at = None
            return
        if self._takeoff_settled_at is None:
            self._takeoff_settled_at = pose.timestamp
        elif pose.timestamp - self._takeoff_settled_at >= self.TAKEOFF_SETTLE_S:
            logger.info("Takeoff complete (z=%.2f m); SEARCH", pose.z)
            self._transition(_Phase.SEARCH)

    def _tick_search(
        self, tracks: list[_Track], w: int, h: int, now: float
    ) -> None:
        candidate = self._pick_initial_target(tracks, w, h)
        if candidate is not None:
            self._lock(candidate.id)
            self._last_seen_t = now
            self._transition(_Phase.APPROACH)
            return
        self._emit(Setpoint(
            vx=0.0, vy=0.0,
            yaw_rate=self.SEARCH_YAW_RATE_DPS,
            height=self._height_target,
        ))

    def _tick_approach(
        self, tracks: list[_Track], w: int, h: int, now: float
    ) -> None:
        target = self._tracker.get(self._locked_id)
        if target is None:
            # Detector drops the gate at point-blank range; if the last good
            # snapshot was big and roughly centered, the lock was lost
            # *because* we're inside the gate — commit instead of retreating.
            if self._last_big_centered:
                self._commit_until_t = now + self.COMMIT_DURATION_S
                self._transition(_Phase.COMMIT)
                return
            self._handle_lost(now)
            return
        self._last_seen_t = now

        frame_area = float(w * h)
        area_frac = target.area / frame_area
        if (
            area_frac >= self.COMMIT_AREA_FRAC
            and self._corners_straddle(target.corners, w, h)
        ):
            self._commit_until_t = now + self.COMMIT_DURATION_S
            self._transition(_Phase.COMMIT)
            return

        cx, cy = w / 2.0, h / 2.0
        # Track "we were close" so we can still commit if the detector blinks
        # out at point-blank range. Updated before computing the new command.
        self._last_big_centered = (
            area_frac >= 0.6 * self.COMMIT_AREA_FRAC
            and abs(float(target.centroid[0]) - cx) < 0.30 * w
            and abs(float(target.centroid[1]) - cy) < 0.30 * h
        )

        err_x = float(target.centroid[0]) - cx
        err_y = float(target.centroid[1]) - cy

        dt = max(1e-3, now - (self._last_tick_t or now))
        self._last_tick_t = now

        yaw_rate = -self.YAW_RATE_GAIN_DPS_PER_PX * err_x
        yaw_rate = float(np.clip(yaw_rate, -self.MAX_YAW_RATE_DPS, self.MAX_YAW_RATE_DPS))

        # Image +y points DOWN; if the gate centroid is below the image
        # center we need to descend. Integrate the pixel error into the
        # absolute-height setpoint and clamp to a sane range.
        self._height_target -= self.HEIGHT_GAIN_M_PER_PX_S * err_y * dt
        self._height_target = float(np.clip(
            self._height_target, self.MIN_HEIGHT_M, self.MAX_HEIGHT_M,
        ))

        asym = _trapezoidal_asymmetry(target.corners)
        vy = self.LATERAL_GAIN_MPS * asym
        vy = float(np.clip(vy, -self.MAX_LATERAL_MPS, self.MAX_LATERAL_MPS))

        off_center = math.hypot(err_x / cx, err_y / cy)
        forward_scale = max(0.0, 1.0 - self.FORWARD_OFFCENTER_PENALTY * off_center)
        vx = self.FORWARD_BASE_MPS * forward_scale
        vx = float(np.clip(vx, 0.0, self.MAX_FORWARD_MPS))

        self._emit(Setpoint(
            vx=vx, vy=vy, yaw_rate=yaw_rate, height=self._height_target,
        ))

    def _tick_commit(self, now: float) -> None:
        if self._commit_until_t is not None and now >= self._commit_until_t:
            self._gates_done += 1
            logger.info("Gate %d/%d cleared (IBVS commit)",
                        self._gates_done, self._n_gates)
            self._lock(None)
            self._commit_until_t = None
            # Drop every track — the gate we just flew through is now behind
            # us, and any leftover track ID would be re-picked instantly.
            self._tracker.reset()
            if self._gates_done >= self._n_gates:
                self._emit(Setpoint(
                    vx=0.0, vy=0.0, yaw_rate=0.0, height=self._height_target,
                ))
                self._transition(_Phase.DONE)
                self.mission_done.emit()
                return
            self._transition(_Phase.SEARCH)
            return
        self._emit(Setpoint(
            vx=self.COMMIT_FORWARD_MPS,
            vy=0.0, yaw_rate=0.0,
            height=self._height_target,
        ))

    # --- Helpers ---

    def _handle_lost(self, now: float) -> None:
        if self._last_seen_t is None:
            self._last_seen_t = now
        gap = now - self._last_seen_t
        if gap <= self.LOST_HOLD_S and self._last_setpoint is not None:
            self.setpoint_ready.emit(self._last_setpoint)
            return
        if gap <= self.LOST_SLOW_S:
            self._emit(Setpoint(
                vx=0.1, vy=0.0, yaw_rate=0.0, height=self._height_target,
            ))
            return
        if gap > self.LOST_RESET_S:
            logger.info("Lock lost > %.1fs; returning to SEARCH", self.LOST_RESET_S)
            self._lock(None)
            self._transition(_Phase.SEARCH)
            return
        self._emit(Setpoint(
            vx=0.0, vy=0.0, yaw_rate=0.0, height=self._height_target,
        ))

    def _emit(self, sp: Setpoint) -> None:
        self._last_setpoint = sp
        self.setpoint_ready.emit(sp)

    def _transition(self, phase: _Phase) -> None:
        if self._phase is phase:
            return
        self._phase = phase
        logger.info("IBVS phase -> %s", phase.value)
        self.phase_changed.emit(phase.value)

    def _lock(self, track_id: int | None) -> None:
        if self._locked_id == track_id:
            return
        self._locked_id = track_id
        self._last_big_centered = False
        self.locked_track_changed.emit(track_id)

    def _pick_initial_target(
        self, tracks: list[_Track], w: int, h: int
    ) -> _Track | None:
        if not tracks:
            return None
        cx, cy = w / 2.0, h / 2.0
        frame_area = float(w * h)
        best: _Track | None = None
        best_score = -math.inf
        for t in tracks:
            if t.age < self.SEARCH_MIN_AGE:
                continue
            area_score = min(1.0, t.area / (0.05 * frame_area))
            center_score = 1.0 - min(
                1.0,
                math.hypot((t.centroid[0] - cx) / cx, (t.centroid[1] - cy) / cy),
            )
            ar = _aspect_ratio(t.corners)
            ar_score = math.exp(-((ar - 1.0) ** 2) / 0.5)
            age_score = min(1.0, t.age / 10.0)
            score = (
                0.40 * area_score
                + 0.30 * center_score
                + 0.15 * ar_score
                + 0.15 * age_score
            )
            if score > best_score:
                best_score = score
                best = t
        return best

    def _corners_straddle(self, corners: np.ndarray, w: int, h: int) -> bool:
        """True when the gate quad's corners exit opposite edges — i.e. the
        drone is so close that part of the frame is filled by the gate."""
        margin = 6
        xs, ys = corners[:, 0], corners[:, 1]
        return (
            (xs.min() <= margin and xs.max() >= w - margin)
            or (ys.min() <= margin and ys.max() >= h - margin)
        )
