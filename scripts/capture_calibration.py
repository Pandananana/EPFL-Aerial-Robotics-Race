"""Live calibration-capture tool.

Opens the AI-deck UDP video stream, runs chessboard corner detection on
each frame, overlays the detected corners on a preview window, and writes
the raw frame to disk when you press SPACE. Use this to build a directory
of calibration images, then feed it to scripts/calibrate_camera.py.

Topology mirrors src/main.py:

    UdpVideoThread.frame_ready --+--> CornerDetector.on_frame
                                 +--> CaptureWindow.on_frame
    CornerDetector.corners_ready ---> CaptureWindow.on_corners

CornerDetector lives on its own QThread; detection on 324x244 is fast
but the cost of a missed quad search is unbounded, and keeping it off
the UI thread keeps the preview smooth.

Run (no Crazyflie radio needed — only the AI-deck video link):

    uv run python scripts/capture_calibration.py
    uv run python scripts/capture_calibration.py \\
        --output data/calibration_run1 --inner-corners 6 8

Keys:
    SPACE  save the latest frame (refuses if corners aren't currently detected)
    F      force-save even when corners aren't detected
    Q/Esc  quit
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt6 import QtCore, QtGui, QtWidgets

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `src` importable when this script is run directly.
sys.path.insert(0, str(REPO_ROOT))

from src.io.live.video import UdpVideoThread  # noqa: E402
from src.messages import Frame  # noqa: E402


class CornerDetector(QtCore.QObject):
    """Runs cv2.findChessboardCornersSB on each incoming frame and emits
    the result (or None) keyed by frame.seq, so the window can pair the
    overlay with the right image."""

    corners_ready = QtCore.pyqtSignal(int, object)  # (frame.seq, corners or None)

    def __init__(self, pattern_size: tuple[int, int], parent=None):
        super().__init__(parent)
        self._pattern_size = pattern_size

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        img = frame.image
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # No EXHAUSTIVE/ACCURACY flags here — those are for the offline
        # calibration pass. Live preview prioritises responsiveness.
        found, corners = cv2.findChessboardCornersSB(gray, self._pattern_size)
        self.corners_ready.emit(frame.seq, corners if found else None)


class CaptureWindow(QtWidgets.QWidget):
    """Preview window. Buffers recent frames and only paints once their
    matching detection arrives — exactly the same pairing pattern as
    src/ui/fpv_window.py, so the overlay always lines up with the image
    underneath, even if detection lags video by a frame or two."""

    SCALE = 2
    FRAME_BUFFER = 16

    def __init__(
        self,
        output_dir: Path,
        pattern_size: tuple[int, int],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration capture — SPACE to save, Q to quit")
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pattern_size = pattern_size

        self._frames: OrderedDict[int, Frame] = OrderedDict()
        self._latest_painted: Frame | None = None
        self._latest_corners: np.ndarray | None = None
        self._rgb_buf: np.ndarray | None = None
        self._saved = 0

        self.image_label = QtWidgets.QLabel("Waiting for video...")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label = QtWidgets.QLabel(
            f"Output: {self._output_dir} — saved 0"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status_label)

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        self._frames[frame.seq] = frame
        while len(self._frames) > self.FRAME_BUFFER:
            self._frames.popitem(last=False)

    @QtCore.pyqtSlot(int, object)
    def on_corners(self, frame_seq: int, corners: np.ndarray | None) -> None:
        frame = self._frames.pop(frame_seq, None)
        if frame is None:
            return  # detection arrived after its frame fell out of the buffer
        for seq in [s for s in self._frames if s < frame_seq]:
            del self._frames[seq]
        self._latest_painted = frame
        self._latest_corners = corners
        self._paint(frame, corners)
        self._update_status()

    def _paint(self, frame: Frame, corners: np.ndarray | None) -> None:
        img = frame.image
        h, w = img.shape[:2]
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        if corners is not None:
            cv2.drawChessboardCorners(rgb, self._pattern_size, corners, True)
        self._rgb_buf = np.ascontiguousarray(rgb)
        qimg = QtGui.QImage(
            self._rgb_buf.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        self.image_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg.scaled(w * self.SCALE, h * self.SCALE))
        )

    def _update_status(self) -> None:
        state = (
            "corners DETECTED — SPACE to save"
            if self._latest_corners is not None
            else "no corners — move / rotate the board"
        )
        self.status_label.setText(
            f"{state} | saved {self._saved} -> {self._output_dir}"
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self.close()
            return
        if key == QtCore.Qt.Key.Key_Space:
            if self._latest_corners is None:
                self.status_label.setText(
                    f"NOT saved — no corners detected. Press F to force-save. "
                    f"({self._saved} saved)"
                )
                return
            self._save(self._latest_painted)
            return
        if key == QtCore.Qt.Key.Key_F:
            self._save(self._latest_painted)

    def _save(self, frame: Frame | None) -> None:
        if frame is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self._output_dir / f"calib_{ts}.png"
        cv2.imwrite(str(path), frame.image)
        self._saved += 1
        flagged = "" if self._latest_corners is not None else " (no-corners)"
        self.status_label.setText(
            f"saved {path.name}{flagged} — total {self._saved}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "data" / "calibration"
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Directory to save captured frames. Default: "
             "data/calibration/<timestamp>/",
    )
    ap.add_argument(
        "--inner-corners", type=int, nargs=2, default=[6, 8],
        metavar=("COLS", "ROWS"),
        help="Inner-corner count (cols rows). A 7x9-square board has 6 8. "
             "Default: 6 8.",
    )
    ap.add_argument(
        "--config", type=Path, default=REPO_ROOT / "config" / "default.yaml",
        help="Path to default.yaml — used only for AI-deck network settings.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    pattern = (args.inner_corners[0], args.inner_corners[1])

    app = QtWidgets.QApplication(sys.argv[:1])

    video = UdpVideoThread(
        aideck_ip=cfg["network"]["aideck_ip"],
        aideck_port=cfg["network"]["aideck_port"],
        local_port=cfg["network"]["local_port"],
    )

    detector = CornerDetector(pattern)
    detector_thread = QtCore.QThread()
    detector_thread.setObjectName("CornerDetectorThread")
    detector.moveToThread(detector_thread)
    detector_thread.start()

    window = CaptureWindow(args.output, pattern)
    window.resize(720, 600)
    window.show()

    video.frame_ready.connect(detector.on_frame)
    video.frame_ready.connect(window.on_frame)
    detector.corners_ready.connect(window.on_corners)

    print(f"Saving frames to {args.output}")
    print("Keys: SPACE save, F force-save, Q/Esc quit")
    video.start()
    try:
        return app.exec()
    finally:
        detector_thread.quit()
        detector_thread.wait()
        print(f"Done — {window._saved} frame(s) in {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
