# epfl-drone-race

Perception + control for an EPFL drone-racing project. A Crazyflie with an AI-deck streams grayscale 324×244 frames over UDP; the system detects rectangular LED gates, flies through them, and races.

## Race goal

One mission, three phases:

1. **Reconnaissance lap.** Take off from the ground, detect gates online, fly through each one in order, return to start, land. Gate 3D estimates from this lap are cached.
2. **Racing laps (×2).** Fly the same circuit as fast as possible, using the cached gate estimates as a prior — perception doesn't run, the planner makes an optimal path based on the known gate estimates.
3. Land at start.

## Architecture

The system is a set of small `QObject` modules wired together with Qt signals. Every module has one job and a stable message contract — the dataclasses in `src/messages.py` (`Frame`, `DronePose`, `GateDetection2D`, `Gate3D`, `Setpoint`). If a contract changes, every subscriber breaks; that's the point.

The core pipeline is mode-independent:

```
VideoSource ──frame──▶ GateDetector ──2D──▶ PoseEstimator ──3D──▶ Planner ──▶ Controller ──setpoint──▶ DroneLink
DroneLink   ──pose──▶ Planner, Controller (and Recorder, when recording)
```

`VideoSource` and `DroneLink` are duck-typed protocols (see `src/io/sources.py`). The same perception + control graph runs against any backend that satisfies them.

## Three modes, one pipeline

`src/main.py --source {live,replay,webots}` selects the IO backend; nothing downstream changes.

- **live** — `UdpVideoThread` (AI-deck JPEG stream) + `CrazyflieLink` (cflib radio). Real hardware. Recording is on by default.
- **replay** — a single `ReplayThread` plays back a recording directory as both video and pose source. `set_setpoint`/`send_stop` are no-ops; controller output is dropped. Useful for re-running perception on logged flights.
- **webots** — a single `WebotsBackend` attaches to a running Webots sim as an extern controller. It feeds the simulator's camera + sensors into the same pipeline and runs an in-process cascaded PID that converts hover `Setpoint`s into rotor PWMs. The sim world uses emissive pink-panel gates instead of LED rectangles, so the config forces the `pink` HSV detector to avoid the domain gap.

Swapping detectors is the same shape: each model under `src/perception/models/<name>/` exposes `predict_gates(image) -> list[(4,2) np.ndarray]`, and the active one is a config string. Current detectors include classical (Hough, pink HSV) and fine-tuned ultralytics (YOLO seg / pose / OBB) variants.

## Two persistence patterns

- **Qt signals** — every consumer reacts to every message. Use for frames, detections, poses flowing through the pipeline.
- `**Latest[T]`\*\* (`src/bus.py`) — single-slot latch holding the most recent value. Use when a consumer polls at its own rate (e.g. the controller reading the latest pose).
