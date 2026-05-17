# Setup Instructions

1. Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync`

## Connecting to the drone

The address for our drone is

```bash
radio://0/80/2M/E7E7E7E718
```

### Ubuntu/Linux

The Crazyradio is easily recognized on Linux, but you need to set up udev permissions. See the [USB permission instructions](/docs/installation/usb_permissions.md) to configure udev on Ubuntu/Linux.

### Windows

Install the Crazyradio drivers using the [Zadig instructions](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/).

If you're using Python 3.13, you need to install [Visual Studio](https://visualstudio.microsoft.com/downloads/). During the installation process, you only need to select the Desktop Development with C++ workload in the Visual Studio Installer.

### macOS

For Python 3.12+ on macOS, you need to install libusb using Homebrew:

```bash
$ brew install libusb
```

If your Homebrew installation is in a non-default location, you might need to link the libusb library:

```bash
$ export DYLD_LIBRARY_PATH="YOUR_HOMEBREW_PATH/lib:$DYLD_LIBRARY_PATH"
```

## Running Crazyflie Client

The GUI client can be run using

```bash
uvx cfclient
```

## Verifying the connection

A minimal script is provided to check that `cflib` can see and talk to the drone:

```bash
uv run python test_connection.py
```

On success it prints the firmware revision and disconnects cleanly. If you see
`Too many packets lost`, move the drone closer to the Crazyradio and retry.

## Running the live system

The integrated system (UDP video → recording, perception, planning, control)
lives under `src/`. Edit `config/default.yaml` and `config/calibration.yaml`
before flying.

```bash
uv run python scripts/live_viewer.py                                        # live stack: AI-deck + Crazyflie + FPV window
uv run python scripts/live_viewer.py --source replay --recording data/recordings/<run>   # replay through perception + FPV window, no drone
uv run python scripts/replay_log.py data/recordings/<run>                   # replay through perception, print only
```

The viewer is backend-agnostic — IO is wired through the `VideoSource` and
`DroneLink` protocols in `src/io/sources.py`. In replay mode the controller
and manual setpoints are dropped on the floor (no drone to command).

## Running in Webots simulation

For offline iteration without the radio + AI-deck, the same stack can attach
to a Webots simulation as an extern controller. The simulated camera is
converted to the AI-deck's 324×244 grayscale format on emission, so the
perception models see identical input; the control surface is the cflib
hover-setpoint `(vx, vy, yaw_rate, height)`, run through a cascaded PID
inside the backend.

Install [Webots](https://cyberbotics.com/) (R2023b or newer recommended) and
edit `webots.binary` in `config/default.yaml` if it isn't at the default
macOS path. Then:

```bash
uv run python scripts/sim_viewer.py
```

This launches Webots in the background (`--no-rendering --minimize --batch
--mode=realtime`) and runs the integrated stack with `--source webots`. The
FPV window in this process is the only UI; close it to terminate the run
(Webots is cleaned up on exit). Recordings land in `data/recordings/` just
like live flights — useful for generating labeled frames against the sim
gates.

The world (`sim/worlds/race.wbt`) and the racing-gate proto
(`sim/protos/RacingGate.proto`) are adapted from the EPFL aerial-robotics
course. Edit the .wbt to relocate gates or extend the scene.

Module layout:

- `src/messages.py` — shared dataclasses (Frame, DronePose, GateDetection2D, Gate3D, Setpoint)
- `src/bus.py` — `Latest[T]` latch for "most recent value" sharing (Qt signals handle events)
- `src/io/` — backend protocols (`sources.py`), UDP video stream, Crazyflie radio link, disk recorder, recording replay
- `src/perception/` — gate detector (wraps `src/perception/models/`), 3D pose estimator
- `src/perception/models/` — detector backends + shared YOLO dataset builder + committed `best.pt` weights
- `src/control/` — planner, controller (stubs), and keyboard manual override
- `src/ui/` — FPV display window
- `src/main.py` — orchestrator: instantiates modules and wires their signals/slots

## Labeling workflow

```bash
uv run python tools/sample_to_label.py data/recordings/<run> --n 50
uvx labelme to_label --output data/labels/seg/<run> --labels gate --nodata --autosave
uv run python tools/finalize_no_gates.py data/recordings/<run>
uv run python tools/build_splits.py
```

`sample_to_label.py` symlinks a random subset of _unlabeled_ PNGs into `to_label/` (gitignored). The labelme `--output` flag drops JSONs into the parallel `data/labels/seg/<run>/` tree, not next to the images. After the batch, `finalize_no_gates.py` writes empty-shape JSONs for any sampled image you skipped past. Finally rebuild the manifest.

To pre-label new recordings with the current YOLO seg model (only writes sidecars for images that don't already have one), run:

```bash
uv run python tools/auto_label.py
```

## Training and evaluation

Train a YOLO detector (rebuilds the YOLO-format dataset from `data/splits.json`, fine-tunes, copies `best.pt` next to the detector, then evaluates on the test split):

```bash
uv run python -m src.perception.models.yolo_seg.train
uv run python -m src.perception.models.yolo_obb.train
```

Evaluate any detector on the test split:

```bash
uv run python test.py --model {hough,yolo_obb,yolo_seg} --iou 0.5
```

Step through predictions vs. ground truth visually:

```bash
uv run python tools/visualize_preds.py --model yolo_seg
```
