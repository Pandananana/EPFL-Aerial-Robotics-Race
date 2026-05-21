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
brew install libusb
```

If your Homebrew installation is in a non-default location, you might need to link the libusb library:

```bash
export DYLD_LIBRARY_PATH="YOUR_HOMEBREW_PATH/lib:$DYLD_LIBRARY_PATH"
```

## Starting Crazyflie client

Make sure you have `uv` installed, then run

```bash
uvx cfclient
```

## Running the system

One entry point, three IO backends — chosen by `--source`:

```bash
make live                                            # AI-deck + Crazyflie over radio
make webots                                          # Webots sim (extern controller)
make replay REC=data/recordings/<id> [SPEED=1.0]     # play back a recording, perception only
make no-fly                                          # live IO with arming + setpoints disabled
make race GATES=data/gates/<file>.csv                # race-only on hardware: skip recon, fly the supplied gates
make race-webots                                     # race-only in Webots (uses data/gates/sim_gates.csv)
make measure                                         # capture gate-corner ground-truth positions
```

Equivalent without `make`:

```bash
uv run python -m src.main --source live
uv run python -m src.main --source webots
uv run python -m src.main --source replay --recording data/recordings/<id> [--speed 1.0] [--replay-step]
uv run python -m src.main --source live --no-fly
uv run python -m src.main --source live --race-only --true-gates data/gates/<file>.csv
uv run python -m src.main --source webots --race-only
uv run python -m scripts.measure_gates
```

Extra flags that work with any `--source`:

- `--debug` — open the 3D gate-debug viewer (true vs. measured gates). Needs a truth source: `--true-gates` for live, `<recording>/gates.csv` fallback for replay, `data/gates/sim_gates.csv` automatically in webots.
- `--replay-step` (replay only) — wait for a keypress in the FPV window before emitting each recorded pose/frame row.
- `--true-gates <csv>` — ground-truth gates. Feeds the 3D debug plot, and with `--race-only` is loaded as the preloaded gate set. Ignored under `--source webots` (always uses `data/gates/sim_gates.csv`).
- `--race-only` — skip the recon lap and drop straight from takeoff into the racing trajectory. Requires `--true-gates` for live/replay; webots picks up the sim gates automatically.
- `--no-fly` (live only) — connect to the AI-deck and Crazyflie for video + pose but never arm or send setpoints. Also suppresses the autonomous mission. Use when recording calibration / training frames while holding the drone.

## Labels and training data

Labels live under `data/labels/` and are tracked by `data/splits.json` (the source-of-truth manifest of which labeled frame is in which split). Two flavours, both derived from the same recordings:

- **Seg labels** — hand-traced labelme polygons of each gate's inner LED edge (the hole the drone flies through). Variable vertex count depending on how much of the gate is in-frame. Authored manually; see README for the labeling rulebook.
- **Pose labels** — 4-corner (TL, TR, BR, BL) labels with per-corner visibility flags, generated from the seg polygons. For fully-visible gates the four polygon vertices are taken directly (v=2). For partially in-frame gates the converter either extrapolates the single missing corner from the two edges exiting the frame, or — when two or more corners are off-image — clips them to the image boundary and marks them v=0. Gates with too few interior vertices are skipped.

The YOLO models train off whichever label flavour they need; a shared dataset builder materializes the on-disk YOLO layout from `splits.json`.

### Hand traced labeling conventions

Gate labels are labelme polygons traced on raw flight frames. Pose labels are derived from these polygons by a converter script.

- Each gate is an LED rectangle with equal height, but variable width and pose.
- Trace the **inner edge** of the LED frame (the hole the drone flies through), not the outer bloom or the centerline.
- Click order is clockwise starting from the upper-left corner.
- If the whole gate hole is visible, the polygon has 4 points. If the gate is partially out of frame, point count varies (typically 3–6+); polygon vertices on the image boundary are treated as clip points, not real corners.
- Skip a gate when it is near-edge-on, partially blocked from view, or any of its corners are off-image and the gate is also otherwise hard to interpret.
- An image with no labelable gates is still saved with `{"shapes": []}` — this distinguishes "reviewed, empty" from "not yet reviewed".
