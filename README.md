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

## Labeling workflow

```bash
uv run python tools/sample_to_label.py recordings/<run> --n 50
uvx labelme to_label --output recordings/<run> --labels gate --nodata --autosave
uv run python tools/finalize_no_gates.py recordings/<run>
uv run python tools/build_splits.py
```

`sample_to_label.py` symlinks a random subset of _unlabeled_ PNGs into `to_label/` (gitignored). `--output` makes labelme write the JSONs back next to the originals. After the batch, `finalize_no_gates.py` writes empty-shape JSONs for any sampled image you skipped past. Finally rebuild the manifest.
