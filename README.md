# Setup Instructions

1. Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync`

## Connecting to the drone

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
