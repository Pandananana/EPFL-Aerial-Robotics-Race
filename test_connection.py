"""Verify cflib can connect to our Crazyflie."""
import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = "radio://0/80/2M/E7E7E7E718"


def main() -> int:
    cflib.crtp.init_drivers()
    print(f"Connecting to {URI} ...")
    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
            fw = scf.cf.param.get_value("firmware.revision0")
            print(f"Connected. firmware.revision0 = {fw}")
        print("Disconnected cleanly.")
        return 0
    except Exception as e:
        print(f"Connection failed: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
