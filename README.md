# raspberry-pi-ai-camera

Real-time object detection on a **Raspberry Pi 5** using either:

- the **Raspberry Pi AI Camera** (Sony IMX500), where inference runs on the camera sensor's own NPU, or
- the **Raspberry Pi AI HAT+** (Hailo-8L 13 TOPS / Hailo-8 26 TOPS), where inference runs on the Hailo NPU over PCIe.

## Hardware

| Part | Notes |
| --- | --- |
| Raspberry Pi 5 | 64-bit Raspberry Pi OS (Trixie recommended, which is what the official docs target) |
| Raspberry Pi AI Camera (IMX500) | Connect to a CAM/DISP port. It's detected automatically (`camera_auto_detect=1`) |
| Raspberry Pi AI HAT+ | PCIe ribbon cable + GPIO stacking header. AI HAT+ 2 (Hailo-10H) is supported via `--hailo-h10` |
| Active Cooler | Recommended |

Attach the camera first, then the HAT, with the Pi powered off.

## Setup

```bash
git clone https://github.com/imteevo/raspberry-pi-ai-camera.git
cd raspberry-pi-ai-camera
sudo ./scripts/setup.sh            # AI HAT+ / AI Kit  -> hailo-all
# sudo ./scripts/setup.sh --hailo-h10   # AI HAT+ 2 -> hailo-h10-all
sudo reboot
./scripts/verify.sh
```

`setup.sh` is idempotent. It:

1. Checks that it's running on a Raspberry Pi 5 with 64-bit OS.
2. Runs `apt update`, `apt full-upgrade` and `rpi-eeprom-update -a` (skip these with `--no-upgrade`).
3. Installs `dkms`, `hailo-all` (or `hailo-h10-all`), `imx500-all`, `rpicam-apps`, `python3-picamera2`, `python3-opencv`, `python3-yaml` and `python3-numpy`.
4. Makes sure `/boot/firmware/config.txt` contains `dtparam=pciex1_gen=3` and `camera_auto_detect=1`. It saves a backup as `config.txt.bak-ai-camera`.
5. Asks you to reboot if anything changed.

`verify.sh` checks the following:

- `rpicam-hello --list-cameras` lists the IMX500.
- The `/lib/firmware/imx500_*.fpk` firmware files exist, and `/usr/share/imx500-models/*.rpk` models are present.
- `hailortcli fw-control identify` responds, and `/dev/hailo0` exists.
- The Python modules import.

### Python environment

picamera2, OpenCV and HailoRT come from apt, so if you use a venv, create it with system site packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# IMX500 on-sensor inference (SSD MobileNetV2 FPN-Lite 320x320)
python -m src.main --backend imx500 --headless

# Hailo AI HAT+ (YOLOv8s), with a preview window
python -m src.main --backend hailo --preview

# Other options
python -m src.main -b imx500 -t 0.6 -n 100 --model /usr/share/imx500-models/<model>.rpk
```

Example output:

```
[frame 12 |  29.8 fps] 2 detection(s)
    person           0.87  box=(412, 120, 380, 860)
    cup              0.61  box=(1010, 640, 120, 150)
```

The first run on the IMX500 uploads the network firmware to the sensor, which can take a minute. A progress bar shows while it loads.

## Configuration

`config.yaml` sets the backend, preview, threshold, max detections, the optional labels file, and the model path for each backend. CLI flags override it. If you have a Hailo-8 (26 TOPS) HAT, change `hailo.model` to an `_h8.hef` model.

## Layout

```
scripts/setup.sh    installer
scripts/verify.sh   hardware/software checks
src/config.py       YAML config loader (dataclasses)
src/camera.py       Picamera2 + IMX500 object-detection backend
src/hailo.py        Picamera2 + Hailo AI HAT+ backend
src/detection.py    shared Detection type
src/main.py         CLI
config.yaml         default configuration
```

## References

- [AI Camera docs](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [AI HAT+ docs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [AI software (Hailo) docs](https://www.raspberrypi.com/documentation/computers/ai.html)
- [picamera2 IMX500 examples](https://github.com/raspberrypi/picamera2/tree/main/examples/imx500)
