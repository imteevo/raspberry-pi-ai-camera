# raspberry-pi-ai-camera

Real-time object detection on a **Raspberry Pi 5** using either:

- the **Raspberry Pi AI Camera** (Sony IMX500), where inference runs on the camera sensor's own NPU, or
- the **Raspberry Pi AI HAT+** (Hailo-8L 13 TOPS / Hailo-8 26 TOPS) or **AI HAT+ 2** (Hailo-10H), where inference runs on the Hailo NPU over PCIe.

## Hardware

| Part | Notes |
| --- | --- |
| Raspberry Pi 5 | 64-bit Raspberry Pi OS (Trixie recommended, which is what the official docs target) |
| Raspberry Pi AI Camera (IMX500) | Connect to a CAM/DISP port. It's detected automatically (`camera_auto_detect=1`) |
| Raspberry Pi AI HAT+ / AI HAT+ 2 | PCIe ribbon cable + GPIO stacking header. For AI HAT+ 2, run setup with `--hailo-h10` |
| Active Cooler | Recommended |

Attach the camera first, then the HAT, with the Pi powered off.

## Setup

```bash
git clone https://github.com/imteevo/raspberry-pi-ai-camera.git
cd raspberry-pi-ai-camera
sudo ./scripts/setup.sh                 # AI HAT+ / AI Kit  -> hailo-all
# sudo ./scripts/setup.sh --hailo-h10   # AI HAT+ 2         -> hailo-h10-all
sudo reboot
./scripts/verify.sh                     # add --no-hailo if you have no AI HAT
```

`setup.sh` is idempotent. It:

1. Checks that it's running on a Raspberry Pi 5 with 64-bit OS.
2. Runs `apt update`, `apt full-upgrade` and the bootloader EEPROM update. Skip these with `--no-upgrade`.
3. Installs any missing packages: `dkms`, `hailo-all` (or `hailo-h10-all`), `imx500-all`, `rpicam-apps`, `python3-picamera2`, `python3-opencv`, `python3-yaml`, `python3-numpy` and `python3-venv`.
4. Appends one marked block to `/boot/firmware/config.txt` under `[all]`, containing `dtparam=pciex1_gen=3` and `camera_auto_detect=1`. It backs up the original once to `config.txt.bak-ai-camera`.
5. Asks to reboot only when needed: after a new kernel, a driver or firmware package, an EEPROM update, or a config.txt change. Without a terminal, or with `--yes`, it never prompts or reboots.

`verify.sh` checks the following:

- **Camera:** `rpicam-hello --list-cameras` lists the IMX500.
- **IMX500 files:** `/lib/firmware/imx500_loader.fpk`, `/lib/firmware/imx500_firmware.fpk`, and the default `.rpk` model are present.
- **Hailo:** the Hailo kernel module is loaded, `hailortcli fw-control identify` responds, and `/dev/hailo0` exists. It also checks that the PCIe link is at Gen 3 (8 GT/s) and that the YOLOv8 HEFs are installed.
- **Python:** the required modules import.

### Python environment

picamera2, OpenCV, numpy, PyYAML and HailoRT come from **apt**. Install the package without pulling pip copies of them, because pip wheels would shadow the Debian builds that picamera2's C extensions are linked against:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-deps -e .
```

## Usage

```bash
# IMX500 on-sensor inference (SSD MobileNetV2 FPN-Lite 320x320)
pi-ai-camera --backend imx500 --headless

# Hailo NPU; the HEF is chosen automatically for Hailo-8L / Hailo-8 / Hailo-10H
pi-ai-camera --backend hailo --preview

# Other options
pi-ai-camera -b imx500 -t 0.6 -n 100 --model /usr/share/imx500-models/<model>.rpk
pi-ai-camera -b hailo --model /usr/share/hailo-models/yolov8s_h8.hef
```

`python -m pi_ai_camera …` works the same if the entry point isn't on your `PATH`.

Example output:

```
[frame 12 |  29.8 fps] 2 detection(s)
    person           0.87  box=(412, 120, 380, 860)
    cup              0.61  box=(1010, 640, 120, 150)
```

The first run on the IMX500 uploads the network firmware to the sensor, which can take a minute. A progress bar shows while it loads. FPS is shown as a moving average.

Exit codes: `0` means success or Ctrl+C, `2` means a configuration or initialisation error (for example a missing model), and `1` means a runtime error. Add `-v` for full tracebacks.

## Configuration

`config.yaml` sets the backend, preview, threshold, max detections, an optional labels file, and the model for each backend. CLI flags override it.

If `--config` isn't given, the first file found is used, in this order:

1. `./config.yaml`
2. `~/.config/pi-ai-camera/config.yaml`
3. `/etc/pi-ai-camera/config.yaml`
4. the repo's `config.yaml`

Unknown keys and out-of-range values are rejected with a clear error.

- `hailo.model: auto` picks `yolov8s_h8l.hef`, `yolov8s_h8.hef` or `yolov8m_h10.hef` to match the detected NPU.
- If an IMX500 model has no embedded labels, the bundled COCO label list is used.

## Development

```bash
pip install -e '.[test]'   # off-device only; installs pytest, PyYAML, numpy from pip
pytest
```

The tests don't need hardware. They cover:

- config loading and validation
- CLI overrides and exit codes
- HEF selection
- Hailo NMS parsing
- IMX500 tensor parsing, using fake device objects

## Layout

```
scripts/setup.sh                 installer
scripts/verify.sh                hardware/software checks
src/pi_ai_camera/config.py       YAML config loader + validation
src/pi_ai_camera/camera.py       Picamera2 + IMX500 object-detection backend
src/pi_ai_camera/hailo.py        Picamera2 + Hailo backend (HEF auto-selection)
src/pi_ai_camera/detection.py    Detection type + Detector protocol
src/pi_ai_camera/main.py         CLI (pi-ai-camera)
src/pi_ai_camera/assets/         bundled coco_labels.txt (from picamera2-examples)
tests/                           pytest suite (no hardware needed)
config.yaml                      default configuration
pyproject.toml                   packaging / entry point
```

## References

- [AI Camera docs](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [AI HAT+ docs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [AI software (Hailo) docs](https://www.raspberrypi.com/documentation/computers/ai.html)
- [picamera2-examples: IMX500](https://github.com/raspberrypi/picamera2-examples/tree/main/examples/imx500)
- [picamera2-examples: Hailo](https://github.com/raspberrypi/picamera2-examples/tree/main/examples/hailo)
