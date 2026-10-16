#!/usr/bin/env bash
# =============================================================================
# setup.sh - Idempotent installer for the Raspberry Pi 5 AI camera stack.
#
# Target hardware:
#   * Raspberry Pi 5 (BCM2712) running 64-bit Raspberry Pi OS (Trixie/Bookworm)
#   * Raspberry Pi AI Camera (Sony IMX500, on-sensor NPU, CSI-2 connection)
#   * Raspberry Pi AI HAT+ (Hailo-8L / Hailo-8, PCIe) or AI HAT+ 2 (Hailo-10H)
#
# What it does (each step is skipped when already satisfied, so the script is
# safe to re-run):
#   1. Verifies we are root, on a Pi 5, on an aarch64 userland.
#   2. Refreshes apt metadata and (optionally) upgrades packages + EEPROM.
#   3. Installs the IMX500 firmware/models, Hailo runtime, camera apps and
#      Python bindings that the src/ package depends on.
#   4. Ensures /boot/firmware/config.txt enables PCIe Gen 3 and camera
#      auto-detection.
#   5. Offers a reboot when anything that requires one changed.
#
# Usage: sudo ./scripts/setup.sh [--hailo-h10] [--no-upgrade] [--yes]
#   --hailo-h10   install hailo-h10-all (AI HAT+ 2, Hailo-10H) instead of hailo-all
#   --no-upgrade  skip apt full-upgrade / EEPROM update
#   --yes         don't prompt; never reboot automatically
# =============================================================================

# Fail fast: exit on error (-e), on unset variables (-u), and if any command
# in a pipeline fails (pipefail) rather than only the last one.
set -euo pipefail

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
# Default to hailo-all, the meta-package for the AI HAT+ and AI Kit
# (Hailo-8/8L). The AI HAT+ 2 uses a different driver/runtime stack.
HAILO_PKG="hailo-all"
DO_UPGRADE=1   # 1 = run full-upgrade + EEPROM update (recommended on first run)
ASSUME_YES=0   # 1 = non-interactive; never prompt and never reboot

for arg in "$@"; do
  case "$arg" in
    --hailo-h10) HAILO_PKG="hailo-h10-all" ;;
    --no-upgrade) DO_UPGRADE=0 ;;
    --yes|-y) ASSUME_YES=1 ;;
    # Print the usage block from this file's header comment.
    -h|--help) sed -n '/^# Usage:/,/^#   --yes/p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

# -----------------------------------------------------------------------------
# Logging helpers (coloured prefixes; warnings/errors go to stderr)
# -----------------------------------------------------------------------------
log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# apt, config.txt edits and reboot all require root.
[[ $EUID -eq 0 ]] || die "Run as root: sudo $0 $*"

# -----------------------------------------------------------------------------
# Platform checks
# -----------------------------------------------------------------------------
# The device-tree model string is the most reliable board identifier on a Pi.
# It is NUL-terminated, so strip the trailing NUL before comparing.
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
[[ "$MODEL" == *"Raspberry Pi 5"* ]] || die "This script targets Raspberry Pi 5 (found: '${MODEL:-unknown}')."

# Hailo and IMX500 packages are only published for arm64.
[[ "$(uname -m)" == "aarch64" ]] || die "64-bit Raspberry Pi OS (aarch64) is required."

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091  # os-release is provided by the OS at runtime
  . /etc/os-release
  log "OS: ${PRETTY_NAME:-unknown}"
  # Official docs target Trixie; Bookworm shipped the original AI packages.
  # Anything else may work but is untested, so warn rather than fail.
  [[ "${VERSION_CODENAME:-}" =~ ^(bookworm|trixie)$ ]] || warn "Untested OS release '${VERSION_CODENAME:-?}'; official docs target Trixie."
fi

# Bookworm+ mounts the boot partition at /boot/firmware; older images used
# /boot directly. Support both so the script works on upgraded installs.
CONFIG_TXT=/boot/firmware/config.txt
[[ -f "$CONFIG_TXT" ]] || CONFIG_TXT=/boot/config.txt
[[ -f "$CONFIG_TXT" ]] || die "Could not find config.txt"

# Set to 1 by any step whose effect only takes hold after a reboot.
REBOOT_NEEDED=0

# Keep apt from opening interactive dialogs (e.g. config file prompts).
export DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# System update + firmware
# -----------------------------------------------------------------------------
log "apt update"
apt-get update

if [[ $DO_UPGRADE -eq 1 ]]; then
  # The Hailo DKMS driver is built against the running kernel, and the IMX500
  # driver lives in the kernel too, so a current kernel is a prerequisite.
  log "apt full-upgrade"
  apt-get -y full-upgrade

  # Bootloader EEPROM updates carry PCIe and camera fixes for the Pi 5.
  if command -v rpi-eeprom-update >/dev/null; then
    log "Checking bootloader EEPROM"
    if rpi-eeprom-update | grep -q "UPDATE AVAILABLE"; then
      # -a stages the update; it is applied on the next boot.
      rpi-eeprom-update -a && REBOOT_NEEDED=1
    else
      log "EEPROM up to date"
    fi
  fi
fi

# -----------------------------------------------------------------------------
# Packages
# -----------------------------------------------------------------------------
# hailo-all (Hailo-8/8L) and hailo-h10-all (Hailo-10H) ship conflicting
# drivers and runtimes and cannot co-exist. Refuse to proceed rather than let
# apt silently rip out the other stack.
OTHER_HAILO="hailo-h10-all"; [[ $HAILO_PKG == hailo-h10-all ]] && OTHER_HAILO="hailo-all"
if dpkg -s "$OTHER_HAILO" >/dev/null 2>&1; then
  die "$OTHER_HAILO is installed; it conflicts with $HAILO_PKG. Remove it first or change flags."
fi

# Package roles:
#   dkms               builds the out-of-tree Hailo PCIe kernel driver
#   hailo-all / -h10   Hailo driver, firmware, HailoRT, TAPPAS, example models
#   imx500-all         IMX500 sensor firmware + prebuilt .rpk models
#   rpicam-apps        camera CLI tools (rpicam-hello used by verify.sh)
#   python3-picamera2  Picamera2, including picamera2.devices.{IMX500,Hailo}
#   python3-opencv     image utilities for overlays/post-processing
#   python3-yaml/numpy config loading and tensor handling in src/
#   python3-venv       optional venv with --system-site-packages
PKGS=(dkms "$HAILO_PKG" imx500-all rpicam-apps python3-picamera2 python3-opencv python3-yaml python3-numpy python3-venv)

# Only hand apt the packages that are actually missing, so re-runs are fast
# and we can tell whether anything changed (and thus whether to reboot).
MISSING=()
for p in "${PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done

if ((${#MISSING[@]})); then
  log "Installing: ${MISSING[*]}"
  apt-get install -y "${MISSING[@]}"
  # The new Hailo kernel module and the IMX500 firmware are only picked up
  # cleanly after a reboot.
  REBOOT_NEEDED=1
else
  log "All packages already installed"
fi

# -----------------------------------------------------------------------------
# config.txt
# -----------------------------------------------------------------------------
# ensure_line KEY=VALUE
#   Idempotently make config.txt contain exactly KEY=VALUE.
#   * If the exact line already exists, do nothing.
#   * Otherwise back up config.txt once, comment out any other value for the
#     same KEY (so there is a single source of truth), and append the line
#     under an [all] section so it isn't trapped inside a conditional block
#     such as [cm4] or [pi4] earlier in the file.
ensure_line() {
  local line="$1"
  # Escape dots so they match literally inside the regular expression.
  if grep -Eq "^[[:space:]]*${line//./\\.}[[:space:]]*$" "$CONFIG_TXT"; then
    log "config.txt already has: $line"
  else
    # -n: never overwrite an earlier backup, so the original is preserved.
    cp -n "$CONFIG_TXT" "${CONFIG_TXT}.bak-ai-camera" || true
    local k="${line%=*}"   # key portion, e.g. "dtparam=pciex1_gen"
    sed -i -E "s|^([[:space:]]*${k//./\\.}=.*)$|# \1  # disabled by setup.sh|" "$CONFIG_TXT"
    printf '\n[all]\n%s\n' "$line" >> "$CONFIG_TXT"
    log "Added to config.txt: $line"
    REBOOT_NEEDED=1
  fi
}

# The Pi 5 PCIe x1 link defaults to Gen 2 (5 GT/s). Gen 3 (8 GT/s) roughly
# doubles NPU bandwidth. The AI HAT+ requests this automatically via its HAT+
# EEPROM; setting it explicitly is harmless and is required for the AI Kit.
ensure_line "dtparam=pciex1_gen=3"

# Camera auto-detection is the default on Raspberry Pi OS and is what loads
# the imx500 device-tree overlay. Make sure nobody has disabled it.
if grep -Eq '^[[:space:]]*camera_auto_detect=0' "$CONFIG_TXT"; then
  warn "camera_auto_detect=0 found; re-enabling"
fi
ensure_line "camera_auto_detect=1"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
log "Setup complete. After reboot run: ./scripts/verify.sh"
if [[ $REBOOT_NEEDED -eq 1 ]]; then
  if [[ $ASSUME_YES -eq 1 ]]; then
    # Non-interactive runs (CI, provisioning) must never reboot unexpectedly.
    warn "Reboot required. Run: sudo reboot"
  else
    read -r -p "A reboot is required. Reboot now? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] && reboot || warn "Remember to reboot before using the hardware."
  fi
fi
