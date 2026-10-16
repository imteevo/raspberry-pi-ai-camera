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
#      Python bindings that the pi_ai_camera package depends on.
#   4. Appends one marked block to /boot/firmware/config.txt enabling PCIe
#      Gen 3 and camera auto-detection.
#   5. Offers a reboot when anything that requires one changed (kernel,
#      kernel-module packages, firmware, EEPROM or config.txt).
#
# Usage: sudo ./scripts/setup.sh [--hailo-h10] [--no-upgrade] [--yes]
#   --hailo-h10   install hailo-h10-all (AI HAT+ 2, Hailo-10H) instead of hailo-all
#   --no-upgrade  skip apt full-upgrade / EEPROM update
#   --yes         don't prompt; never reboot automatically
# =============================================================================

# Fail fast: exit on error (-e), on unset variables (-u), and if any command
# in a pipeline fails (pipefail) rather than only the last one. -E makes the
# ERR trap below fire inside functions too.
set -Eeuo pipefail

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

# Without this, a failure under `set -e` exits silently mid-run and the user
# can't tell which step broke.
# shellcheck disable=SC2154  # rc is assigned inside the trap string itself
trap 'rc=$?; printf "\033[1;31m[error]\033[0m %s failed (exit %d) at line %d\n" "${BASH_COMMAND}" "$rc" "$LINENO" >&2' ERR

# apt, config.txt edits and reboot all require root. We deliberately don't
# auto re-exec through sudo: escalation should be an explicit user choice.
[[ $EUID -eq 0 ]] || die "Run as root: sudo $0 $*"

# is_installed PKG
#   True only if dpkg reports the package as fully "installed". `dpkg -s`
#   alone also succeeds for packages in "rc" state (removed, config files
#   left behind), which would wrongly skip installs or flag false conflicts.
is_installed() {
  [[ "$(dpkg-query -W -f='${db:Status-Status}' "$1" 2>/dev/null)" == "installed" ]]
}

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

# Keep apt from opening interactive dialogs. DEBIAN_FRONTEND alone doesn't
# answer conffile prompts, so also keep existing config files by default
# (--force-confdef/--force-confold) - safest for a user-tuned system.
export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

# -----------------------------------------------------------------------------
# System update + firmware
# -----------------------------------------------------------------------------
log "apt update"
apt-get update

if [[ $DO_UPGRADE -eq 1 ]]; then
  # The Hailo DKMS driver is built against the kernel, and the IMX500
  # driver lives in the kernel too, so a current kernel is a prerequisite.
  log "apt full-upgrade"
  apt-get "${APT_OPTS[@]}" full-upgrade

  # A new kernel means DKMS just built hailo_pci for *that* kernel, not the
  # running one, so the NPU won't work until we reboot. Detect it two ways:
  # Debian's reboot-required flag, and the newest installed module tree
  # differing from the running kernel (Pi kernels don't always set the flag).
  [[ -e /var/run/reboot-required ]] && REBOOT_NEEDED=1
  newest_kernel="$(ls -1 /lib/modules 2>/dev/null | sort -V | tail -n1)"
  if [[ -n "$newest_kernel" && "$newest_kernel" != "$(uname -r)" ]]; then
    log "Kernel $newest_kernel installed (running $(uname -r))"
    REBOOT_NEEDED=1
  fi

  # Bootloader EEPROM updates carry PCIe and camera fixes for the Pi 5.
  # rpi-eeprom-update exits 1 (EXIT_UPDATE_REQUIRED) when an update is
  # available, so capture the status explicitly instead of piping into grep,
  # which under pipefail would always take the "up to date" branch.
  if command -v rpi-eeprom-update >/dev/null; then
    log "Checking bootloader EEPROM"
    eeprom_status=0
    eeprom_out="$(rpi-eeprom-update 2>&1)" || eeprom_status=$?
    case $eeprom_status in
      0) log "EEPROM up to date" ;;
      1)
        log "EEPROM update available; staging it (applied on next boot)"
        rpi-eeprom-update -a || die "rpi-eeprom-update -a failed"
        REBOOT_NEEDED=1
        ;;
      *) warn "rpi-eeprom-update returned $eeprom_status:"$'\n'"$eeprom_out" ;;
    esac
  fi
fi

# -----------------------------------------------------------------------------
# Packages
# -----------------------------------------------------------------------------
# hailo-all (Hailo-8/8L) and hailo-h10-all (Hailo-10H) ship conflicting
# drivers and runtimes and cannot co-exist. Refuse to proceed rather than let
# apt silently rip out the other stack.
OTHER_HAILO="hailo-h10-all"; [[ $HAILO_PKG == hailo-h10-all ]] && OTHER_HAILO="hailo-all"
if is_installed "$OTHER_HAILO"; then
  die "$OTHER_HAILO is installed; it conflicts with $HAILO_PKG. Remove it first or change flags."
fi

# Package roles:
#   dkms               builds the out-of-tree Hailo PCIe kernel driver
#   hailo-all / -h10   Hailo driver, firmware, HailoRT, TAPPAS, example models
#   imx500-all         IMX500 sensor firmware + prebuilt .rpk models
#   rpicam-apps        camera CLI tools (rpicam-hello used by verify.sh)
#   python3-picamera2  Picamera2, including picamera2.devices.{IMX500,Hailo}
#   python3-opencv     image utilities for overlays/post-processing
#   python3-yaml/numpy config loading and tensor handling in pi_ai_camera
#   python3-venv       optional venv with --system-site-packages
PKGS=(dkms "$HAILO_PKG" imx500-all rpicam-apps python3-picamera2 python3-opencv python3-yaml python3-numpy python3-venv)

# Only these load kernel modules or firmware at boot; installing anything
# else (e.g. python3-venv) doesn't need a reboot.
REBOOT_PKGS=" dkms hailo-all hailo-h10-all imx500-all "

# Only hand apt the packages that are actually missing, so re-runs are fast
# and we can tell whether anything changed.
MISSING=()
for p in "${PKGS[@]}"; do
  is_installed "$p" || MISSING+=("$p")
done

if ((${#MISSING[@]})); then
  log "Installing: ${MISSING[*]}"
  apt-get "${APT_OPTS[@]}" install "${MISSING[@]}"
  for p in "${MISSING[@]}"; do
    [[ $REBOOT_PKGS == *" $p "* ]] && REBOOT_NEEDED=1
  done
else
  log "All packages already installed"
fi

# -----------------------------------------------------------------------------
# config.txt
# -----------------------------------------------------------------------------
# Rather than editing existing lines (which may live inside conditional
# sections like [cm5] or [pi4] that the user set up deliberately), append a
# single marked block once, under [all]. In config.txt later settings win,
# so placing [all] at the end makes these values effective without touching
# anything else. The marker makes the step idempotent.
#
# Settings:
#   dtparam=pciex1_gen=3  The Pi 5 PCIe x1 link defaults to Gen 2 (5 GT/s);
#                         Gen 3 (8 GT/s) roughly doubles NPU bandwidth. The AI
#                         HAT+ requests this automatically via its HAT+ EEPROM;
#                         setting it explicitly is harmless and required for
#                         the AI Kit.
#   camera_auto_detect=1  Default on Raspberry Pi OS; it's what loads the
#                         imx500 device-tree overlay. Guard against it having
#                         been disabled.
MARK_BEGIN="# >>> raspberry-pi-ai-camera >>>"
MARK_END="# <<< raspberry-pi-ai-camera <<<"
BACKUP="${CONFIG_TXT}.bak-ai-camera"

if grep -qF "$MARK_BEGIN" "$CONFIG_TXT"; then
  log "config.txt already contains the raspberry-pi-ai-camera block"
else
  # Back up once, preserving the very first original. Test explicitly rather
  # than `cp -n ... || true`, which would also hide real failures (full /boot).
  if [[ ! -e "$BACKUP" ]]; then
    cp -a "$CONFIG_TXT" "$BACKUP"
    log "Backed up config.txt to $BACKUP"
  fi
  if grep -Eq '^[[:space:]]*camera_auto_detect=0' "$CONFIG_TXT"; then
    warn "camera_auto_detect=0 found earlier in config.txt; it will be overridden"
  fi
  printf '\n%s\n[all]\ndtparam=pciex1_gen=3\ncamera_auto_detect=1\n%s\n' \
    "$MARK_BEGIN" "$MARK_END" >> "$CONFIG_TXT"
  log "Appended PCIe Gen 3 + camera auto-detect block to $CONFIG_TXT"
  REBOOT_NEEDED=1
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
log "Setup complete. After reboot run: ./scripts/verify.sh"
if [[ $REBOOT_NEEDED -eq 1 ]]; then
  # Never prompt without a terminal (ssh without -t, cloud-init, Ansible):
  # `read` would hit EOF, return 1, and abort a successful run under set -e.
  if [[ $ASSUME_YES -eq 1 || ! -t 0 ]]; then
    warn "Reboot required. Run: sudo reboot"
  else
    read -r -p "A reboot is required. Reboot now? [y/N] " ans || ans=""
    if [[ $ans =~ ^[Yy]$ ]]; then
      reboot
    else
      warn "Remember to reboot before using the hardware."
    fi
  fi
else
  log "No reboot required."
fi
