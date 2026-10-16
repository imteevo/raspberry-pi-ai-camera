#!/usr/bin/env bash
# Idempotent installer for Raspberry Pi 5 + AI Camera (IMX500) + AI HAT+ (Hailo).
# Usage: sudo ./scripts/setup.sh [--hailo-h10] [--no-upgrade] [--yes]
#   --hailo-h10   install hailo-h10-all (AI HAT+ 2, Hailo-10H) instead of hailo-all
#   --no-upgrade  skip apt full-upgrade / EEPROM update
#   --yes         don't prompt; never reboot automatically
set -euo pipefail

HAILO_PKG="hailo-all"
DO_UPGRADE=1
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --hailo-h10) HAILO_PKG="hailo-h10-all" ;;
    --no-upgrade) DO_UPGRADE=0 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo $0 $*"

# --- Platform checks --------------------------------------------------------
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
[[ "$MODEL" == *"Raspberry Pi 5"* ]] || die "This script targets Raspberry Pi 5 (found: '${MODEL:-unknown}')."
[[ "$(uname -m)" == "aarch64" ]] || die "64-bit Raspberry Pi OS (aarch64) is required."
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  log "OS: ${PRETTY_NAME:-unknown}"
  [[ "${VERSION_CODENAME:-}" =~ ^(bookworm|trixie)$ ]] || warn "Untested OS release '${VERSION_CODENAME:-?}'; official docs target Trixie."
fi

CONFIG_TXT=/boot/firmware/config.txt
[[ -f "$CONFIG_TXT" ]] || CONFIG_TXT=/boot/config.txt
[[ -f "$CONFIG_TXT" ]] || die "Could not find config.txt"

REBOOT_NEEDED=0
export DEBIAN_FRONTEND=noninteractive

# --- System update + firmware ----------------------------------------------
log "apt update"
apt-get update
if [[ $DO_UPGRADE -eq 1 ]]; then
  log "apt full-upgrade"
  apt-get -y full-upgrade
  if command -v rpi-eeprom-update >/dev/null; then
    log "Checking bootloader EEPROM"
    if rpi-eeprom-update | grep -q "UPDATE AVAILABLE"; then
      rpi-eeprom-update -a && REBOOT_NEEDED=1
    else
      log "EEPROM up to date"
    fi
  fi
fi

# --- Packages ----------------------------------------------------------------
# hailo-all and hailo-h10-all cannot co-exist.
OTHER_HAILO="hailo-h10-all"; [[ $HAILO_PKG == hailo-h10-all ]] && OTHER_HAILO="hailo-all"
if dpkg -s "$OTHER_HAILO" >/dev/null 2>&1; then
  die "$OTHER_HAILO is installed; it conflicts with $HAILO_PKG. Remove it first or change flags."
fi

PKGS=(dkms "$HAILO_PKG" imx500-all rpicam-apps python3-picamera2 python3-opencv python3-yaml python3-numpy python3-venv)
MISSING=()
for p in "${PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done
if ((${#MISSING[@]})); then
  log "Installing: ${MISSING[*]}"
  apt-get install -y "${MISSING[@]}"
  # New kernel driver (hailo) / IMX500 firmware need a reboot to load.
  REBOOT_NEEDED=1
else
  log "All packages already installed"
fi

# --- config.txt --------------------------------------------------------------
ensure_line() {
  local line="$1"
  if grep -Eq "^[[:space:]]*${line//./\\.}[[:space:]]*$" "$CONFIG_TXT"; then
    log "config.txt already has: $line"
  else
    cp -n "$CONFIG_TXT" "${CONFIG_TXT}.bak-ai-camera" || true
    # comment out conflicting values of the same key, then append under [all]
    local k="${line%=*}"
    sed -i -E "s|^([[:space:]]*${k//./\\.}=.*)$|# \1  # disabled by setup.sh|" "$CONFIG_TXT"
    printf '\n[all]\n%s\n' "$line" >> "$CONFIG_TXT"
    log "Added to config.txt: $line"
    REBOOT_NEEDED=1
  fi
}
# AI HAT+ applies Gen 3 automatically; explicit setting is harmless and needed for the AI Kit.
ensure_line "dtparam=pciex1_gen=3"
# Camera auto-detection (default on Raspberry Pi OS; make sure it isn't disabled).
if grep -Eq '^[[:space:]]*camera_auto_detect=0' "$CONFIG_TXT"; then
  warn "camera_auto_detect=0 found; re-enabling"
fi
ensure_line "camera_auto_detect=1"

# --- Done --------------------------------------------------------------------
log "Setup complete. After reboot run: ./scripts/verify.sh"
if [[ $REBOOT_NEEDED -eq 1 ]]; then
  if [[ $ASSUME_YES -eq 1 ]]; then
    warn "Reboot required. Run: sudo reboot"
  else
    read -r -p "A reboot is required. Reboot now? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] && reboot || warn "Remember to reboot before using the hardware."
  fi
fi
