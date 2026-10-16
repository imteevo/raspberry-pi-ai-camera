#!/usr/bin/env bash
# =============================================================================
# verify.sh - Post-install health check for the Pi 5 AI camera stack.
#
# Checks, in order:
#   * Board is a Raspberry Pi 5
#   * The IMX500 AI Camera is enumerated by libcamera (rpicam-hello)
#   * IMX500 sensor firmware and .rpk network models are installed
#   * The Hailo NPU answers over PCIe (hailortcli) and /dev/hailo0 exists
#   * Required Python modules import
#
# Every check runs even if an earlier one fails, so a single run reports
# everything that's wrong. Exit status is 0 only if all checks pass.
# Usage: ./scripts/verify.sh   (no root required)
# =============================================================================

# Deliberately no `-e`: failing checks are counted, not fatal.
set -uo pipefail

PASS=0
FAIL=0

# Result helpers. Counters are updated so the summary can set the exit code.
ok()   { printf '  \033[1;32mPASS\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[1;31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
info() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
# Platform
# -----------------------------------------------------------------------------
info "Platform"
# Device-tree model string is NUL-terminated; strip it before matching.
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
[[ "$MODEL" == *"Raspberry Pi 5"* ]] && ok "$MODEL" || bad "Not a Raspberry Pi 5 ($MODEL)"

# -----------------------------------------------------------------------------
# Camera enumeration
# -----------------------------------------------------------------------------
info "Camera (rpicam-hello --list-cameras)"
if command -v rpicam-hello >/dev/null; then
  # --list-cameras asks libcamera for detected sensors without opening a
  # preview. If the IMX500 is missing here, the problem is cabling or the
  # device-tree overlay (camera_auto_detect), not our Python code.
  CAMS="$(rpicam-hello --list-cameras 2>&1 || true)"
  echo "$CAMS" | sed 's/^/    /'
  if echo "$CAMS" | grep -qi imx500; then ok "IMX500 AI Camera detected"
  else bad "IMX500 not listed (check ribbon cable / camera_auto_detect)"; fi
else
  bad "rpicam-hello not installed"
fi

# -----------------------------------------------------------------------------
# IMX500 firmware and models (from the imx500-all package)
# -----------------------------------------------------------------------------
info "IMX500 firmware & models"
# The loader and main firmware are uploaded to the sensor's MCU at startup;
# without them the camera can stream images but cannot run networks.
FW_FILES=(/lib/firmware/imx500_loader.fpk /lib/firmware/imx500_main.fpk)
for f in "${FW_FILES[@]}"; do
  [[ -f "$f" ]] && ok "$f" || bad "missing $f (install imx500-all)"
done

# Packaged .rpk network files; src/camera.py loads one of these.
MODEL_DIR=/usr/share/imx500-models
if [[ -d $MODEL_DIR ]] && compgen -G "$MODEL_DIR/*.rpk" >/dev/null; then
  ok "$(ls "$MODEL_DIR"/*.rpk | wc -l) models in $MODEL_DIR"
else
  bad "no .rpk models in $MODEL_DIR"
fi

# -----------------------------------------------------------------------------
# Hailo NPU (from hailo-all / hailo-h10-all)
# -----------------------------------------------------------------------------
info "Hailo NPU (hailortcli fw-control identify)"
if command -v hailortcli >/dev/null; then
  # identify talks to the device firmware over PCIe; success proves the
  # driver loaded, the firmware booted and the link is up.
  OUT="$(hailortcli fw-control identify 2>&1)"; RC=$?
  echo "$OUT" | sed 's/^/    /'
  [[ $RC -eq 0 ]] && ok "Hailo device responded" || bad "hailortcli failed (rc=$RC)"
else
  bad "hailortcli not installed (install hailo-all)"
fi

# Character device created by the hailo_pci kernel module.
[[ -e /dev/hailo0 ]] && ok "/dev/hailo0 present" || bad "/dev/hailo0 missing"

# Informational only: show the negotiated PCIe link (8GT/s means Gen 3).
# Full `-vv` detail may need root; without it LnkSta can be hidden.
if command -v lspci >/dev/null; then
  SPEED="$(lspci -vv 2>/dev/null | grep -A40 -i hailo | grep -m1 'LnkSta:' | sed 's/^\s*//')"
  [[ -n "$SPEED" ]] && echo "    PCIe $SPEED"
fi

# -----------------------------------------------------------------------------
# Python modules used by src/
# -----------------------------------------------------------------------------
info "Python modules"
for m in picamera2 cv2 yaml numpy; do
  python3 -c "import $m" 2>/dev/null && ok "python3 import $m" || bad "python3 cannot import $m"
done
# hailo_platform is HailoRT's Python API (python3-hailort), used by
# picamera2.devices.Hailo under the hood.
python3 -c "import hailo_platform" 2>/dev/null && ok "python3 import hailo_platform" || bad "python3 cannot import hailo_platform (python3-hailort)"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf '\nSummary: %d passed, %d failed\n' "$PASS" "$FAIL"
# Exit status of the script = result of this test (0 only when no failures).
[[ $FAIL -eq 0 ]]
