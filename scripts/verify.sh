#!/usr/bin/env bash
# =============================================================================
# verify.sh - Post-install health check for the Pi 5 AI camera stack.
#
# Checks, in order:
#   * Board is a Raspberry Pi 5
#   * The IMX500 AI Camera is enumerated by libcamera (rpicam-hello)
#   * IMX500 sensor firmware and .rpk network models are installed
#   * The Hailo NPU: driver loaded, answers over PCIe (hailortcli),
#     /dev/hailo0 exists, PCIe link speed, default HEFs installed
#   * Required Python modules import
#
# Every check runs even if an earlier one fails, so a single run reports
# everything that's wrong. Exit status is 0 only if all checks pass.
#
# Usage: ./scripts/verify.sh [--no-hailo]   (no root required)
#   --no-hailo   skip Hailo checks (IMX500-only setups without an AI HAT)
# =============================================================================

# Deliberately no `-e`: failing checks are counted, not fatal.
set -uo pipefail

CHECK_HAILO=1
for arg in "$@"; do
  case "$arg" in
    --no-hailo) CHECK_HAILO=0 ;;
    -h|--help) sed -n '/^# Usage:/,/^#   --no-hailo/p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

PASS=0
FAIL=0

# Result helpers. Counters are updated so the summary can set the exit code.
ok()   { printf '  \033[1;32mPASS\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[1;31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
note() { printf '  \033[1;33mWARN\033[0m %s\n' "$*"; }
info() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# check MESSAGE CMD [ARGS...]
#   Run CMD; report PASS/FAIL with MESSAGE. An explicit if/else instead of
#   `A && ok || bad` (SC2015), which would misreport if `ok` ever failed.
#   Optional FAIL_MSG env var overrides the failure text.
check() {
  local msg=$1; shift
  if "$@"; then ok "$msg"; else bad "${FAIL_MSG:-$msg}"; fi
}

# -----------------------------------------------------------------------------
# Platform
# -----------------------------------------------------------------------------
info "Platform"
# Device-tree model string is NUL-terminated; strip it before matching.
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
FAIL_MSG="Not a Raspberry Pi 5 ($MODEL)" check "$MODEL" [ "${MODEL#*Raspberry Pi 5}" != "$MODEL" ]

# -----------------------------------------------------------------------------
# Camera enumeration
# -----------------------------------------------------------------------------
info "Camera (rpicam-hello --list-cameras)"
if command -v rpicam-hello >/dev/null; then
  # --list-cameras asks libcamera for detected sensors without opening a
  # preview. If the IMX500 is missing here, the problem is cabling or the
  # device-tree overlay (camera_auto_detect), not the Python code.
  CAMS="$(rpicam-hello --list-cameras 2>&1 || true)"
  echo "$CAMS" | sed 's/^/    /'
  if grep -qi imx500 <<<"$CAMS"; then
    ok "IMX500 AI Camera detected"
  else
    bad "IMX500 not listed (check ribbon cable / camera_auto_detect)"
  fi
  # libcamera can't enumerate a camera another process has open.
  if grep -qi busy <<<"$CAMS"; then
    note "Camera appears busy: stop other camera apps (e.g. a running pi-ai-camera) and retry"
  fi
else
  bad "rpicam-hello not installed"
fi

# -----------------------------------------------------------------------------
# IMX500 firmware and models (from the imx500-all package)
# -----------------------------------------------------------------------------
info "IMX500 firmware & models"
# The loader and main firmware are uploaded to the sensor's MCU at startup;
# without them the camera can stream images but cannot run networks.
# File names per the official AI Camera getting-started docs.
for f in /lib/firmware/imx500_loader.fpk /lib/firmware/imx500_firmware.fpk; do
  FAIL_MSG="missing $f (install imx500-all)" check "$f" test -f "$f"
done

# Packaged .rpk network files, plus the specific default model the app loads.
MODEL_DIR=/usr/share/imx500-models
if [[ -d $MODEL_DIR ]] && compgen -G "$MODEL_DIR/*.rpk" >/dev/null; then
  ok "$(find "$MODEL_DIR" -maxdepth 1 -name '*.rpk' | wc -l) models in $MODEL_DIR"
else
  bad "no .rpk models in $MODEL_DIR"
fi
DEFAULT_RPK="$MODEL_DIR/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
FAIL_MSG="default model missing: $DEFAULT_RPK" check "default model $(basename "$DEFAULT_RPK")" test -f "$DEFAULT_RPK"

# -----------------------------------------------------------------------------
# Hailo NPU (from hailo-all / hailo-h10-all)
# -----------------------------------------------------------------------------
if [[ $CHECK_HAILO -eq 1 ]]; then
  info "Hailo NPU"

  # Kernel driver loaded? (hailo_pci for Hailo-8/8L; name may differ on
  # Hailo-10H / HailoRT 5.x, hence the loose match.)
  FAIL_MSG="hailo kernel module not loaded (reboot after install?)" \
    check "hailo kernel module loaded" grep -qi '^hailo' /proc/modules

  if command -v hailortcli >/dev/null; then
    # identify talks to the device firmware over PCIe; success proves the
    # driver loaded, the firmware booted and the link is up.
    OUT="$(hailortcli fw-control identify 2>&1)"; RC=$?
    echo "$OUT" | sed 's/^/    /'
    if [[ $RC -eq 0 ]]; then ok "Hailo device responded"; else bad "hailortcli failed (rc=$RC)"; fi
  else
    bad "hailortcli not installed (install hailo-all)"
  fi

  # Character device created by the kernel driver.
  FAIL_MSG="/dev/hailo0 missing" check "/dev/hailo0 present" test -e /dev/hailo0

  # PCIe link speed. Select the device by Hailo's PCI vendor ID (1e60) rather
  # than grepping text. LnkSta needs root, so try non-interactive sudo and
  # skip quietly if unavailable. 8GT/s = Gen 3; anything lower halves throughput.
  if command -v lspci >/dev/null; then
    LNK="$( { sudo -n lspci -d 1e60: -vv 2>/dev/null || lspci -d 1e60: -vv 2>/dev/null; } | grep -m1 'LnkSta:' | sed 's/^[[:space:]]*//')"
    if [[ -z "$LNK" ]]; then
      note "PCIe link status unavailable (run as root to see it)"
    elif [[ "$LNK" == *"8GT/s"* ]]; then
      ok "PCIe Gen 3: $LNK"
    else
      note "PCIe not at Gen 3 (8GT/s): $LNK - check dtparam=pciex1_gen=3"
    fi
  fi

  # At least one of the HEFs that `hailo.model: auto` can select.
  HEF_DIR=/usr/share/hailo-models
  if compgen -G "$HEF_DIR/yolov8*_h*.hef" >/dev/null; then
    ok "default HEF(s): $(cd "$HEF_DIR" && ls yolov8*_h*.hef | tr '\n' ' ')"
  else
    bad "no yolov8 HEFs in $HEF_DIR (install hailo-all / hailo-h10-all)"
  fi
else
  info "Hailo NPU (skipped: --no-hailo)"
fi

# -----------------------------------------------------------------------------
# Python modules used by pi_ai_camera
# -----------------------------------------------------------------------------
info "Python modules"
PY_MODS=(picamera2 cv2 yaml numpy)
# hailo_platform is HailoRT's Python API (python3-hailort), used by
# picamera2.devices.Hailo under the hood.
[[ $CHECK_HAILO -eq 1 ]] && PY_MODS+=(hailo_platform)
# Quiet import probe: tracebacks would drown out the PASS/FAIL report.
py_import() { python3 -c "import $1" 2>/dev/null; }
for m in "${PY_MODS[@]}"; do
  FAIL_MSG="python3 cannot import $m" check "python3 import $m" py_import "$m"
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf '\nSummary: %d passed, %d failed\n' "$PASS" "$FAIL"
# Exit status of the script = result of this test (0 only when no failures).
[[ $FAIL -eq 0 ]]
