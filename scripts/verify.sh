#!/usr/bin/env bash
# Verify AI Camera (IMX500) and AI HAT+ (Hailo) are detected and ready.
set -uo pipefail

PASS=0; FAIL=0
ok()   { printf '  \033[1;32mPASS\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[1;31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
info() { printf '\n\033[1m%s\033[0m\n' "$*"; }

info "Platform"
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
[[ "$MODEL" == *"Raspberry Pi 5"* ]] && ok "$MODEL" || bad "Not a Raspberry Pi 5 ($MODEL)"

info "Camera (rpicam-hello --list-cameras)"
if command -v rpicam-hello >/dev/null; then
  CAMS="$(rpicam-hello --list-cameras 2>&1 || true)"
  echo "$CAMS" | sed 's/^/    /'
  if echo "$CAMS" | grep -qi imx500; then ok "IMX500 AI Camera detected"
  else bad "IMX500 not listed (check ribbon cable / camera_auto_detect)"; fi
else
  bad "rpicam-hello not installed"
fi

info "IMX500 firmware & models"
FW_FILES=(/lib/firmware/imx500_loader.fpk /lib/firmware/imx500_main.fpk)
for f in "${FW_FILES[@]}"; do
  [[ -f "$f" ]] && ok "$f" || bad "missing $f (install imx500-all)"
done
MODEL_DIR=/usr/share/imx500-models
if [[ -d $MODEL_DIR ]] && compgen -G "$MODEL_DIR/*.rpk" >/dev/null; then
  ok "$(ls "$MODEL_DIR"/*.rpk | wc -l) models in $MODEL_DIR"
else
  bad "no .rpk models in $MODEL_DIR"
fi

info "Hailo NPU (hailortcli fw-control identify)"
if command -v hailortcli >/dev/null; then
  OUT="$(hailortcli fw-control identify 2>&1)"; RC=$?
  echo "$OUT" | sed 's/^/    /'
  [[ $RC -eq 0 ]] && ok "Hailo device responded" || bad "hailortcli failed (rc=$RC)"
else
  bad "hailortcli not installed (install hailo-all)"
fi
[[ -e /dev/hailo0 ]] && ok "/dev/hailo0 present" || bad "/dev/hailo0 missing"
if command -v lspci >/dev/null; then
  SPEED="$(lspci -vv 2>/dev/null | grep -A40 -i hailo | grep -m1 'LnkSta:' | sed 's/^\s*//')"
  [[ -n "$SPEED" ]] && echo "    PCIe $SPEED"
fi

info "Python modules"
for m in picamera2 cv2 yaml numpy; do
  python3 -c "import $m" 2>/dev/null && ok "python3 import $m" || bad "python3 cannot import $m"
done
python3 -c "import hailo_platform" 2>/dev/null && ok "python3 import hailo_platform" || bad "python3 cannot import hailo_platform (python3-hailort)"

printf '\nSummary: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
