DIR="$HOME/mnt/remote_data/motor_scan/dac_scan/nd45/dac300/stepsize4um"
VAL='300'

for f in "$DIR"/*.csv; do
  if head -n1 "$f" | grep -q '"DAC"'; then
    echo "skip (key exists): $f"
    continue
  fi
  cp -p "$f" "$f.bak"
  sed -i "1s/}[[:space:]]*\$/, \"DAC\": $VAL}/" "$f"
  if head -n1 "$f" | grep -q '"DAC"'; then
    echo "ok: $f"
  else
    echo "FAILED (first line not as expected): $f"
  fi
done