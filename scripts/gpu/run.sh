#!/usr/bin/env bash
# One command: check the box, then train. Usage:  ./scripts/gpu/run.sh /data [members]
set -euo pipefail
DATA="${1:?usage: run.sh /path/to/matrices [members]}"
MEMBERS="${2:-24}"
OUT="nets_cuda"

python scripts/gpu/check_environment.py --data-dir "$DATA"
echo
echo "Training $MEMBERS members into $OUT/ — safe to interrupt and rerun,"
echo "finished members are skipped."
python scripts/gpu/train_nets_cuda.py \
    --data-dir "$DATA" --out-dir "$OUT" --members "$MEMBERS" --epochs 14
echo
echo "Done. Copy $OUT/ back to the laptop workspace root."
