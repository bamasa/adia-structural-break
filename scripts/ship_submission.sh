#!/usr/bin/env bash
# Assemble, verify and push one submission in a single command.
#
#   scripts/ship_submission.sh <submission-dir> <interface.py> <meta.txt> <resources-dir> "<message>"
#
# The steps are the ones every shipped submission went through by hand, in
# the order that caught the mistakes: assemble from the library (the channel
# check refuses a width mismatch between interface and artifact), verify the
# assembled monitor against the training matrices and profile milliseconds
# per step (the verifier script lives next to the interface as verify_*.py),
# and only then copy into the crunch workspace and push. A failure at any
# step stops the chain — nothing reaches the platform unverified.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(cd "$ROOT/.." && pwd)"                 # the workspace holding matrices and resources*/
SUB="$1"; INTERFACE="$2"; META="$3"; RESOURCES="$4"; MESSAGE="$5"
PY="$WORK/.venv/bin/python"
CRUNCH_WS="$WORK/structural-break-real-time-test"

mkdir -p "$ROOT/submissions/$SUB"
rm -rf "$ROOT/submissions/$SUB/resources"
ln -s "../../../$(basename "$RESOURCES")" "$ROOT/submissions/$SUB/resources"
cd "$WORK"
"$PY" "$ROOT/scripts/assemble_submission.py" "$ROOT/submissions/$SUB/main.py" "$INTERFACE" "$META"
# The platform requirements: the shared file when it exists, else the 078 copy every submission carried.
if [ -f "$ROOT/submissions/requirements.txt" ]; then REQ="$ROOT/submissions/requirements.txt"; else REQ="$ROOT/submissions/078-two-classifiers/requirements.txt"; fi
cp "$REQ" "$ROOT/submissions/$SUB/requirements.txt"
VERIFY="$(dirname "$INTERFACE")/verify_$(basename "$INTERFACE" | sed 's/interface_//')"
if [ -f "$VERIFY" ]; then "$PY" "$VERIFY"; else echo "no verifier next to the interface — refusing to ship" >&2; exit 2; fi
cp "$ROOT/submissions/$SUB/main.py" "$CRUNCH_WS/main.py"
cp "$ROOT/submissions/$SUB/requirements.txt" "$CRUNCH_WS/requirements.txt"
cp "$RESOURCES/model.joblib" "$CRUNCH_WS/resources/model.joblib"
cd "$CRUNCH_WS" && "$WORK/.venv/bin/crunch" push --no-pip-freeze -m "$MESSAGE"
rm -f "$ROOT/submissions/$SUB/resources"
