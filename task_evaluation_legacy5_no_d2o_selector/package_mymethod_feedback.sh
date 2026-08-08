#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RESULTS_DIR" >&2
  exit 2
fi
RESULTS_DIR="$(readlink -f "$1")"
if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Missing results directory: ${RESULTS_DIR}" >&2
  exit 2
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$(dirname "${RESULTS_DIR}")/$(basename "${RESULTS_DIR}")_feedback_${STAMP}.zip"
python - "${RESULTS_DIR}" "${OUT}" <<'PY'
from pathlib import Path
import sys, zipfile
src=Path(sys.argv[1]); out=Path(sys.argv[2])
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(src.parent))
print(out)
PY
