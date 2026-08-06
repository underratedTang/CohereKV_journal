#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="${1:-${TASK_DIR}/npc2026_pre_results}"
if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "[ERROR] result directory not found: ${RESULT_DIR}" >&2
  exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${RESULT_DIR%/}_feedback_${STAMP}.zip"
python - "${RESULT_DIR}" "${OUT}" <<'PY'
import json, os, sys, zipfile
from pathlib import Path
root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]).resolve()
files=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and p.suffix.lower() in {'.json','.jsonl','.csv','.txt','.md','.log','.pt'}:
        files.append(p)
manifest={
    'result_dir': str(root),
    'files': [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size} for p in files],
    'requested_feedback_priority': [
        'grouping_diagnostics.json',
        'compression_microbenchmark.json',
        'mymethod_profile_table.txt',
        'ppl_ablation_*.jsonl and *_diagnostics.jsonl',
        'all main PPL/LongBench/system JSON outputs',
    ],
}
manifest_path=root/'feedback_manifest.json'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
files.append(manifest_path)
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, arcname=p.relative_to(root))
print(out)
PY
