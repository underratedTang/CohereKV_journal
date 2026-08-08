#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash package_mymethod_engineering_feedback.sh RESULT_DIR" >&2
  exit 2
fi
RESULT_DIR="$1"
RESULT_DIR="$(cd "${RESULT_DIR}" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${RESULT_DIR%/}_feedback_${STAMP}.zip"
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${TMP}/results"
find "${RESULT_DIR}" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.txt' -o -name '*.log' \) -exec cp {} "${TMP}/results/" \;
{
  echo "result_dir=${RESULT_DIR}"
  echo "created=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "files:"; find "${TMP}/results" -type f -printf '%f\n' | sort
} > "${TMP}/MANIFEST.txt"
(cd "${TMP}" && zip -qr "${OUT}" .)
echo "${OUT}"
