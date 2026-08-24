#!/usr/bin/env bash
# Compatibility wrapper for the immutable, versioned AMI visual builder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_DIR="${ROOT_DIR}/data/ami_full_v2"
JOBS=2
MAX_TURNS=""
MAX_PER_SPK=""
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)      RUN_DIR="$2"      ; shift 2 ;;
        --jobs)         JOBS="$2"         ; shift 2 ;;
        --max-turns)    MAX_TURNS="$2"    ; shift 2 ;;
        --max-per-spk)  MAX_PER_SPK="$2"  ; shift 2 ;;
        --resume)       RESUME=1          ; shift ;;
        *) echo "[error] Unknown flag: $1" >&2 ; exit 2 ;;
    esac
done

ARGS=(
    --root "$ROOT_DIR"
    --run-dir "$RUN_DIR"
    --splits train
    --jobs "$JOBS"
)
if [[ -n "$MAX_TURNS" ]]; then
    ARGS+=(--max-turns "$MAX_TURNS")
fi
if [[ -n "$MAX_PER_SPK" ]]; then
    ARGS+=(--max-turns-per-speaker "$MAX_PER_SPK")
fi
if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

exec python "${ROOT_DIR}/scripts/build_ami_visual_manifests.py" "${ARGS[@]}"
