#!/usr/bin/env bash
# =============================================================================
# flatten_ami_corpus.sh  —  Flatten official AMI wget script output into
#                           the project's flat layout
#
# The official AMI wget script downloads into a nested structure:
#   amicorpus/{MID}/audio/{MID}.Headset-N.wav
#   amicorpus/{MID}/video/{MID}.Closeup{C}.avi
#   amicorpus/{MID}/video/{MID}.PreferredOverview.avi
#   amicorpus/{MID}/video/{MID}.Corner.avi          ← not needed, skipped
#   amicorpus/{MID}/video/{MID}.Overhead.avi        ← not needed, skipped
#
# This script moves (or copies) only the files the pipeline uses into:
#   datasets/ami/audio/{MID}.Headset-N.wav
#   datasets/ami/video/{MID}.Closeup{C}.avi
#   datasets/ami/video/{MID}.PreferredOverview.avi
#
# Usage:
#   bash scripts/flatten_ami_corpus.sh [OPTIONS]
#
# Options:
#   --src-dir DIR    Root of official download (default: amicorpus)
#   --out-dir DIR    Project dataset root (default: datasets/ami)
#   --copy           Copy instead of move (safe if you want to keep originals)
#   --overwrite      Overwrite existing files in dst (default: skip)
#   --dry-run        Print actions without moving anything
#   -h, --help       Show this help
#
# Files moved:
#   *.Headset-{0,1,2,3}.wav       (audio)
#   *.Closeup{1,2,3,4}.avi        (video — speaker close-ups)
#   *.PreferredOverview.avi        (video — full room, for speaker counting)
#
# Files intentionally skipped:
#   *.Corner.avi, *.Overhead.avi   (not used by pipeline)
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${ROOT_DIR}/amicorpus"
OUT_DIR="${ROOT_DIR}/datasets/ami"
COPY_MODE=0
OVERWRITE=0
DRY_RUN=0

# --------------------------------------------------------------------------- #
# Arg parse
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-dir)   SRC_DIR="$2"  ; shift 2 ;;
        --out-dir)   OUT_DIR="$2"  ; shift 2 ;;
        --copy)      COPY_MODE=1   ; shift ;;
        --overwrite) OVERWRITE=1   ; shift ;;
        --dry-run)   DRY_RUN=1     ; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//' | sed -n '2,40p'
            exit 0 ;;
        *) echo "[error] Unknown argument: $1" >&2 ; exit 1 ;;
    esac
done

AUDIO_DST="${OUT_DIR}/audio"
VIDEO_DST="${OUT_DIR}/video"

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$AUDIO_DST" "$VIDEO_DST"
fi

ACTION="move"
[[ $COPY_MODE -eq 1 ]] && ACTION="copy"

echo "============================================================"
echo " AMI flatten  —  ${ACTION}"
echo "============================================================"
printf " src : %s\n" "$SRC_DIR"
printf " dst : %s\n" "$OUT_DIR"
printf " overwrite : %s\n" "$([[ $OVERWRITE -eq 1 ]] && echo yes || echo no)"
printf " dry-run   : %s\n" "$([[ $DRY_RUN  -eq 1 ]] && echo yes || echo no)"
echo "------------------------------------------------------------"

N_MOVED=0
N_SKIP=0
N_MISS=0

_transfer() {
    local src="$1" dst="$2"
    local fname
    fname="$(basename "$src")"

    if [[ ! -f "$src" ]]; then
        echo "[miss] ${fname}"
        (( N_MISS++ )) || true
        return
    fi

    if [[ -f "$dst" && $OVERWRITE -eq 0 ]]; then
        echo "[skip] ${fname}"
        (( N_SKIP++ )) || true
        return
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry]  ${src} -> ${dst}"
        (( N_MOVED++ )) || true
        return
    fi

    if [[ $COPY_MODE -eq 1 ]]; then
        cp "$src" "$dst"
    else
        mv "$src" "$dst"
    fi
    echo "[ok]   ${fname}"
    (( N_MOVED++ )) || true
}

# --------------------------------------------------------------------------- #
# Walk source tree
# --------------------------------------------------------------------------- #
if [[ ! -d "$SRC_DIR" ]]; then
    echo "[error] src-dir not found: ${SRC_DIR}" >&2
    exit 1
fi

for MID_DIR in "${SRC_DIR}"/*/; do
    MID="$(basename "$MID_DIR")"

    # Audio: Headset-0..3
    for N in 0 1 2 3; do
        src="${MID_DIR}audio/${MID}.Headset-${N}.wav"
        dst="${AUDIO_DST}/${MID}.Headset-${N}.wav"
        _transfer "$src" "$dst"
    done

    # Video: Closeup1..4
    for C in 1 2 3 4; do
        src="${MID_DIR}video/${MID}.Closeup${C}.avi"
        dst="${VIDEO_DST}/${MID}.Closeup${C}.avi"
        _transfer "$src" "$dst"
    done

    # Video: PreferredOverview
    src="${MID_DIR}video/${MID}.PreferredOverview.avi"
    dst="${VIDEO_DST}/${MID}.PreferredOverview.avi"
    _transfer "$src" "$dst"
done

echo ""
echo "============================================================"
printf " ok   : %d\n" "$N_MOVED"
printf " skip : %d\n" "$N_SKIP"
printf " miss : %d\n" "$N_MISS"
echo "============================================================"

if [[ $N_MISS -gt 0 ]]; then
    echo ""
    echo " Some files were not found in ${SRC_DIR}."
    echo " They may not have been downloaded yet, or the meeting IDs"
    echo " don't exist in AMI (IS1002a, IS1005d are known missing)."
fi

if [[ $COPY_MODE -eq 0 && $DRY_RUN -eq 0 && $N_MOVED -gt 0 ]]; then
    echo ""
    echo " Files moved. You can now remove empty amicorpus/ dirs with:"
    echo "   find ${SRC_DIR} -type d -empty -delete"
fi
