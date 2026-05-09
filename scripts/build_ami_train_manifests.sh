#!/usr/bin/env bash
# =============================================================================
# build_ami_train_manifests.sh  —  Build visual manifests for AMI training set
#
# For each training meeting that has both audio and closeup video, calls
# prepare_ami_visual_manifest.py to extract mouth-ROI clips and produce a
# manifest JSON with speaker_mask_v + enrollment_face populated.
#
# Speaker → Closeup mapping used across all AMI scenarios:
#   A = Closeup1,  B = Closeup2,  C = Closeup3,  D = Closeup4
#
# Output layout:
#   data/ami_train_visual/{MID}.json          ← per-meeting manifest
#   data/ami_train_visual/{MID}/mouth_roi/    ← .npy mouth-ROI clips
#   data/ami_train_visual/{MID}/enrollment_faces/  ← .jpg enrollment frames
#
# Usage:
#   bash scripts/build_ami_train_manifests.sh [--jobs N] [--max-turns N]
#
# Requirements: dlib models in checkpoints/, AV-HuBERT on PYTHONPATH
# =============================================================================
set -euo pipefail

JOBS=2          # ROI extraction is CPU-heavy; keep low unless you have many cores
MAX_TURNS=50    # per meeting — increase for more training data
MAX_PER_SPK=13  # balanced cap: 50 turns / ~4 speakers
MIN_SECS=1.0
MAX_SECS=12.0
ROI_BACKEND=dlib

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)        JOBS="$2"        ; shift 2 ;;
        --max-turns)   MAX_TURNS="$2"   ; shift 2 ;;
        --max-per-spk) MAX_PER_SPK="$2" ; shift 2 ;;
        *) echo "[error] Unknown flag: $1" >&2 ; exit 1 ;;
    esac
done

AUDIO_DIR="${ROOT_DIR}/datasets/ami/audio"
VIDEO_DIR="${ROOT_DIR}/datasets/ami/video"
MANIFEST_SRC="${ROOT_DIR}/data/ami_test/manifests"   # reuse existing annotation JSONs
# Training annotations live in a different dir — point to yours if available
# Fall back to test manifests dir prefix-matched on meeting ID
OUT_BASE="${ROOT_DIR}/data/ami_train_visual"
mkdir -p "$OUT_BASE"

# Standard speaker→closeup mapping for 4-person AMI meetings
SPK_CLOSEUP="A=Closeup1 B=Closeup2 C=Closeup3 D=Closeup4"

# --------------------------------------------------------------------------- #
# Training meetings (same list as download_ami_train.sh)
# --------------------------------------------------------------------------- #
TRAIN_MEETINGS=(
    ES2002a ES2002b ES2002c ES2002d
    ES2003a ES2003b ES2003c ES2003d
    ES2005a ES2005b ES2005c ES2005d
    ES2006a ES2006b ES2006c ES2006d
    ES2007a ES2007b ES2007c ES2007d
    ES2008a ES2008b ES2008c ES2008d
    ES2009a ES2009b ES2009c ES2009d
    ES2010a ES2010b ES2010c ES2010d
    ES2012a ES2012b ES2012c ES2012d
    ES2013a ES2013b ES2013c ES2013d
    IS1000a IS1000b IS1000c IS1000d
    IS1001a IS1001b IS1001c IS1001d
    IS1002b IS1002c IS1002d
    IS1003a IS1003b IS1003c IS1003d
    IS1004a IS1004b IS1004c IS1004d
    IS1005a IS1005b IS1005c IS1005d
    IS1006a IS1006b IS1006c IS1006d
    IS1007a IS1007b IS1007c IS1007d
    TS3005a TS3005b TS3005c TS3005d
    TS3006a TS3006b TS3006c TS3006d
    TS3007a TS3007b TS3007c TS3007d
    TS3008a TS3008b TS3008c TS3008d
    TS3009a TS3009b TS3009c TS3009d
    TS3010a TS3010b TS3010c TS3010d
    TS3011a TS3011b TS3011c TS3011d
    TS3012a TS3012b TS3012c TS3012d
)

_process_meeting() {
    local MID="$1"

    # Need both the base annotation manifest and at least one closeup video
    local base_manifest
    # Training annotations path — adjust if you have a separate train annotations dir
    # For now we look in data/ami_test/manifests (same XML-derived JSONs) and
    # also check data/ami_train/manifests if it exists.
    for candidate in \
        "${ROOT_DIR}/data/ami_train/manifests/${MID}.json" \
        "${ROOT_DIR}/data/ami_test/manifests/${MID}.json"; do
        if [[ -f "$candidate" ]]; then
            base_manifest="$candidate"
            break
        fi
    done

    if [[ -z "${base_manifest:-}" ]]; then
        echo "[skip] ${MID}: no annotation manifest found"
        return
    fi

    # Need at least Closeup1 video
    if [[ ! -f "${VIDEO_DIR}/${MID}.Closeup1.avi" ]]; then
        echo "[skip] ${MID}: Closeup1 video missing"
        return
    fi

    local out_manifest="${OUT_BASE}/${MID}.json"
    if [[ -f "$out_manifest" ]]; then
        echo "[exists] ${MID} — skipping (delete to re-extract)"
        return
    fi

    echo "[proc]  ${MID}"
    python "${ROOT_DIR}/scripts/prepare_ami_visual_manifest.py" \
        --manifest         "$base_manifest" \
        --ami-video-dir    "$VIDEO_DIR" \
        --out-manifest     "$out_manifest" \
        --out-dir          "${OUT_BASE}/${MID}" \
        --speaker-closeup  $SPK_CLOSEUP \
        --max-turns        "$MAX_TURNS" \
        --max-turns-per-speaker "$MAX_PER_SPK" \
        --min-turn-secs    "$MIN_SECS" \
        --max-turn-secs    "$MAX_SECS" \
        --roi-backend      "$ROI_BACKEND" \
        2>&1 | tail -5
}

export -f _process_meeting
export ROOT_DIR OUT_BASE VIDEO_DIR SPK_CLOSEUP MAX_TURNS MAX_PER_SPK MIN_SECS MAX_SECS ROI_BACKEND

echo "[info] Building manifests for ${#TRAIN_MEETINGS[@]} meetings (jobs=${JOBS})"

if command -v parallel &>/dev/null; then
    printf '%s\n' "${TRAIN_MEETINGS[@]}" | \
        parallel --jobs "$JOBS" --bar _process_meeting {}
else
    for MID in "${TRAIN_MEETINGS[@]}"; do
        _process_meeting "$MID"
    done
fi

echo ""
echo "=== Summary ==="
DONE=$(find "$OUT_BASE" -maxdepth 1 -name '*.json' | wc -l)
echo "Manifests built: ${DONE} / ${#TRAIN_MEETINGS[@]}"
echo ""
echo "Next step — run Stage 2 training:"
echo "  python scripts/train_stage2.py \\"
echo "      --config one_go/runs/config_real_en.yaml \\"
echo "      --train-manifest-dir data/ami_train_visual \\"
echo "      --out checkpoints/stage2"
