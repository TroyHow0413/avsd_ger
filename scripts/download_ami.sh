#!/usr/bin/env bash
# =============================================================================
# download_ami.sh  —  Download AMI corpus (train / dev / test)
#
# Standard AMI split (Kaldi / ESPnet / pyannote):
#   train : ES2002-3, ES2005-10, ES2012-13, IS1000-7, TS3005-12  (~136 sessions)
#   dev   : ES2011a-d, IS1008a-d, TS3004a-d                       (12 sessions)
#   test  : ES2004a-d, IS1009a-d, TS3003a-d                       (12 sessions)
#
# Usage:
#   bash scripts/download_ami.sh [SPLIT...] [OPTIONS]
#
# SPLIT (repeatable, default = all three):
#   --train          Download training set
#   --dev            Download dev set
#   --test           Download test set
#
# OPTIONS:
#   --audio-only     Skip Closeup video (~saves 200-400 GB)
#   --jobs N         Parallel wget workers (default: 4)
#   --out-dir DIR    Root dir with audio/ and video/ (default: datasets/ami)
#   --overwrite      Re-download even if file already exists
#   -h, --help       Show this help
#
# Examples:
#   bash scripts/download_ami.sh                        # everything
#   bash scripts/download_ami.sh --test --dev           # eval sets only
#   bash scripts/download_ami.sh --train --audio-only   # train audio only
#   bash scripts/download_ami.sh --test --overwrite     # force re-download test
#
# Size estimates:
#   Audio (all splits) : ~22 GB
#   Video (all splits) : ~300-450 GB
#
# Requirements: wget; GNU parallel optional (speeds up --jobs)
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DO_TRAIN=0
DO_DEV=0
DO_TEST=0
AUDIO_ONLY=0
OVERWRITE=0
JOBS=4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${ROOT_DIR}/datasets/ami"

# --------------------------------------------------------------------------- #
# Arg parse
# --------------------------------------------------------------------------- #
if [[ $# -eq 0 ]]; then
    DO_TRAIN=1; DO_DEV=1; DO_TEST=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)      DO_TRAIN=1   ; shift ;;
        --dev)        DO_DEV=1     ; shift ;;
        --test)       DO_TEST=1    ; shift ;;
        --audio-only) AUDIO_ONLY=1 ; shift ;;
        --overwrite)  OVERWRITE=1  ; shift ;;
        --jobs)       JOBS="$2"    ; shift 2 ;;
        --out-dir)    OUT_DIR="$2" ; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//' | sed -n '2,30p'
            exit 0 ;;
        *) echo "[error] Unknown argument: $1" >&2 ; exit 1 ;;
    esac
done

# If no split flag was given after parsing (shouldn't happen but guard anyway)
if [[ $DO_TRAIN -eq 0 && $DO_DEV -eq 0 && $DO_TEST -eq 0 ]]; then
    DO_TRAIN=1; DO_DEV=1; DO_TEST=1
fi

AUDIO_DIR="${OUT_DIR}/audio"
VIDEO_DIR="${OUT_DIR}/video"
mkdir -p "$AUDIO_DIR" "$VIDEO_DIR"

BASE="http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

# --------------------------------------------------------------------------- #
# Meeting ID lists per split
# --------------------------------------------------------------------------- #

# ---- TRAIN -----------------------------------------------------------------
# ES: skip ES2004 (test), ES2011 (dev)
ES_TRAIN=(
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
)
# IS: IS1002a does not exist in AMI; skip IS1008 (dev), IS1009 (test)
IS_TRAIN=(
    IS1000a IS1000b IS1000c IS1000d
    IS1001a IS1001b IS1001c IS1001d
    IS1002b IS1002c IS1002d
    IS1003a IS1003b IS1003c IS1003d
    IS1004a IS1004b IS1004c IS1004d
    IS1005a IS1005b IS1005c IS1005d
    IS1006a IS1006b IS1006c IS1006d
    IS1007a IS1007b IS1007c IS1007d
)
# TS: skip TS3003 (test), TS3004 (dev)
TS_TRAIN=(
    TS3005a TS3005b TS3005c TS3005d
    TS3006a TS3006b TS3006c TS3006d
    TS3007a TS3007b TS3007c TS3007d
    TS3008a TS3008b TS3008c TS3008d
    TS3009a TS3009b TS3009c TS3009d
    TS3010a TS3010b TS3010c TS3010d
    TS3011a TS3011b TS3011c TS3011d
    TS3012a TS3012b TS3012c TS3012d
)

# ---- DEV -------------------------------------------------------------------
ES_DEV=( ES2011a ES2011b ES2011c ES2011d )
IS_DEV=( IS1008a IS1008b IS1008c IS1008d )
TS_DEV=( TS3004a TS3004b TS3004c TS3004d )

# ---- TEST ------------------------------------------------------------------
ES_TEST=( ES2004a ES2004b ES2004c ES2004d )
IS_TEST=( IS1009a IS1009b IS1009c IS1009d )
TS_TEST=( TS3003a TS3003b TS3003c TS3003d )

# --------------------------------------------------------------------------- #
# Build the final list of meetings to download
# --------------------------------------------------------------------------- #
MEETINGS=()
[[ $DO_TRAIN -eq 1 ]] && MEETINGS+=( "${ES_TRAIN[@]}" "${IS_TRAIN[@]}" "${TS_TRAIN[@]}" )
[[ $DO_DEV   -eq 1 ]] && MEETINGS+=( "${ES_DEV[@]}"   "${IS_DEV[@]}"   "${TS_DEV[@]}"   )
[[ $DO_TEST  -eq 1 ]] && MEETINGS+=( "${ES_TEST[@]}"  "${IS_TEST[@]}"  "${TS_TEST[@]}"  )

# --------------------------------------------------------------------------- #
# Print plan
# --------------------------------------------------------------------------- #
SPLITS=""
[[ $DO_TRAIN -eq 1 ]] && SPLITS+="train "
[[ $DO_DEV   -eq 1 ]] && SPLITS+="dev "
[[ $DO_TEST  -eq 1 ]] && SPLITS+="test"

echo "============================================================"
echo " AMI downloader"
echo "============================================================"
printf " Splits     : %s\n"   "${SPLITS}"
printf " Sessions   : %d\n"   "${#MEETINGS[@]}"
printf " Audio-only : %s\n"   "$([[ $AUDIO_ONLY -eq 1 ]] && echo yes || echo no)"
printf " Overwrite  : %s\n"   "$([[ $OVERWRITE  -eq 1 ]] && echo yes || echo no)"
printf " Jobs       : %d\n"   "$JOBS"
printf " Out dir    : %s\n"   "$OUT_DIR"
echo "------------------------------------------------------------"
echo ""

# --------------------------------------------------------------------------- #
# Download helper
# --------------------------------------------------------------------------- #
_wget_one() {
    # Args: url|dst|overwrite
    local url dst overwrite
    IFS='|' read -r url dst overwrite <<< "$1"

    if [[ -f "$dst" && "$overwrite" -eq 0 ]]; then
        echo "[skip] $(basename "$dst")  (exists)"
        return 0
    fi

    # Atomic write via .part file
    wget \
        --quiet \
        --tries=5 \
        --retry-connrefused \
        --timeout=60 \
        --continue \
        -O "${dst}.part" \
        "$url" 2>&1 || true

    if [[ -f "${dst}.part" && -s "${dst}.part" ]]; then
        mv "${dst}.part" "$dst"
        echo "[ok]   $(basename "$dst")"
    else
        rm -f "${dst}.part"
        # 404 or timeout — not fatal (some Closeup3/4 don't exist for all meetings)
        echo "[miss] $(basename "$dst")"
    fi
}
export -f _wget_one

# --------------------------------------------------------------------------- #
# Build job list
# --------------------------------------------------------------------------- #
AUDIO_LIST=$(mktemp)
VIDEO_LIST=$(mktemp)
trap 'rm -f "$AUDIO_LIST" "$VIDEO_LIST"' EXIT

for MID in "${MEETINGS[@]}"; do
    for N in 0 1 2 3; do
        fname="${MID}.Headset-${N}.wav"
        echo "${BASE}/${MID}/audio/${fname}|${AUDIO_DIR}/${fname}|${OVERWRITE}" \
            >> "$AUDIO_LIST"
    done

    if [[ $AUDIO_ONLY -eq 0 ]]; then
        for C in 1 2 3 4; do
            fname="${MID}.Closeup${C}.avi"
            echo "${BASE}/${MID}/video/${fname}|${VIDEO_DIR}/${fname}|${OVERWRITE}" \
                >> "$VIDEO_LIST"
        done
    fi
done

AUDIO_TOTAL=$(wc -l < "$AUDIO_LIST")
VIDEO_TOTAL=$(wc -l < "$VIDEO_LIST")

# Count already-present files (skip candidates)
AUDIO_HAVE=$(awk -F'|' '{print $2}' "$AUDIO_LIST" | xargs -I{} bash -c '[[ -f "{}" ]] && echo 1 || echo 0' | grep -c 1 || true)
VIDEO_HAVE=$(awk -F'|' '{print $2}' "$VIDEO_LIST" | xargs -I{} bash -c '[[ -f "{}" ]] && echo 1 || echo 0' | grep -c 1 || true)

echo "[info] Audio : ${AUDIO_HAVE}/${AUDIO_TOTAL} already on disk"
echo "[info] Video : ${VIDEO_HAVE}/${VIDEO_TOTAL} already on disk"
echo ""

# --------------------------------------------------------------------------- #
# Run downloads
# --------------------------------------------------------------------------- #
_run() {
    local list="$1" label="$2"
    [[ -s "$list" ]] || return
    echo "=== ${label} ==="
    if command -v parallel &>/dev/null; then
        parallel --jobs "$JOBS" --bar _wget_one {} :::: "$list"
    else
        # xargs fallback
        xargs -P "$JOBS" -I '{}' bash -c '_wget_one "$@"' _ '{}' < "$list"
    fi
    echo ""
}

_run "$AUDIO_LIST" "Audio — Headset WAV"
_run "$VIDEO_LIST" "Video — Closeup AVI"

# --------------------------------------------------------------------------- #
# Final summary
# --------------------------------------------------------------------------- #
echo "============================================================"
echo " Summary"
echo "============================================================"
AUDIO_FINAL=$(find "$AUDIO_DIR" -name '*.wav' | wc -l)
VIDEO_FINAL=$(find "$VIDEO_DIR" -name '*.avi' | wc -l)
printf " WAV files on disk : %d\n" "$AUDIO_FINAL"
printf " AVI files on disk : %d\n" "$VIDEO_FINAL"
echo ""
du -sh "$AUDIO_DIR" "$VIDEO_DIR" 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  bash scripts/build_ami_train_manifests.sh   # extract mouth-ROI for train"
echo "  python scripts/eval_ablations.py ...        # run eval on test/dev"
