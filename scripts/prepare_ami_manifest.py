#!/usr/bin/env python3
"""
Prepare AMI corpus → AVSD-GER session manifests.

What this does:
  1. Parses annotations/segments/*.xml  → turn boundaries per speaker
  2. Parses annotations/words/*.xml     → ref_text per turn
  3. Slices audio/Headset-N.wav         → per-turn 16 kHz mono WAV clips
  4. Cuts an enrollment clip            → first --enroll-secs of each headset
  5. Writes one manifest JSON per meeting under --out/manifests/

Usage:
    python scripts/prepare_ami_manifest.py \\
        --ami  datasets/ami \\
        --out  data/ami_test \\
        [--meetings ES2004a ES2011b ...]  # default: all meetings found in audio/
        [--enroll-secs 30]
        [--min-turn-secs 1.0]

Then run eval over all produced manifests:
    for f in data/ami_test/manifests/*.json; do
      python scripts/eval_ablations.py \\
        --config configs/default.yaml \\
        --manifest "$f" \\
        --pool   checkpoints/identity_pool.pt \\
        --out    out/ami_ablation_$(basename "$f" .json).json
    done

Notes on AMI coverage:
  - Channel → speaker mapping: Headset-0=A, 1=B, 2=C, 3=D (from meetings.xml).
  - Video: only Closeup1 and Closeup2 are present (2 of 4 speakers per meeting).
    mouth_roi is set to null for ALL turns — run av_hubert/preparation/align_mouth.py
    on the Closeup AVI files if you need lip features.
  - The standard AMI test split consists of the meetings present in datasets/ami/audio/:
    ES2004a-d, ES2011a-d, IS1008a-d, IS1009a-d  (16 meetings, 4 speakers each).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

# AMI convention: nxt_agent letter → Headset channel number
SPK_TO_CHANNEL: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}


# ---------------------------------------------------------------------------
# XML parsers
# ---------------------------------------------------------------------------

def parse_words(words_xml: Path) -> list[dict]:
    """Return [{id, start, end, text}, ...] sorted by start time."""
    tree = ET.parse(words_xml)
    items: list[dict] = []
    for elem in tree.getroot():
        start_s = elem.get("starttime")
        end_s = elem.get("endtime")
        if start_s is None or end_s is None:
            continue
        start = float(start_s)
        end = float(end_s)
        if elem.tag == "w":
            text = (elem.text or "").strip()
            if text:
                nid = elem.get("{http://nite.sourceforge.net/}id", "")
                items.append({"id": nid, "start": start, "end": end, "text": text})
    items.sort(key=lambda x: x["start"])
    return items


def parse_segments(segments_xml: Path) -> list[dict]:
    """Return [{id, start, end}, ...] sorted by start time."""
    tree = ET.parse(segments_xml)
    segs: list[dict] = []
    for seg in tree.getroot().findall("segment"):
        nid = seg.get("{http://nite.sourceforge.net/}id", "")
        start_s = seg.get("transcriber_start")
        end_s = seg.get("transcriber_end")
        if start_s is None or end_s is None:
            continue
        start = float(start_s)
        end = float(end_s)
        if end > start:
            segs.append({"id": nid, "start": start, "end": end})
    segs.sort(key=lambda x: x["start"])
    return segs


def ref_text_for_segment(words: list[dict], seg_start: float, seg_end: float) -> str:
    """Collect words whose starttime falls within [seg_start, seg_end]."""
    toks = [
        w["text"] for w in words
        if seg_start <= w["start"] <= seg_end
    ]
    return " ".join(toks).strip()


# ---------------------------------------------------------------------------
# Audio helpers (uses ffmpeg — no extra Python dep needed)
# ---------------------------------------------------------------------------

def ffmpeg_slice(src: Path, dst: Path, start: float, end: float, sr: int = 16000) -> None:
    """Cut src[start:end] → dst at sr Hz mono 16-bit PCM WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ss", f"{start:.6f}", "-to", f"{end:.6f}",
            "-ar", str(sr), "-ac", "1",
            str(dst),
        ],
        check=True,
    )


def ffmpeg_enrollment(src: Path, dst: Path, duration_s: float, sr: int = 16000) -> None:
    """Extract first duration_s seconds of src → dst at sr Hz mono."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-t", f"{duration_s:.3f}",
            "-ar", str(sr), "-ac", "1",
            str(dst),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Per-meeting logic
# ---------------------------------------------------------------------------

def process_meeting(
    meeting_id: str,
    ami_root: Path,
    out_root: Path,
    enroll_secs: float,
    min_turn_secs: float,
) -> dict:
    """
    Process one AMI meeting and return a session manifest dict.

    Speaker IDs in the manifest use the format  {meeting_id}_{spk},
    e.g. "ES2004a_A".  This is stable across meetings so the pipeline
    can track cross-meeting speaker consistency if desired.
    """
    audio_dir = ami_root / "audio"
    ann_dir   = ami_root / "annotations"

    # ------------------------------------------------------------------
    # 1. Discover which speakers have complete data
    # ------------------------------------------------------------------
    speakers_present: list[str] = []
    for spk, ch in SPK_TO_CHANNEL.items():
        wav      = audio_dir / f"{meeting_id}.Headset-{ch}.wav"
        seg_xml  = ann_dir / "segments" / f"{meeting_id}.{spk}.segments.xml"
        wrd_xml  = ann_dir / "words"    / f"{meeting_id}.{spk}.words.xml"
        if wav.exists() and seg_xml.exists() and wrd_xml.exists():
            speakers_present.append(spk)

    if not speakers_present:
        raise RuntimeError(
            f"No complete data (audio + segments + words) for meeting {meeting_id}"
        )

    # ------------------------------------------------------------------
    # 2. Build enrollment clips
    # ------------------------------------------------------------------
    enrol_entries: list[dict] = []
    for spk in speakers_present:
        ch      = SPK_TO_CHANNEL[spk]
        src_wav = audio_dir / f"{meeting_id}.Headset-{ch}.wav"
        enrol_wav = out_root / "enrollment" / f"{meeting_id}_{spk}.wav"
        print(f"  [{spk}] enrollment clip → {enrol_wav.name}")
        ffmpeg_enrollment(src_wav, enrol_wav, duration_s=enroll_secs)
        enrol_entries.append({
            "speaker_id":       f"{meeting_id}_{spk}",
            "enrollment_audio": str(enrol_wav),
        })

    # ------------------------------------------------------------------
    # 3. Build turns
    # ------------------------------------------------------------------
    turns: list[dict] = []
    for spk in speakers_present:
        ch      = SPK_TO_CHANNEL[spk]
        src_wav = audio_dir / f"{meeting_id}.Headset-{ch}.wav"
        seg_xml = ann_dir / "segments" / f"{meeting_id}.{spk}.segments.xml"
        wrd_xml = ann_dir / "words"    / f"{meeting_id}.{spk}.words.xml"

        words = parse_words(wrd_xml)
        segs  = parse_segments(seg_xml)

        skipped_short = 0
        skipped_silent = 0
        for seg in segs:
            dur = seg["end"] - seg["start"]
            if dur < min_turn_secs:
                skipped_short += 1
                continue

            ref_text = ref_text_for_segment(words, seg["start"], seg["end"])
            if not ref_text:
                skipped_silent += 1
                continue

            # Safe filename: replace non-alphanumeric with _
            seg_fname = re.sub(r"[^a-zA-Z0-9]", "_", seg["id"])
            turn_wav  = out_root / "audio" / f"{seg_fname}.wav"
            ffmpeg_slice(src_wav, turn_wav, seg["start"], seg["end"])

            turns.append({
                "turn_id":    seg["id"],
                "start":      round(seg["start"], 3),
                "end":        round(seg["end"],   3),
                "audio":      str(turn_wav),
                "mouth_roi":  None,  # set to .npy path after running av_hubert preprocessor
                "ref_text":   ref_text,
                "ref_speaker": f"{meeting_id}_{spk}",
            })

        print(
            f"  [{spk}] {len(segs)} segments → "
            f"{len([t for t in turns if t['ref_speaker'].endswith(spk)])} turns "
            f"({skipped_short} too short, {skipped_silent} no text)"
        )

    # Sort turns chronologically (interleaved across speakers)
    turns.sort(key=lambda t: t["start"])

    return {"speakers": enrol_entries, "turns": turns}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare AMI corpus → AVSD-GER session manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ami",           default="datasets/ami",
                    help="Path to AMI dataset root (default: datasets/ami)")
    ap.add_argument("--out",           default="data/ami_test",
                    help="Output root directory (default: data/ami_test)")
    ap.add_argument("--meetings",      nargs="*",
                    help="Meeting IDs to process (default: all found in audio/)")
    ap.add_argument("--enroll-secs",   type=float, default=30.0,
                    help="Enrollment clip duration in seconds (default: 30)")
    ap.add_argument("--min-turn-secs", type=float, default=1.0,
                    help="Skip turns shorter than this (default: 1.0 s)")
    args = ap.parse_args()

    ami_root = Path(args.ami)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover meetings
    if args.meetings:
        meetings = args.meetings
    else:
        wav_files = sorted((ami_root / "audio").glob("*.Headset-0.wav"))
        meetings  = [w.stem.replace(".Headset-0", "") for w in wav_files]

    if not meetings:
        print(f"ERROR: no meetings found under {ami_root}/audio/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(meetings)} meeting(s): {meetings}\n")

    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for mid in meetings:
        print(f"=== {mid} ===")
        try:
            manifest  = process_meeting(mid, ami_root, out_root,
                                        enroll_secs=args.enroll_secs,
                                        min_turn_secs=args.min_turn_secs)
            out_json  = manifests_dir / f"{mid}.json"
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, ensure_ascii=False)
            n_spk   = len(manifest["speakers"])
            n_turns = len(manifest["turns"])
            print(f"  → {n_spk} speakers, {n_turns} turns → {out_json}\n")
            ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}\n", file=sys.stderr)
            fail += 1

    print(f"Done: {ok} OK, {fail} failed.\n")

    if ok:
        print("Run stub eval over all manifests (Phase 0/A setup, no real models needed):")
        print(f"""
  for f in {manifests_dir}/*.json; do
    python scripts/eval_ablations.py \\
      --config configs/default.yaml \\
      --manifest "$f" \\
      --pool   checkpoints/identity_pool.pt \\
      --out    out/ami_ablation_$(basename "$f" .json).json \\
      --no-power
  done
""")
        print("Or run a single meeting as a quick smoke-test:")
        first = sorted(manifests_dir.glob("*.json"))[0]
        print(f"""
  python scripts/eval_ablations.py \\
    --config configs/default.yaml \\
    --manifest {first} \\
    --pool   checkpoints/identity_pool.pt \\
    --out    out/ami_ablation_{first.stem}.json \\
    --no-power
""")


if __name__ == "__main__":
    main()
