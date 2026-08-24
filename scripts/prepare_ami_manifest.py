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
  - Channel, camera, and corpus-global participant mappings are read from the
    official corpusResources/meetings.xml. They must not be hard-coded.
  - mouth_roi is set to null here and populated by
    prepare_ami_visual_manifest.py from the verified Closeup stream.
  - Keep train/dev/test meeting lists disjoint. The batch visual builder also
    excludes legacy test entries duplicated in train/dev.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ami_metadata import apply_ami_speaker_metadata, load_ami_meetings  # noqa: E402

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

def ffmpeg_slice(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    sr: int = 16000,
    overwrite: bool = True,
) -> None:
    """Cut src[start:end] → dst at sr Hz mono 16-bit PCM WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
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


def ffmpeg_concat_enrollment(
    src: Path,
    dst: Path,
    segments: list[dict],
    duration_s: float,
    sr: int = 16000,
) -> None:
    """Concatenate selected source turns into a single enrollment WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ami_enroll_", dir=dst.parent) as tmp:
        tmp_dir = Path(tmp)
        list_path = tmp_dir / "concat.txt"
        part_paths: list[Path] = []
        for i, seg in enumerate(segments):
            part = tmp_dir / f"part_{i:03d}.wav"
            ffmpeg_slice(src, part, float(seg["start"]), float(seg["end"]), sr=sr)
            part_paths.append(part)
        with open(list_path, "w", encoding="utf-8") as fh:
            for part in part_paths:
                escaped = part.resolve().as_posix().replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-t", f"{duration_s:.3f}",
                "-ar", str(sr), "-ac", "1",
                str(dst),
            ],
            check=True,
        )


def segment_rms_dbfs(src: Path, start: float, end: float, sr: int = 16000) -> float:
    """Approximate segment loudness; -inf means unavailable/invalid."""
    try:
        import numpy as np
        raw = subprocess.check_output(
            [
                "ffmpeg", "-v", "error",
                "-ss", f"{start:.6f}", "-to", f"{end:.6f}",
                "-i", str(src),
                "-ar", str(sr), "-ac", "1",
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-",
            ]
        )
        samples = np.frombuffer(raw, dtype=np.float32)
        if samples.size == 0:
            return -math.inf
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        return 20.0 * math.log10(max(rms, 1.0e-8))
    except Exception:
        return -math.inf


def select_enrollment_segments(
    src_wav: Path,
    segs: list[dict],
    words: list[dict],
    target_secs: float,
    min_secs: float,
    max_secs: float,
) -> list[dict]:
    """Pick high-RMS, text-bearing turns for speaker enrollment."""
    candidates: list[dict] = []
    for seg in segs:
        dur = float(seg["end"]) - float(seg["start"])
        if dur < min_secs or dur > max_secs:
            continue
        if not ref_text_for_segment(words, float(seg["start"]), float(seg["end"])):
            continue
        item = dict(seg)
        item["rms_dbfs"] = segment_rms_dbfs(src_wav, float(seg["start"]), float(seg["end"]))
        candidates.append(item)

    candidates.sort(key=lambda x: x["rms_dbfs"], reverse=True)
    selected: list[dict] = []
    total = 0.0
    for seg in candidates:
        selected.append(seg)
        total += float(seg["end"]) - float(seg["start"])
        if total >= target_secs:
            break
    selected.sort(key=lambda x: x["start"])
    return selected


# ---------------------------------------------------------------------------
# Per-meeting logic
# ---------------------------------------------------------------------------

def process_meeting(
    meeting_id: str,
    ami_root: Path,
    out_root: Path,
    enroll_secs: float,
    min_turn_secs: float,
    enrollment_mode: str,
    enroll_min_turn_secs: float,
    enroll_max_turn_secs: float,
    meeting_speakers: dict[str, dict] | None = None,
    overwrite_turn_audio: bool = False,
    dataset_build_id: str | None = None,
) -> dict:
    """
    Process one AMI meeting and return a session manifest dict.

    With meetings.xml metadata, speaker IDs use AMI corpus-global participant
    IDs. The local ``{meeting_id}_{agent}`` ID is retained separately.
    """
    audio_dir = ami_root / "audio"
    ann_dir   = ami_root / "annotations"

    # ------------------------------------------------------------------
    # 1. Discover which speakers have complete data
    # ------------------------------------------------------------------
    channel_by_speaker = (
        {
            agent: int(meta["channel"])
            for agent, meta in meeting_speakers.items()
            if meta.get("channel") is not None
        }
        if meeting_speakers
        else dict(SPK_TO_CHANNEL)
    )
    speakers_present: list[str] = []
    for spk, ch in channel_by_speaker.items():
        wav      = audio_dir / f"{meeting_id}.Headset-{ch}.wav"
        seg_xml  = ann_dir / "segments" / f"{meeting_id}.{spk}.segments.xml"
        wrd_xml  = ann_dir / "words"    / f"{meeting_id}.{spk}.words.xml"
        if wav.exists() and seg_xml.exists() and wrd_xml.exists():
            speakers_present.append(spk)

    if not speakers_present:
        raise RuntimeError(
            f"No complete data (audio + segments + words) for meeting {meeting_id}"
        )

    speaker_data: dict[str, dict] = {}
    for spk in speakers_present:
        speaker_data[spk] = {
            "segs": parse_segments(ann_dir / "segments" / f"{meeting_id}.{spk}.segments.xml"),
            "words": parse_words(ann_dir / "words" / f"{meeting_id}.{spk}.words.xml"),
        }

    # ------------------------------------------------------------------
    # 2. Build enrollment clips
    # ------------------------------------------------------------------
    enrol_entries: list[dict] = []
    for spk in speakers_present:
        ch      = channel_by_speaker[spk]
        src_wav = audio_dir / f"{meeting_id}.Headset-{ch}.wav"
        enrol_wav = out_root / "enrollment" / f"{meeting_id}_{spk}.wav"
        selected = (
            select_enrollment_segments(
                src_wav,
                speaker_data[spk]["segs"],
                speaker_data[spk]["words"],
                target_secs=enroll_secs,
                min_secs=enroll_min_turn_secs,
                max_secs=enroll_max_turn_secs,
            )
            if enrollment_mode == "turn_quality"
            else []
        )
        print(f"  [{spk}] enrollment clip → {enrol_wav.name}")
        if selected:
            selected_secs = sum(float(s["end"]) - float(s["start"]) for s in selected)
            print(f"      using {len(selected)} quality turns ({selected_secs:.1f}s)")
            ffmpeg_concat_enrollment(src_wav, enrol_wav, selected, duration_s=enroll_secs)
            actual_enrollment_mode = "turn_quality"
        else:
            print(f"      using first {enroll_secs:.1f}s fallback")
            ffmpeg_enrollment(src_wav, enrol_wav, duration_s=enroll_secs)
            actual_enrollment_mode = "first_seconds_fallback"
        enrol_entries.append({
            "speaker_id":       f"{meeting_id}_{spk}",
            "session_speaker_id": f"{meeting_id}_{spk}",
            "nxt_agent": spk,
            "identity_scope": "meeting_local",
            "microphone_id": f"Headset-{ch}",
            "enrollment_audio": str(enrol_wav),
            "meta": {
                "enrollment_mode": actual_enrollment_mode,
                "enrollment_selected_seconds": round(
                    sum(float(s["end"]) - float(s["start"]) for s in selected), 3
                ),
                "enrollment_selected_turns": [
                    {
                        "id": s.get("id"),
                        "start": round(float(s["start"]), 3),
                        "end": round(float(s["end"]), 3),
                        "rms_dbfs": round(float(s.get("rms_dbfs", -math.inf)), 2),
                    }
                    for s in selected
                ],
            },
        })

    # ------------------------------------------------------------------
    # 3. Build turns
    # ------------------------------------------------------------------
    turns: list[dict] = []
    for spk in speakers_present:
        ch      = channel_by_speaker[spk]
        src_wav = audio_dir / f"{meeting_id}.Headset-{ch}.wav"

        words = speaker_data[spk]["words"]
        segs  = speaker_data[spk]["segs"]

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
            ffmpeg_slice(
                src_wav,
                turn_wav,
                seg["start"],
                seg["end"],
                overwrite=overwrite_turn_audio,
            )

            turns.append({
                "turn_id":    seg["id"],
                "start":      round(seg["start"], 3),
                "end":        round(seg["end"],   3),
                "audio":      str(turn_wav),
                "mouth_roi":  None,  # set to .npy path after running av_hubert preprocessor
                "ref_text":   ref_text,
                "ref_speaker": f"{meeting_id}_{spk}",
                "session_speaker_id": f"{meeting_id}_{spk}",
                "nxt_agent": spk,
                "identity_scope": "meeting_local",
                "audio_source_type": "individual_headset_microphone",
                "turn_boundary_source": "oracle_reference_transcript",
            })

        print(
            f"  [{spk}] {len(segs)} segments → "
            f"{len([t for t in turns if t['ref_speaker'].endswith(spk)])} turns "
            f"({skipped_short} too short, {skipped_silent} no text)"
        )

    # Sort turns chronologically (interleaved across speakers)
    turns.sort(key=lambda t: t["start"])

    manifest = {
        "meeting_id": meeting_id,
        "speakers": enrol_entries,
        "turns": turns,
        "meta": {
            "source": "AMI manual annotations",
            "config": "ihm",
            "audio_condition": {
                "ami_microphone_setup": "ihm",
                "description": "individual_headset_microphone",
                "far_field": False,
            },
            "turn_boundary_source": "oracle_reference_transcript",
            "diarization_is_system_output": False,
            "speaker_identity_scope": "meeting_local",
            "dataset_build_id": dataset_build_id,
        },
    }
    if meeting_speakers is not None:
        manifest = apply_ami_speaker_metadata(manifest, meeting_id, meeting_speakers)
    return manifest


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
    ap.add_argument("--enrollment-mode", choices=["first_seconds", "turn_quality"],
                    default="turn_quality",
                    help="How to build enrollment clips (default: turn_quality)")
    ap.add_argument("--enroll-min-turn-secs", type=float, default=3.0,
                    help="Minimum duration for turn-quality enrollment segments (default: 3)")
    ap.add_argument("--enroll-max-turn-secs", type=float, default=8.0,
                    help="Maximum duration for turn-quality enrollment segments (default: 8)")
    ap.add_argument("--min-turn-secs", type=float, default=1.0,
                    help="Skip turns shorter than this (default: 1.0 s)")
    ap.add_argument(
        "--meetings-xml",
        type=Path,
        default=None,
        help="AMI meetings.xml; defaults to <ami>/corpusResources/meetings.xml.",
    )
    ap.add_argument(
        "--allow-meeting-local-speakers",
        action="store_true",
        help="Allow legacy A-D meeting-local identities if meetings.xml is unavailable.",
    )
    ap.add_argument(
        "--overwrite-turn-audio",
        action="store_true",
        help="Re-slice turn WAVs even when the destination already exists.",
    )
    ap.add_argument(
        "--dataset-build-id",
        default=None,
        help="Immutable dataset version label recorded in manifest metadata.",
    )
    args = ap.parse_args()

    ami_root = Path(args.ami)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    meetings_xml = args.meetings_xml or (ami_root / "corpusResources" / "meetings.xml")
    if meetings_xml.is_file():
        ami_meetings = load_ami_meetings(meetings_xml)
        print(f"Loaded {len(ami_meetings)} meetings from {meetings_xml}")
    elif args.allow_meeting_local_speakers:
        ami_meetings = {}
        print(
            f"WARNING: {meetings_xml} missing; using meeting-local speaker IDs",
            file=sys.stderr,
        )
    else:
        print(
            f"ERROR: AMI metadata missing: {meetings_xml}\n"
            "Run scripts/fetch_ami_metadata.py or pass --allow-meeting-local-speakers.",
            file=sys.stderr,
        )
        sys.exit(2)

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
            meeting_speakers = ami_meetings.get(mid)
            if ami_meetings and meeting_speakers is None:
                raise KeyError(f"Meeting {mid} is missing from {meetings_xml}")
            manifest  = process_meeting(mid, ami_root, out_root,
                                         enroll_secs=args.enroll_secs,
                                         min_turn_secs=args.min_turn_secs,
                                         enrollment_mode=args.enrollment_mode,
                                         enroll_min_turn_secs=args.enroll_min_turn_secs,
                                         enroll_max_turn_secs=args.enroll_max_turn_secs,
                                         meeting_speakers=meeting_speakers,
                                         overwrite_turn_audio=args.overwrite_turn_audio,
                                         dataset_build_id=args.dataset_build_id)
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
