"""Prepare a tiny AMI ES2004a smoke-test manifest.

This extracts one short utterance and one enrollment clip from the AMI headset
audio using only the Python standard library, then writes a manifest compatible
with one_go/main.py and scripts/run_sample.py.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import wave
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
AMI = ROOT / "datasets" / "ami"
OUT_DIR = ROOT / "one_go" / "runs" / "ami_es2004a"
SPEAKER_TO_HEADSET = {"A": 0, "B": 1, "C": 2, "D": 3}


def _child_ids(href: str) -> tuple[int, int]:
    ids = [int(x) for x in re.findall(r"words(\d+)", href)]
    if not ids:
        raise ValueError(f"Cannot parse word ids from href={href!r}")
    return min(ids), max(ids)


def _load_words(path: Path) -> dict[int, str]:
    root = ET.parse(path).getroot()
    words: dict[int, str] = {}
    for elem in root:
        if not elem.tag.endswith("w"):
            continue
        nid = elem.attrib.get("{http://nite.sourceforge.net/}id", "")
        m = re.search(r"words(\d+)$", nid)
        if not m:
            continue
        token = html.unescape((elem.text or "").strip())
        if token:
            words[int(m.group(1))] = token
    return words


def _load_segments(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    segs: list[dict] = []
    for seg in root:
        if not seg.tag.endswith("segment"):
            continue
        child = next((c for c in seg if c.tag.endswith("child")), None)
        if child is None:
            continue
        start_id, end_id = _child_ids(child.attrib["href"])
        segs.append({
            "start": float(seg.attrib["transcriber_start"]),
            "end": float(seg.attrib["transcriber_end"]),
            "start_id": start_id,
            "end_id": end_id,
        })
    return segs


def _text_for_segment(words: dict[int, str], seg: dict) -> str:
    toks = [words[i] for i in range(seg["start_id"], seg["end_id"] + 1) if i in words]
    text = " ".join(toks)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
    return text.strip()


def _pick_segment(segs: list[dict], words: dict[int, str], min_s: float, max_s: float) -> dict:
    candidates = []
    for seg in segs:
        dur = seg["end"] - seg["start"]
        text = _text_for_segment(words, seg)
        n_words = len(text.split())
        if min_s <= dur <= max_s and n_words >= 4:
            candidates.append((n_words, dur, seg, text))
    if not candidates:
        raise RuntimeError("No suitable AMI segment found.")
    candidates.sort(key=lambda x: (abs(x[1] - 6.0), -x[0]))
    _, _, seg, text = candidates[0]
    seg = dict(seg)
    seg["text"] = text
    return seg


def _write_wav_slice(src: Path, dst: Path, start_s: float, end_s: float, pad_s: float = 0.15) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src), "rb") as r:
        rate = r.getframerate()
        start = max(0, int((start_s - pad_s) * rate))
        end = min(r.getnframes(), int((end_s + pad_s) * rate))
        r.setpos(start)
        frames = r.readframes(end - start)
        params = r.getparams()
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> int:
    p = argparse.ArgumentParser(description="Build one AMI ES2004a smoke manifest.")
    p.add_argument("--meeting", default="ES2004a")
    p.add_argument("--speaker", choices=["A", "B", "C", "D"], default="B")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    headset = SPEAKER_TO_HEADSET[args.speaker]
    wav = AMI / "audio" / f"{args.meeting}.Headset-{headset}.wav"
    words_path = AMI / "annotations" / "words" / f"{args.meeting}.{args.speaker}.words.xml"
    segs_path = AMI / "annotations" / "segments" / f"{args.meeting}.{args.speaker}.segments.xml"

    words = _load_words(words_path)
    segs = _load_segments(segs_path)
    utt_seg = _pick_segment(segs, words, min_s=3.0, max_s=8.0)
    enrol_seg = _pick_segment(
        [s for s in segs if s["start"] > utt_seg["end"] + 30.0],
        words,
        min_s=8.0,
        max_s=16.0,
    )

    utt_wav = out_dir / f"{args.meeting}_{args.speaker}_utt.wav"
    enrol_wav = out_dir / f"{args.meeting}_{args.speaker}_enroll.wav"
    manifest_path = out_dir / f"{args.meeting}_{args.speaker}_manifest.json"

    _write_wav_slice(wav, utt_wav, utt_seg["start"], utt_seg["end"])
    _write_wav_slice(wav, enrol_wav, enrol_seg["start"], enrol_seg["end"])

    manifest = {
        "speakers": [
            {
                "speaker_id": f"{args.meeting}_{args.speaker}",
                "enrollment_audio": _repo_path(enrol_wav),
                "meta": {
                    "meeting": args.meeting,
                    "speaker": args.speaker,
                    "headset": headset,
                    "enroll_start": enrol_seg["start"],
                    "enroll_end": enrol_seg["end"],
                },
            }
        ],
        "utterances": [
            {
                "utt_id": f"{args.meeting}_{args.speaker}_utt",
                "speaker_id": f"{args.meeting}_{args.speaker}",
                "audio": _repo_path(utt_wav),
                "mouth_roi": None,
                "transcript_gold": utt_seg["text"],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ami] utterance: {utt_seg['start']:.3f}-{utt_seg['end']:.3f}s  {utt_seg['text']}")
    print(f"[ami] enroll  : {enrol_seg['start']:.3f}-{enrol_seg['end']:.3f}s")
    print(f"[ami] wrote   : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
