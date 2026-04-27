"""Run the full AVSD-GER pipeline on a single manifest utterance.

With `stub_backbones: true` in the config, this runs entirely on random
tensors and prints the trace of C1 → C2 → C3 so you can confirm the loop
routes correctly.

    python scripts/run_sample.py --manifest data/sample_manifest.json --utt utt_0001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.pipeline import AVSDGERPipeline  # noqa: E402


def _load_audio(path: str | None) -> np.ndarray | torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.randn(16000 * 3)
    import soundfile as sf
    wav, sr = sf.read(path)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _load_video_mouth_roi(path: str | None, num_frames: int = 75) -> torch.Tensor:
    """Return [T, 1, 96, 96] mouth ROI in [0, 1]. Stub if path missing."""
    if path is None or not Path(path).exists():
        return torch.rand(num_frames, 1, 96, 96)
    # TODO: wire a real mouth-ROI preprocessor (landmarks + crop) — this
    # repo expects you to run av_hubert/avhubert/preparation/align_mouth.py
    # or equivalent and cache [T, 1, 96, 96] tensors on disk.
    arr = np.load(path)
    t = torch.from_numpy(arr).float()
    if t.ndim == 3:
        t = t.unsqueeze(1)  # add channel dim
    return t / 255.0 if t.max() > 1.5 else t


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--manifest", required=True)
    p.add_argument("--utt", required=True)
    p.add_argument("--pool", default=str(ROOT / "checkpoints/identity_pool.pt"))
    args = p.parse_args()

    pipe = AVSDGERPipeline.from_config(args.config)

    # Try to load an enrolled pool; if absent, enrol from the manifest on the fly.
    if Path(args.pool).exists():
        pipe.load_pool(args.pool)
        print(f"[pool] loaded from {args.pool} — {len(pipe.pool)} speakers")
    else:
        print("[pool] no pool file; enrolling on the fly from manifest")
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for spk in manifest["speakers"]:
            pipe.enroll(
                spk["speaker_id"],
                _load_audio(spk.get("enrollment_audio")),
                _load_face_stub(spk.get("enrollment_face")),
                meta=spk.get("meta"),
            )

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    utt = next(u for u in manifest["utterances"] if u["utt_id"] == args.utt)

    audio = _load_audio(utt.get("audio"))
    video = _load_video_mouth_roi(utt.get("mouth_roi"))

    out = pipe.run(audio, video)
    print("\n=== AVSD-GER output ===")
    print(f"text        : {out['text']}")
    print(f"speaker_id  : {out['speaker_id']}")
    print(f"confidence  : {out['confidence']:.3f}")
    print(f"iterations  : {out['iterations']}")
    print("trace       :")
    pprint(out["trace"], width=100)
    return 0


def _load_face_stub(path: str | None):
    if path is None or not Path(path).exists():
        return (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    import cv2
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    raise SystemExit(main())
