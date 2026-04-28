"""C1-only identity enrollment.

The normal scripts/enroll_identity.py constructs the full AVSDGERPipeline,
which also loads AV-HuBERT and Llama. For real-model smoke tests that is
unnecessary during enrollment: C1 only needs ECAPA, InsightFace, and the
IdentityPool. This script produces the same pool format without touching C2/C3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avsd_ger.c1_identity import FaceEncoder, IdentityPool, VoiceEncoder  # noqa: E402
from avsd_ger.utils import load_config, resolve_device, seed_all  # noqa: E402


def _load_audio(path: str | None) -> np.ndarray | torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.randn(16000 * 3)
    wav, sr = sf.read(path)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _load_face(path: str | None) -> np.ndarray:
    if path is None or not Path(path).exists():
        return (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img = cv2.imread(path)
    if img is None:
        return (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    p = argparse.ArgumentParser(description="Enroll speakers using only C1 modules.")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-pool", required=True)
    p.add_argument("--in-pool", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_all(int(cfg.get("seed", 1337)))
    device = resolve_device(cfg.get("device", "cpu"))
    stub = bool(cfg.get("stub_backbones", True))

    voice = VoiceEncoder(cfg["identity"]["voice_encoder"], stub=stub, device=device)
    face = FaceEncoder(cfg["identity"]["face_encoder"], stub=stub, device=device)
    pool = IdentityPool(cfg["identity"], device=device)
    if args.in_pool and Path(args.in_pool).exists():
        pool.load(args.in_pool)
        print(f"[in-pool] loaded {args.in_pool} -- start size = {len(pool)}")

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for spk in manifest.get("speakers", []):
        voice_emb = voice.embed(_load_audio(spk.get("enrollment_audio")))
        face_emb = face.embed(_load_face(spk.get("enrollment_face")))
        pool.enroll(spk["speaker_id"], voice_emb, face_emb, meta=spk.get("meta"))
        print(f"[enroll:C1] {spk['speaker_id']}  pool size = {len(pool)}")

    pool.save(args.out_pool)
    print(f"[save] identity pool -> {args.out_pool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
