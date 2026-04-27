"""Enroll speakers into the Identity Pool from a manifest.

Manifest format (see data/sample_manifest.json):

    {
      "speakers": [
        {
          "speaker_id": "spk_01",
          "enrollment_audio": "data/spk_01/enroll.wav",
          "enrollment_face":  "data/spk_01/enroll.jpg",
          "meta": {"name": "Alice"}
        }
      ],
      "utterances": [ ... ]
    }

Runs on whatever `device` + `stub_backbones` the config specifies — with
`stub_backbones: true` (default), this script exercises the pool code path
on random embeddings so you can verify the wiring without real data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.pipeline import AVSDGERPipeline  # noqa: E402
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args  # noqa: E402


def _load_audio(path: str | None) -> np.ndarray | torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.randn(16000 * 3)  # 3s dummy
    import soundfile as sf
    wav, sr = sf.read(path)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _load_face(path: str | None) -> np.ndarray:
    if path is None or not Path(path).exists():
        return (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    import cv2
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--in-pool", default=None,
        help="Path to an existing pool file to LOAD before enrolling. "
             "Use this after Stage-2 to re-enrol with the trained fuser "
             "weights instead of the freshly-initialised ones."
    )
    p.add_argument("--out-pool", default=str(ROOT / "checkpoints/identity_pool.pt"))
    p.add_argument(
        "--llm-quant", default=None,
        choices=["auto", "fp16", "int8", "4bit"],
        help="Override Llama-3 weight precision. auto = pick from GPU VRAM. "
             "Default: read from configs/default.yaml (ger.llm_quant)."
    )
    add_wandb_args(p)
    args = p.parse_args()

    # Apply --llm-quant override before constructing the pipeline.
    if args.llm_quant is not None:
        from avsd_ger.utils import load_config
        cfg = load_config(args.config)
        cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant
        pipe = AVSDGERPipeline(cfg)
    else:
        pipe = AVSDGERPipeline.from_config(args.config)

    # W&B: log enrolment metadata so different smoke runs are sortable.
    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"enroll-{Path(args.manifest).stem}",
        job_type="enroll",
        config={
            "config_path": args.config,
            "manifest": args.manifest,
            "in_pool": args.in_pool,
            "out_pool": args.out_pool,
            "llm_quant": args.llm_quant,
        },
    )

    try:
        # Load existing pool first (preserves trained fuser weights + any prior
        # enrollments) before adding new speakers from the manifest.
        in_pool = getattr(args, "in_pool", None)
        if in_pool and Path(in_pool).exists():
            pipe.load_pool(in_pool)
            print(f"[in-pool] loaded {in_pool} -- start size = {len(pipe.pool)}")
            wb.log({"enroll/start_pool_size": len(pipe.pool)})

        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for spk in manifest["speakers"]:
            audio = _load_audio(spk.get("enrollment_audio"))
            face = _load_face(spk.get("enrollment_face"))
            pipe.enroll(spk["speaker_id"], audio, face, meta=spk.get("meta"))
            print(f"[enroll] {spk['speaker_id']}  pool size = {len(pipe.pool)}")

        pipe.save_pool(args.out_pool)
        print(f"[save]  identity pool -> {args.out_pool}")

        wb.summary({
            "summary/n_speakers_enrolled": len(pipe.pool),
            "summary/out_pool": args.out_pool,
        })
    finally:
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
