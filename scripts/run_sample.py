"""Run the full AVSD-GER pipeline on a single manifest utterance.

    python scripts/run_sample.py \
        --manifest data/sample_manifest.json \
        --utt utt_0001 \
        --llm-quant 4bit
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
from avsd_ger.utils import load_config         # noqa: E402
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args  # noqa: E402


def _load_audio(path: str | None) -> np.ndarray | torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.randn(16000 * 3)
    import soundfile as sf
    wav, sr = sf.read(path)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _load_video_mouth_roi(path: str | None, num_frames: int = 75) -> tuple[torch.Tensor | None, bool]:
    """Return a real [T, 1, 96, 96] mouth ROI plus a provenance flag."""
    if path is None or not Path(path).exists():
        return None, False
    # Real preprocessing pipeline lives in av_hubert/avhubert/preparation/align_mouth.py.
    # This caller expects pre-aligned [T, 1, 96, 96] tensors cached on disk.
    arr = np.load(path)
    t = torch.from_numpy(arr).float()
    if t.ndim == 3:
        t = t.unsqueeze(1)  # add channel dim
    return (t / 255.0 if t.max() > 1.5 else t), True


def _load_face(path: str | None) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    import cv2
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--manifest", required=True)
    p.add_argument("--utt", required=True)
    p.add_argument("--pool", default=str(ROOT / "checkpoints/identity_pool.pt"))
    p.add_argument(
        "--llm-quant", default=None,
        choices=["auto", "fp16", "int8", "4bit"],
        help="Override Llama-3 weight precision. auto = pick from GPU VRAM. "
             "Default: read from configs/default.yaml (ger.llm_quant).",
    )
    p.add_argument(
        "--ger-mode",
        default=None,
        choices=["audio_only", "av", "visual_only"],
        help="Override cfg.ger.mode for this sample run.",
    )
    add_wandb_args(p)
    args = p.parse_args()

    # Apply --llm-quant override before constructing the pipeline.
    # In-memory dict mutation -- matches enroll_identity.py style; avoids the
    # tempfile dance.
    if args.llm_quant is not None or args.ger_mode is not None:
        cfg = load_config(args.config)
        if args.llm_quant is not None:
            cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant
            print(f"[run_sample] Override llm_quant -> {args.llm_quant}")
        if args.ger_mode is not None:
            cfg.setdefault("ger", {})["mode"] = args.ger_mode
            print(f"[run_sample] Override ger.mode -> {args.ger_mode}")
        pipe = AVSDGERPipeline(cfg)
    else:
        pipe = AVSDGERPipeline.from_config(args.config)

    # W&B: log per-utterance smoke metadata + final trace.
    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"run-{Path(args.manifest).stem}-{args.utt}",
        job_type="run-sample",
        config={
            "config_path": args.config,
            "manifest": args.manifest,
            "utt": args.utt,
            "pool": args.pool,
            "llm_quant": args.llm_quant,
            "ger_mode": args.ger_mode,
        },
    )

    try:
        # Read the manifest ONCE, both for optional on-the-fly enrol and for the
        # utterance lookup below.
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if Path(args.pool).exists():
            pipe.load_pool(args.pool)
            print(f"[pool] loaded from {args.pool} -- {len(pipe.pool)} speakers")
        else:
            print("[pool] no pool file; enrolling on the fly from manifest")
            for spk in manifest["speakers"]:
                pipe.enroll(
                    spk["speaker_id"],
                    _load_audio(spk.get("enrollment_audio")),
                    _load_face(spk.get("enrollment_face")),
                    meta=spk.get("meta"),
                )

        utt = next(u for u in manifest["utterances"] if u["utt_id"] == args.utt)
        audio = _load_audio(utt.get("audio"))
        audio_len = int(audio.numel() if isinstance(audio, torch.Tensor) else len(audio))
        num_video_frames = max(1, round(audio_len / 16000 * 25))
        video, has_visual = _load_video_mouth_roi(
            utt.get("mouth_roi"), num_frames=num_video_frames
        )
        if pipe.ger_mode == "av" and not has_visual:
            print(
                "[warning] no valid mouth_roi for this utterance; "
                "pipeline will run audio_only. Do not report this sample as AV."
            )

        out = pipe.run(audio, video, has_visual=has_visual)

        print("\n=== AVSD-GER output ===")
        print(f"text        : {out['text']}")
        print(f"speaker_id  : {out['speaker_id']}")
        print(f"confidence  : {out['confidence']:.3f}")
        print(f"iterations  : {out['iterations']}")
        print("trace       :")
        pprint(out["trace"], width=100)

        # Pin the headline numbers + final text to W&B summary.
        last_decision = out["trace"][-1].get("decision") if out.get("trace") else None
        wb.summary({
            "summary/text":        out["text"],
            "summary/speaker_id":  out["speaker_id"] or "UNKNOWN",
            "summary/confidence":  float(out["confidence"]),
            "summary/iterations":  int(out["iterations"]),
            "summary/s_acoustic":  out.get("s_acoustic"),
            "summary/decision":    last_decision,
            "summary/pool_updated": bool(out.get("pool_updated", False)),
        })
    finally:
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
