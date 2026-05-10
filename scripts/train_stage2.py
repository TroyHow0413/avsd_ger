"""Stage-2 multi-task trainer (spec Section 7).

Composes:
  L_total = w_ctc * L_CTC(f_align, y)
          + w_ger * L_GER_CE(prompt, y)
          + w_info * (L_{A->V} + L_{V->A})

Key invariants (spec Section 7, non-negotiable):
  * LR = 0.1x Stage 1 LR (exactly). Enforced here by pulling Stage 1 LR
    from the config and multiplying by ``lr_ratio_to_stage1``.
  * All params unfrozen.
  * Stop after ``stage2.epochs`` or when dev SA-WER plateaus (plateau
    detection is TODO; epochs cap is the current stop criterion).

This script is a training skeleton: it consumes a JSONL manifest and
synthesises random inputs in stub mode so the control flow is
verifiable without downloading weights. Real data ingestion is project-
specific (AMI loader, AISHELL-4 loader, ...).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from avsd_ger.backbones import AVHubertVSR, WhisperASR
from avsd_ger.c1_identity import FaceEncoder, IdentityPool, VoiceEncoder
from avsd_ger.c2_alignment import GERHead, IDConditionedAligner
from avsd_ger.training import BidirectionalInfoNCE
from avsd_ger.training.ctc_loss import CTCHead
from avsd_ger.training.ger_loss import GERCrossEntropy
from avsd_ger.utils import load_config, pool_encoder_to_tokens, resolve_device, seed_all
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args


def iter_manifest(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ------------------------------------------------------------------ stubs
def _stub_batch(cfg, device) -> dict[str, torch.Tensor]:
    """Fabricate a minimal batch for wiring verification."""
    return {
        "audio":        torch.randn(16000 * 3),
        "video":        torch.randn(75, 1, 96, 96),   # ~3s @ 25fps
        "face":         (np.random.rand(112, 112, 3) * 255).astype(np.uint8),
        "target":       "the quick brown fox jumps over the lazy dog",
        "voice_pair":   torch.randn(cfg["identity"]["voice_dim"]),
        "face_pair":    torch.randn(cfg["identity"]["face_dim"]),
    }


# ------------------------------------------------------------------ trainer
def train(
    cfg: dict[str, Any],
    manifest: str | Path,
    out_dir: str | Path,
    wb: "WandbLogger | None" = None,
) -> None:
    if wb is None:
        wb = WandbLogger(None)
    seed_all(int(cfg.get("seed", 1337)))
    device = resolve_device(cfg.get("device", "cpu"))
    stub = bool(cfg.get("stub_backbones", True))

    # Build full stack
    asr = WhisperASR(cfg["asr"], stub=stub, device=device)
    vsr = AVHubertVSR(cfg["vsr"], stub=stub, device=device)
    voice = VoiceEncoder(cfg["identity"]["voice_encoder"], stub=stub, device=device)
    face = FaceEncoder(cfg["identity"]["face_encoder"], stub=stub, device=device)
    pool = IdentityPool(cfg["identity"], device=device)
    stage1_pool_path = cfg.get("training", {}).get("stage2", {}).get(
        "stage1_pool", "checkpoints/stage1/identity_pool_stage1.pt"
    )
    if stage1_pool_path and Path(stage1_pool_path).exists():
        pool.load(stage1_pool_path)
        print(f"[stage2] loaded Stage-1 identity fuser from {stage1_pool_path}")
    elif not stub:
        print(f"[stage2] warning: Stage-1 identity fuser not found: {stage1_pool_path}")
    aligner = IDConditionedAligner(
        cfg["alignment"],
        z_dim=cfg["identity"]["fused_dim"],
        d_asr=WhisperASR.ENCODER_DIM,
        d_vsr=AVHubertVSR.FEATURE_DIM,
    ).to(device)
    ger = GERHead(
        cfg["ger"], z_dim=cfg["identity"]["fused_dim"],
        d_align=cfg["alignment"]["d_model"], stub=stub, device=device,
    )

    ctc = CTCHead(d_align=cfg["alignment"]["d_model"]).to(device)
    ger_ce = GERCrossEntropy(ger)
    info = BidirectionalInfoNCE(cfg["training"]["infonce"]).to(device)

    # LR: enforce 0.1x Stage 1 LR (spec Section 7)
    stage1_lr = float(cfg["training"]["stage1"]["lr"])
    ratio = float(cfg["training"]["stage2"].get("lr_ratio_to_stage1", 0.1))
    stage2_lr_cfg = float(cfg["training"]["stage2"]["lr"])
    expected = stage1_lr * ratio
    if abs(stage2_lr_cfg - expected) > 1e-9:
        raise ValueError(
            f"Stage 2 LR ({stage2_lr_cfg}) must equal Stage 1 LR ({stage1_lr}) * {ratio}; "
            f"got {stage2_lr_cfg} vs expected {expected}. Spec Section 7 forbids deviation."
        )

    params = []
    for m in (pool.fuser, aligner, ctc):
        params += list(m.parameters())
    # GER LoRA params only (base LLM stays frozen inside peft model)
    if not stub and ger._llm is not None:
        params += [p for p in ger._llm.parameters() if p.requires_grad]
    params += list(ger.qformer.parameters()) + list(ger.id_proj.parameters())
    optim = torch.optim.AdamW(params, lr=stage2_lr_cfg)

    w_ctc = 1.0
    w_ger = 1.0
    w_info = 0.5
    ger_mode = str(cfg.get("ger", {}).get("mode", "audio_only")).lower()
    use_av_context = ger_mode in {"av", "visual_only"}

    n_epochs = int(cfg["training"]["stage2"]["epochs"])
    records = list(iter_manifest(manifest)) if Path(manifest).exists() else [None] * 8

    step = 0
    for epoch in range(n_epochs):
        running = {"ctc": 0.0, "ger": 0.0, "info": 0.0, "n": 0}
        for rec in records:
            batch = _stub_batch(cfg, device) if (rec is None or stub) else _load_record(rec)

            # ---- forward full pipeline ----------------------------------
            asr_out = asr.transcribe(batch["audio"])
            vsr_out = vsr.extract(batch["video"])
            asr_feats = (asr_out.encoder_features if asr_out.encoder_features is not None
                         else torch.randn(150, WhisperASR.ENCODER_DIM, device=device))
            asr_tok = pool_encoder_to_tokens(asr_feats.to(device), asr_out.words, asr_out.frame_rate_hz)

            v_emb = voice.embed(batch["audio"])
            f_emb = face.embed(batch["face"])
            id_q = pool.query(v_emb, f_emb)
            z_id = id_q.z_id
            if len(pool) == 0:
                z_id = pool.fuser(v_emb.unsqueeze(0), f_emb.unsqueeze(0)).squeeze(0)

            f_align = aligner(
                asr_tok_feats=asr_tok,
                vsr_feats=vsr_out["vsr_features"].to(device),
                e_id=z_id,
            )

            # ---- losses --------------------------------------------------
            l_ctc = ctc(f_align, targets=[batch["target"]]).loss
            l_ger = ger_ce(
                z_id=z_id, f_align=f_align,
                nbest=asr_out.nbest, lip_hyp=vsr_out.get("lip_hyp", ""),
                target=batch["target"],
                speaker_id=batch.get("speaker_id"),
                mode=ger_mode,
                use_av_context=use_av_context,
            ).loss
            # Bidirectional InfoNCE on a micro-batch of 2 pairs (self + swap)
            neg_audio = batch.get("neg_audio")
            neg_face = batch.get("neg_face")
            voice_pair = (
                voice.embed(neg_audio)
                if neg_audio is not None
                else batch["voice_pair"].to(device)
            )
            face_pair = (
                face.embed(neg_face)
                if neg_face is not None
                else batch["face_pair"].to(device)
            )
            a = pool.fuser.voice_proj(torch.stack([v_emb, voice_pair]))
            v = pool.fuser.face_proj(torch.stack([f_emb, face_pair]))
            l_info = info(a, v).loss

            loss = w_ctc * l_ctc + w_ger * l_ger + w_info * l_info
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1

            running["ctc"] += float(l_ctc.item())
            running["ger"] += float(l_ger.item())
            running["info"] += float(l_info.item())
            running["n"] += 1

            wb.log({
                "stage2/loss/total": float(loss.item()),
                "stage2/loss/ctc":   float(l_ctc.item()),
                "stage2/loss/ger":   float(l_ger.item()),
                "stage2/loss/info":  float(l_info.item()),
                "stage2/lr":         float(optim.param_groups[0]["lr"]),
                "stage2/epoch":      int(epoch),
            }, step=step)

        n = max(1, running["n"])
        print(
            f"[epoch {epoch+1:02d}] ctc={running['ctc']/n:.4f} "
            f"ger={running['ger']/n:.4f} info={running['info']/n:.4f}"
        )
        wb.log({
            "stage2/epoch_end/ctc":  running["ctc"] / n,
            "stage2/epoch_end/ger":  running["ger"] / n,
            "stage2/epoch_end/info": running["info"] / n,
        }, step=step)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool.save(out / "identity_pool_stage2.pt")
    torch.save(aligner.state_dict(), out / "aligner_stage2.pt")
    torch.save(ctc.state_dict(), out / "ctc_head_stage2.pt")
    ger_dir = out / "ger"
    ger_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "qformer": ger.qformer.state_dict(),
            "id_proj": ger.id_proj.state_dict(),
        },
        ger_dir / "ger_projectors.pt",
    )
    if not stub and ger._llm is not None:
        ger._llm.save_pretrained(ger_dir / "lora_adapter")
        if ger._tok is not None:
            ger._tok.save_pretrained(ger_dir / "tokenizer")
    print(f"[done] saved Stage-2 checkpoints to {out}")


def _load_record(rec: dict[str, Any]) -> dict[str, Any]:
    audio = _load_wav_path(rec.get("wav_path") or rec.get("audio"))
    video = _load_video_path(rec.get("video_path") or rec.get("mouth_roi") or rec.get("video"))
    face = _load_face_path(rec.get("face_path") or rec.get("enrollment_face"))
    target = str(rec.get("target") or rec.get("ref_text") or "")
    if not target:
        raise ValueError(f"Missing target/ref_text for record {rec.get('utt_id', '<unknown>')}")

    out: dict[str, Any] = {
        "audio": audio,
        "video": video,
        "face": face,
        "target": target,
        "speaker_id": rec.get("speaker_id") or rec.get("ref_speaker"),
        "voice_pair": torch.zeros(192),
        "face_pair": torch.zeros(512),
    }
    if rec.get("neg_wav_path"):
        out["neg_audio"] = _load_wav_path(rec["neg_wav_path"])
    if rec.get("neg_face_path"):
        out["neg_face"] = _load_face_path(rec["neg_face_path"])
    return out


def _resolve_record_path(path: str | None, *, kind: str) -> Path:
    if not path:
        raise FileNotFoundError(f"Missing {kind} path in Stage-2 record.")
    raw = Path(path)
    candidates = [raw]
    if "\\" in path:
        candidates.append(Path(path.replace("\\", "/")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{kind} path does not exist: {path!r}")


def _load_wav_path(path: str | None) -> torch.Tensor:
    import soundfile as sf

    p = _resolve_record_path(path, kind="audio")
    wav, sr = sf.read(p)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return torch.from_numpy(np.asarray(wav, dtype=np.float32))


def _load_video_path(path: str | None) -> torch.Tensor:
    p = _resolve_record_path(path, kind="video/mouth_roi")
    arr = np.load(p)
    video = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if video.ndim == 3:
        video = video.unsqueeze(1)
    return video / 255.0 if video.numel() and float(video.max()) > 1.5 else video


def _load_face_path(path: str | None) -> np.ndarray:
    from PIL import Image

    p = _resolve_record_path(path, kind="face")
    return np.array(Image.open(p).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 multi-task trainer (spec Section 7).")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default="checkpoints/stage2/")
    ap.add_argument(
        "--stage1-pool",
        default=None,
        help="Stage-1 identity_pool checkpoint to initialize the Stage-2 fuser.",
    )
    ap.add_argument(
        "--ger-mode",
        default=None,
        choices=["audio_only", "av", "visual_only"],
        help="Override cfg.ger.mode for Stage-2 training.",
    )
    ap.add_argument(
        "--llm-quant", default=None,
        choices=["auto", "fp16", "int8", "4bit"],
        help="Override Llama-3 weight precision. auto = pick from GPU VRAM. "
             "Default: read from configs/default.yaml (ger.llm_quant).",
    )
    add_wandb_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage1_pool is not None:
        cfg.setdefault("training", {}).setdefault("stage2", {})["stage1_pool"] = args.stage1_pool
        print(f"[train_stage2] Override stage1_pool -> {args.stage1_pool}")
    if args.ger_mode is not None:
        cfg.setdefault("ger", {})["mode"] = args.ger_mode
        print(f"[train_stage2] Override ger.mode -> {args.ger_mode}")
    if args.llm_quant is not None:
        cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant
        print(f"[train_stage2] Override llm_quant -> {args.llm_quant}")
    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"stage2-{Path(args.manifest).stem or 'stub'}",
        job_type="train-stage2",
        config={"stage": "stage2", "config_path": args.config, "manifest": args.manifest, **cfg},
    )
    try:
        train(cfg, args.manifest, args.out, wb=wb)
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
