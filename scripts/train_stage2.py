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


def _format_nbest(items: list[str] | tuple[str, ...] | None) -> str:
    if not items:
        return ""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item).strip().split())
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return " | ".join(out)


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
    debug_loss_every: int = 0,
    fail_on_nonfinite: bool = True,
    grad_clip_norm: float = 1.0,
    warmup: str = "joint",
    aligner_checkpoint: str | Path | None = None,
    ctc_checkpoint: str | Path | None = None,
    ger_projectors_checkpoint: str | Path | None = None,
    no_encoder_context: bool = False,
) -> None:
    if wb is None:
        wb = WandbLogger(None)
    seed_all(int(cfg.get("seed", 1337)))
    device = resolve_device(cfg.get("device", "cpu"))
    stub = bool(cfg.get("stub_backbones", True))

    warmup = warmup.lower()
    if warmup not in {"joint", "align_ctc", "ger_lora", "ger_qformer"}:
        raise ValueError(f"Unsupported Stage-2 warmup mode: {warmup!r}")
    if no_encoder_context:
        cfg.setdefault("asr", {})["expose_encoder"] = False
        if warmup in {"joint", "align_ctc", "ger_qformer"}:
            raise ValueError(
                "--no-encoder-context disables Whisper/AV-HuBERT encoder features, "
                f"so warmup={warmup!r} is invalid. Use --warmup ger_lora for "
                "text-only ASR n-best + VSR n-best GER training."
            )

    # Build only the modules needed for the requested warm-up. This matters on
    # 16 GB GPUs: freezing a module does not save VRAM if it was still loaded.
    asr = WhisperASR(cfg["asr"], stub=stub, device=device)
    needs_vsr = warmup in {"joint", "align_ctc", "ger_qformer"}
    needs_identity = warmup in {"joint", "align_ctc", "ger_qformer"}
    needs_ger = warmup in {"joint", "ger_lora", "ger_qformer"}
    ger_mode_cfg = str(cfg.get("ger", {}).get("mode", "audio_only")).lower()
    if no_encoder_context and needs_ger and ger_mode_cfg in {"av", "visual_only"}:
        needs_vsr = True
    if no_encoder_context and needs_ger:
        needs_identity = True
    vsr = AVHubertVSR(cfg["vsr"], stub=stub, device=device) if needs_vsr else None
    voice = VoiceEncoder(cfg["identity"]["voice_encoder"], stub=stub, device=device) if needs_identity else None
    face = FaceEncoder(cfg["identity"]["face_encoder"], stub=stub, device=device) if needs_identity else None
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
    ger = (
        GERHead(
            cfg["ger"], z_dim=cfg["identity"]["fused_dim"],
            d_align=cfg["alignment"]["d_model"], stub=stub, device=device,
        )
        if needs_ger
        else None
    )

    ctc = CTCHead(d_align=cfg["alignment"]["d_model"]).to(device)
    ger_ce = GERCrossEntropy(ger) if ger is not None else None
    info = BidirectionalInfoNCE(cfg["training"]["infonce"]).to(device)
    if aligner_checkpoint:
        p = Path(aligner_checkpoint)
        if p.exists():
            aligner.load_state_dict(torch.load(p, map_location=device))
            print(f"[stage2] loaded aligner checkpoint from {p}")
        else:
            raise FileNotFoundError(f"aligner checkpoint not found: {p}")
    if ctc_checkpoint:
        p = Path(ctc_checkpoint)
        if p.exists():
            ctc.load_state_dict(torch.load(p, map_location=device))
            print(f"[stage2] loaded CTC checkpoint from {p}")
        else:
            raise FileNotFoundError(f"CTC checkpoint not found: {p}")
    if ger is not None and ger_projectors_checkpoint:
        p = Path(ger_projectors_checkpoint)
        if p.exists():
            state = torch.load(p, map_location=device)
            ger.qformer.load_state_dict(state["qformer"])
            ger.id_proj.load_state_dict(state["id_proj"])
            print(f"[stage2] loaded GER projectors checkpoint from {p}")
        else:
            raise FileNotFoundError(f"GER projectors checkpoint not found: {p}")

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
    if warmup == "align_ctc":
        params += list(aligner.parameters()) + list(ctc.parameters())
    elif warmup == "ger_lora":
        if ger is None:
            raise RuntimeError("ger_lora warm-up requires the GER head.")
        if not stub and ger._llm is not None:
            params += [p for p in ger._llm.parameters() if p.requires_grad]
        # No AV soft-prefix training in this phase; keep projectors frozen.
        for p in ger.qformer.parameters():
            p.requires_grad_(False)
        for p in ger.id_proj.parameters():
            p.requires_grad_(False)
    elif warmup == "ger_qformer":
        if ger is None:
            raise RuntimeError("ger_qformer warm-up requires the GER head.")
        if not stub and ger._llm is not None:
            params += [p for p in ger._llm.parameters() if p.requires_grad]
        params += list(ger.qformer.parameters()) + list(ger.id_proj.parameters())
    else:
        for m in (pool.fuser, aligner, ctc):
            params += list(m.parameters())
        # GER LoRA params only (base LLM stays frozen inside peft model)
        if not stub and ger is not None and ger._llm is not None:
            params += [p for p in ger._llm.parameters() if p.requires_grad]
        if ger is not None:
            params += list(ger.qformer.parameters()) + list(ger.id_proj.parameters())
    if not params:
        raise RuntimeError(f"No trainable parameters selected for warmup={warmup!r}")
    optim = torch.optim.AdamW(params, lr=stage2_lr_cfg)

    w_ctc = 1.0 if warmup in {"joint", "align_ctc"} else 0.0
    w_ger = 1.0 if warmup in {"joint", "ger_lora", "ger_qformer"} else 0.0
    w_info = 0.5 if warmup == "joint" else 0.0
    ger_mode = str(cfg.get("ger", {}).get("mode", "audio_only")).lower()
    if warmup == "ger_lora" and not no_encoder_context:
        ger_mode = "audio_only"
    use_av_context = ger_mode in {"av", "visual_only"}
    if no_encoder_context:
        use_av_context = False

    n_epochs = int(cfg["training"]["stage2"]["epochs"])
    records = list(iter_manifest(manifest)) if Path(manifest).exists() else [None] * 8

    step = 0
    for epoch in range(n_epochs):
        running = {"ctc": 0.0, "ger": 0.0, "info": 0.0, "n": 0}
        for rec in records:
            batch = _stub_batch(cfg, device) if (rec is None or stub) else _load_record(rec)

            # ---- forward full pipeline ----------------------------------
            asr_out = asr.transcribe(batch["audio"])
            vsr_out = (
                vsr.extract(batch["video"])
                if vsr is not None
                else {"vsr_features": torch.empty(0, AVHubertVSR.FEATURE_DIM, device=device), "lip_hyp": "", "lip_nbest": []}
            )
            if no_encoder_context:
                asr_tok = torch.empty(0, WhisperASR.ENCODER_DIM, device=device)
            else:
                asr_feats = (asr_out.encoder_features if asr_out.encoder_features is not None
                             else torch.randn(150, WhisperASR.ENCODER_DIM, device=device))
                asr_tok = pool_encoder_to_tokens(asr_feats.to(device), asr_out.words, asr_out.frame_rate_hz)

            if voice is not None and face is not None:
                v_emb = voice.embed(batch["audio"])
                f_emb = face.embed(batch["face"])
                id_q = pool.query(v_emb, f_emb)
                z_id = id_q.z_id
                if len(pool) == 0:
                    z_id = pool.fuser(v_emb.unsqueeze(0), f_emb.unsqueeze(0)).squeeze(0)
            else:
                v_emb = torch.zeros(cfg["identity"]["voice_dim"], device=device)
                f_emb = torch.zeros(cfg["identity"]["face_dim"], device=device)
                z_id = torch.zeros(cfg["identity"]["fused_dim"], device=device)
            if warmup == "ger_lora":
                z_id = z_id.detach()

            if warmup == "ger_lora":
                f_align = torch.empty(0, cfg["alignment"]["d_model"], device=device)
            else:
                f_align = aligner(
                    asr_tok_feats=asr_tok,
                    vsr_feats=vsr_out["vsr_features"].to(device),
                    e_id=z_id,
                )

            # ---- losses --------------------------------------------------
            if w_ctc:
                ctc_report = ctc(f_align, targets=[batch["target"]])
                l_ctc = ctc_report.loss
            else:
                ctc_report = None
                l_ctc = torch.zeros((), device=device)
            if w_ger:
                if ger_ce is None:
                    raise RuntimeError("GER loss requested but GER head was not loaded.")
                lip_hyp = _format_nbest(vsr_out.get("lip_nbest")) or str(vsr_out.get("lip_hyp", ""))
                ger_report = ger_ce(
                    z_id=z_id, f_align=f_align,
                    nbest=asr_out.nbest, lip_hyp=lip_hyp,
                    target=batch["target"],
                    speaker_id=batch.get("speaker_id"),
                    mode=ger_mode,
                    use_av_context=use_av_context and not no_encoder_context,
                )
                l_ger = ger_report.loss
            else:
                ger_report = None
                l_ger = torch.zeros((), device=device)
            # Bidirectional InfoNCE on a micro-batch of 2 pairs (self + swap)
            neg_audio = batch.get("neg_audio")
            neg_face = batch.get("neg_face")
            if w_info:
                voice_pair = (
                    voice.embed(neg_audio)
                    if neg_audio is not None and voice is not None
                    else batch["voice_pair"].to(device)
                )
                face_pair = (
                    face.embed(neg_face)
                    if neg_face is not None and face is not None
                    else batch["face_pair"].to(device)
                )
            if w_info:
                a = pool.fuser.voice_proj(torch.stack([v_emb, voice_pair]))
                v = pool.fuser.face_proj(torch.stack([f_emb, face_pair]))
                l_info = info(a, v).loss
            else:
                l_info = torch.zeros((), device=device)

            loss = w_ctc * l_ctc + w_ger * l_ger + w_info * l_info
            loss_parts = {
                "ctc": l_ctc,
                "ger": l_ger,
                "info": l_info,
                "total": loss,
            }
            finite_parts = {name: bool(torch.isfinite(val.detach()).item()) for name, val in loss_parts.items()}
            target_chars = len(ctc.vocab.encode(batch["target"]))
            ctc_input_len = int(ctc_report.log_probs.shape[1]) if ctc_report is not None else 0
            should_debug = debug_loss_every > 0 and (step == 0 or (step + 1) % debug_loss_every == 0)
            if should_debug:
                loss_text = " ".join(
                    f"{name}={float(val.detach().item()):.6g}/finite={finite_parts[name]}"
                    for name, val in loss_parts.items()
                )
                print(
                    f"[stage2-debug] step={step + 1} epoch={epoch + 1} "
                    f"ctc_input_len={ctc_input_len} ctc_target_chars={target_chars} "
                    f"ger_target_tokens={ger_report.n_target_tokens if ger_report is not None else 0} "
                    f"{loss_text}"
                )
            if fail_on_nonfinite and not all(finite_parts.values()):
                raise FloatingPointError(
                    f"Non-finite Stage-2 loss at step={step + 1}: "
                    + ", ".join(f"{name}={float(val.detach().item())}" for name, val in loss_parts.items())
                )
            optim.zero_grad()
            loss.backward()
            if should_debug:
                grad_rows = []
                grad_sources = [("pool.fuser", pool.fuser), ("aligner", aligner), ("ctc", ctc)]
                if ger is not None:
                    grad_sources.extend([("ger.qformer", ger.qformer), ("ger.id_proj", ger.id_proj)])
                    if not stub and ger._llm is not None:
                        grad_sources.append(("ger.lora", ger._llm))
                for prefix, module in grad_sources:
                    norms = [
                        float(p.grad.detach().norm().item())
                        for p in module.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    grad_rows.append(f"{prefix}:n={len(norms)},mean={sum(norms) / max(1, len(norms)):.3g}")
                print("[stage2-debug] grad " + " | ".join(grad_rows))
            if grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                if should_debug:
                    print(
                        f"[stage2-debug] grad_clip total_norm={float(total_grad_norm):.6g} "
                        f"max_norm={grad_clip_norm:.6g}"
                    )
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
    if ger is not None:
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


def _resolve_manifest_arg(manifest: str, manifest_dir: str | None) -> str:
    if manifest_dir:
        directory = Path(manifest_dir)
        sibling_jsonl = directory.with_suffix(".jsonl")
        if sibling_jsonl.exists():
            print(f"[train_stage2] Resolved --manifest-dir {directory} -> {sibling_jsonl}")
            return str(sibling_jsonl)
        raise FileNotFoundError(
            f"--manifest-dir was provided, but converted JSONL was not found: {sibling_jsonl}. "
            "Create it with scripts/ami_visual_to_jsonl.py --manifest-dir <dir> --out <dir>.jsonl"
        )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 multi-task trainer (spec Section 7).")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default="")
    ap.add_argument(
        "--manifest-dir",
        default=None,
        help=(
            "Directory of AMI visual per-meeting manifests. The trainer expects "
            "a JSONL file, so this resolves <dir>.jsonl when present."
        ),
    )
    ap.add_argument("--out", default="checkpoints/stage2/")
    ap.add_argument(
        "--warmup",
        default="joint",
        choices=["joint", "align_ctc", "ger_lora", "ger_qformer"],
        help=(
            "Stage-2 training slice. align_ctc skips the LLM and trains only "
            "ID-conditioned aligner + CTC; ger_lora trains audio-only LoRA; "
            "ger_qformer trains LoRA + QFormer/id_proj with AV context; joint "
            "runs the original multi-task Stage-2 objective."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.stage2.epochs without editing the YAML config.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override training.stage2.lr without editing the YAML config.",
    )
    ap.add_argument(
        "--stage1-pool",
        default=None,
        help="Stage-1 identity_pool checkpoint to initialize the Stage-2 fuser.",
    )
    ap.add_argument(
        "--aligner-checkpoint",
        default=None,
        help="Optional aligner state_dict to initialize this warm-up/joint run.",
    )
    ap.add_argument(
        "--ctc-checkpoint",
        default=None,
        help="Optional CTC head state_dict to initialize this warm-up/joint run.",
    )
    ap.add_argument(
        "--ger-projectors-checkpoint",
        default=None,
        help="Optional ger_projectors.pt with qformer/id_proj state_dicts.",
    )
    ap.add_argument(
        "--ger-mode",
        default=None,
        choices=["audio_only", "av", "visual_only"],
        help="Override cfg.ger.mode for Stage-2 training.",
    )
    ap.add_argument(
        "--asr-backend",
        default=None,
        choices=["faster-whisper", "openai-whisper"],
        help=(
            "Override cfg.asr.backend. Use openai-whisper for ASR text n-best "
            "and Whisper-native rescoring."
        ),
    )
    ap.add_argument(
        "--no-encoder-context",
        action="store_true",
        help=(
            "Do not compute/pass Whisper encoder or AV-HuBERT feature context into GER. "
            "Use text n-best only: ASR n-best + VSR lip n-best. Valid with --warmup ger_lora."
        ),
    )
    ap.add_argument(
        "--llm-quant", default=None,
        choices=["auto", "fp16", "bf16", "int8", "4bit"],
        help="Override Llama-3 weight precision. auto = pick from GPU VRAM. "
             "Default: read from configs/default.yaml (ger.llm_quant).",
    )
    ap.add_argument(
        "--debug-loss-every",
        type=int,
        default=0,
        help="Print Stage-2 loss, CTC length, GER token, and gradient diagnostics every N steps.",
    )
    ap.add_argument(
        "--no-fail-on-nonfinite",
        action="store_true",
        help="Do not raise immediately when any Stage-2 loss becomes NaN or Inf.",
    )
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global gradient norm clipping before optimizer.step(); set <=0 to disable.",
    )
    add_wandb_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.manifest = _resolve_manifest_arg(args.manifest, args.manifest_dir)
    if args.stage1_pool is not None:
        cfg.setdefault("training", {}).setdefault("stage2", {})["stage1_pool"] = args.stage1_pool
        print(f"[train_stage2] Override stage1_pool -> {args.stage1_pool}")
    if args.epochs is not None:
        cfg.setdefault("training", {}).setdefault("stage2", {})["epochs"] = args.epochs
        print(f"[train_stage2] Override stage2.epochs -> {args.epochs}")
    if args.lr is not None:
        cfg.setdefault("training", {}).setdefault("stage2", {})["lr"] = args.lr
        stage1_lr = float(cfg.setdefault("training", {}).setdefault("stage1", {}).get("lr", 0.001))
        cfg["training"]["stage2"]["lr_ratio_to_stage1"] = float(args.lr) / stage1_lr
        print(f"[train_stage2] Override stage2.lr -> {args.lr}")
    if args.ger_mode is not None:
        cfg.setdefault("ger", {})["mode"] = args.ger_mode
        print(f"[train_stage2] Override ger.mode -> {args.ger_mode}")
    if args.asr_backend is not None:
        cfg.setdefault("asr", {})["backend"] = args.asr_backend
        print(f"[train_stage2] Override asr.backend -> {args.asr_backend}")
    if args.no_encoder_context:
        cfg.setdefault("asr", {})["expose_encoder"] = False
        print("[train_stage2] no_encoder_context -> ASR/VSR text n-best only; no encoder soft context")
    if args.llm_quant is not None:
        cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant
        print(f"[train_stage2] Override llm_quant -> {args.llm_quant}")
    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"stage2-{args.warmup}-{Path(args.manifest).stem or 'stub'}",
        job_type="train-stage2",
        config={
            "stage": "stage2",
            "stage2_warmup": args.warmup,
            "config_path": args.config,
            "manifest": args.manifest,
            **cfg,
        },
    )
    try:
        train(
            cfg,
            args.manifest,
            args.out,
            wb=wb,
            debug_loss_every=args.debug_loss_every,
            fail_on_nonfinite=not args.no_fail_on_nonfinite,
            grad_clip_norm=args.grad_clip,
            warmup=args.warmup,
            aligner_checkpoint=args.aligner_checkpoint,
            ctc_checkpoint=args.ctc_checkpoint,
            ger_projectors_checkpoint=args.ger_projectors_checkpoint,
            no_encoder_context=args.no_encoder_context,
        )
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
