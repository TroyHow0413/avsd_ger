"""Cached Stage-2 trainer for a single high-memory GPU.

This entry point preserves the Stage-2 objectives and hyperparameters from
``train_stage2.py`` while moving frozen-backbone inference out of the epoch
loop.  Features are written in shards and reused by warm-up and joint runs.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import torch


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
from scripts import train_stage2 as base


CACHE_VERSION = 1


def _cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(device="cpu").contiguous()


def _cache_signature(cfg: dict[str, Any], manifest: Path) -> str:
    manifest_stat = manifest.stat() if manifest.exists() else None
    payload = {
        "version": CACHE_VERSION,
        "manifest": str(manifest.resolve()) if manifest.exists() else str(manifest),
        "manifest_size": manifest_stat.st_size if manifest_stat else None,
        "manifest_mtime_ns": manifest_stat.st_mtime_ns if manifest_stat else None,
        "stub_backbones": bool(cfg.get("stub_backbones", True)),
        "asr": cfg.get("asr", {}),
        "vsr": cfg.get("vsr", {}),
        "identity_encoders": {
            "voice": cfg.get("identity", {}).get("voice_encoder"),
            "face": cfg.get("identity", {}).get("face_encoder"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_source_records(manifest: Path, stub: bool) -> list[dict[str, Any] | None]:
    if manifest.exists():
        return list(base.iter_manifest(manifest))
    if stub:
        return [None] * 8
    raise FileNotFoundError(f"Stage-2 manifest not found: {manifest}")


def _extract_cached_record(
    cfg: dict[str, Any],
    rec: dict[str, Any] | None,
    *,
    stub: bool,
    device: torch.device,
    asr: WhisperASR,
    vsr: AVHubertVSR,
    voice: VoiceEncoder,
    face: FaceEncoder,
) -> dict[str, Any]:
    batch = base._stub_batch(cfg, device) if (rec is None or stub) else base._load_record(rec)
    with torch.inference_mode():
        asr_out = asr.transcribe(batch["audio"])
        vsr_out = vsr.extract(batch["video"])
        if asr_out.encoder_features is None:
            raise RuntimeError("Feature caching requires cfg.asr.expose_encoder=true.")
        asr_tok = pool_encoder_to_tokens(
            asr_out.encoder_features.to(device),
            asr_out.words,
            asr_out.frame_rate_hz,
        )
        voice_emb = voice.embed(batch["audio"])
        face_emb = face.embed(batch["face"])
        neg_voice_emb = voice.embed(batch["neg_audio"]) if batch.get("neg_audio") is not None else None
        neg_face_emb = face.embed(batch["neg_face"]) if batch.get("neg_face") is not None else None

    return {
        "asr_tok": _cpu_tensor(asr_tok),
        "asr_nbest": list(asr_out.nbest),
        "vsr_features": _cpu_tensor(vsr_out["vsr_features"]),
        "lip_hyp": str(vsr_out.get("lip_hyp", "")),
        "lip_nbest": list(vsr_out.get("lip_nbest") or []),
        "voice_emb": _cpu_tensor(voice_emb),
        "face_emb": _cpu_tensor(face_emb),
        "neg_voice_emb": _cpu_tensor(neg_voice_emb),
        "neg_face_emb": _cpu_tensor(neg_face_emb),
        "target": str(batch["target"]),
        "speaker_id": batch.get("speaker_id"),
    }


def build_feature_cache(
    cfg: dict[str, Any],
    manifest: str | Path,
    cache_dir: str | Path,
    *,
    shard_size: int,
    rebuild: bool,
) -> Path:
    manifest_path = Path(manifest)
    cache_path = Path(cache_dir)
    index_path = cache_path / "index.json"
    signature = _cache_signature(cfg, manifest_path)

    if index_path.exists() and not rebuild:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("signature") != signature:
            raise RuntimeError(
                f"Feature cache does not match the current manifest/config: {cache_path}. "
                "Use --rebuild-cache or choose a new --cache-dir."
            )
        missing = [name for name in index.get("shards", []) if not (cache_path / name).exists()]
        if missing:
            raise FileNotFoundError(f"Feature cache is incomplete; missing shards: {missing[:3]}")
        print(f"[pro6000-cache] Reusing {index.get('records', 0)} records from {cache_path}")
        return index_path

    if rebuild and cache_path.exists():
        for old_shard in cache_path.glob("shard-*.pt"):
            old_shard.unlink()
        if index_path.exists():
            index_path.unlink()
    cache_path.mkdir(parents=True, exist_ok=True)

    seed_all(int(cfg.get("seed", 1337)))
    device = resolve_device(cfg.get("device", "cpu"))
    stub = bool(cfg.get("stub_backbones", True))
    records = _load_source_records(manifest_path, stub)
    print(f"[pro6000-cache] Extracting {len(records)} records on {device}")

    asr = WhisperASR(cfg["asr"], stub=stub, device=device)
    vsr = AVHubertVSR(cfg["vsr"], stub=stub, device=device)
    voice = VoiceEncoder(cfg["identity"]["voice_encoder"], stub=stub, device=device)
    face = FaceEncoder(cfg["identity"]["face_encoder"], stub=stub, device=device)

    shard: list[dict[str, Any]] = []
    shard_names: list[str] = []
    for idx, rec in enumerate(records):
        shard.append(
            _extract_cached_record(
                cfg,
                rec,
                stub=stub,
                device=device,
                asr=asr,
                vsr=vsr,
                voice=voice,
                face=face,
            )
        )
        if len(shard) >= shard_size or idx + 1 == len(records):
            name = f"shard-{len(shard_names):05d}.pt"
            torch.save(shard, cache_path / name)
            shard_names.append(name)
            print(f"[pro6000-cache] Wrote {name} ({len(shard)} records; {idx + 1}/{len(records)})")
            shard = []

    index = {
        "version": CACHE_VERSION,
        "signature": signature,
        "manifest": str(manifest_path),
        "records": len(records),
        "shard_size": shard_size,
        "shards": shard_names,
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    del asr, vsr, voice, face
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return index_path


def iter_cached_records(index_path: str | Path) -> Iterable[dict[str, Any]]:
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for name in index["shards"]:
        records = torch.load(index_path.parent / name, map_location="cpu")
        yield from records
        del records


def _load_checkpoint(module: torch.nn.Module, path: str | Path | None, label: str, device: torch.device) -> None:
    if not path:
        return
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    module.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"[pro6000] Loaded {label} checkpoint from {checkpoint}")


def train_cached(
    cfg: dict[str, Any],
    index_path: str | Path,
    out_dir: str | Path,
    *,
    wb: WandbLogger,
    warmup: str,
    aligner_checkpoint: str | Path | None,
    ctc_checkpoint: str | Path | None,
    ger_projectors_checkpoint: str | Path | None,
    debug_loss_every: int,
    fail_on_nonfinite: bool,
    grad_clip_norm: float,
) -> None:
    if warmup not in {"joint", "align_ctc", "ger_lora", "ger_qformer"}:
        raise ValueError(f"Unsupported Stage-2 warmup mode: {warmup!r}")

    seed_all(int(cfg.get("seed", 1337)))
    device = resolve_device(cfg.get("device", "cpu"))
    stub = bool(cfg.get("stub_backbones", True))
    pool = IdentityPool(cfg["identity"], device=device)
    stage1_pool = cfg.get("training", {}).get("stage2", {}).get("stage1_pool")
    if stage1_pool and Path(stage1_pool).exists():
        pool.load(stage1_pool)
        print(f"[pro6000] Loaded Stage-1 identity pool from {stage1_pool}")
    elif not stub:
        raise FileNotFoundError(f"Stage-1 identity pool not found: {stage1_pool}")

    aligner = IDConditionedAligner(
        cfg["alignment"],
        z_dim=cfg["identity"]["fused_dim"],
        d_asr=WhisperASR.ENCODER_DIM,
        d_vsr=AVHubertVSR.FEATURE_DIM,
    ).to(device)
    ctc = CTCHead(d_align=cfg["alignment"]["d_model"]).to(device)
    needs_ger = warmup in {"joint", "ger_lora", "ger_qformer"}
    ger = (
        GERHead(cfg["ger"], z_dim=cfg["identity"]["fused_dim"], d_align=cfg["alignment"]["d_model"], stub=stub, device=device)
        if needs_ger
        else None
    )
    _load_checkpoint(aligner, aligner_checkpoint, "aligner", device)
    _load_checkpoint(ctc, ctc_checkpoint, "CTC", device)
    if ger is not None and ger_projectors_checkpoint:
        state = torch.load(ger_projectors_checkpoint, map_location=device)
        ger.qformer.load_state_dict(state["qformer"])
        ger.id_proj.load_state_dict(state["id_proj"])

    params: list[torch.nn.Parameter] = []
    if warmup == "align_ctc":
        params += list(aligner.parameters()) + list(ctc.parameters())
    elif warmup == "ger_lora":
        assert ger is not None
        if not stub and ger._llm is not None:
            params += [p for p in ger._llm.parameters() if p.requires_grad]
        for p in ger.qformer.parameters():
            p.requires_grad_(False)
        for p in ger.id_proj.parameters():
            p.requires_grad_(False)
    elif warmup == "ger_qformer":
        assert ger is not None
        if not stub and ger._llm is not None:
            params += [p for p in ger._llm.parameters() if p.requires_grad]
        params += list(ger.qformer.parameters()) + list(ger.id_proj.parameters())
    else:
        params += list(pool.fuser.parameters()) + list(aligner.parameters()) + list(ctc.parameters())
        assert ger is not None
        if not stub and ger._llm is not None:
            params += [p for p in ger._llm.parameters() if p.requires_grad]
        params += list(ger.qformer.parameters()) + list(ger.id_proj.parameters())
    if not params:
        raise RuntimeError(f"No trainable parameters selected for warmup={warmup!r}")

    stage1_lr = float(cfg["training"]["stage1"]["lr"])
    stage2_cfg = cfg["training"]["stage2"]
    lr = float(stage2_cfg["lr"])
    expected = stage1_lr * float(stage2_cfg.get("lr_ratio_to_stage1", 0.1))
    if abs(lr - expected) > 1e-9:
        raise ValueError(f"Stage-2 LR {lr} does not equal Stage-1 LR {stage1_lr} * ratio; expected {expected}")
    optim = torch.optim.AdamW(params, lr=lr)
    ger_ce = GERCrossEntropy(ger) if ger is not None else None
    info = BidirectionalInfoNCE(cfg["training"]["infonce"]).to(device)

    w_ctc = 1.0 if warmup in {"joint", "align_ctc"} else 0.0
    w_ger = 1.0 if warmup in {"joint", "ger_lora", "ger_qformer"} else 0.0
    w_info = 0.5 if warmup == "joint" else 0.0
    ger_mode = str(cfg.get("ger", {}).get("mode", "audio_only")).lower()
    if warmup == "ger_lora":
        ger_mode = "audio_only"
    use_av_context = ger_mode in {"av", "visual_only"}

    step = 0
    for epoch in range(int(stage2_cfg["epochs"])):
        running = {"ctc": 0.0, "ger": 0.0, "info": 0.0, "n": 0}
        for batch in iter_cached_records(index_path):
            asr_tok = batch["asr_tok"].to(device, non_blocking=True)
            vsr_features = batch["vsr_features"].to(device, non_blocking=True)
            voice_emb = batch["voice_emb"].to(device, non_blocking=True)
            face_emb = batch["face_emb"].to(device, non_blocking=True)

            if warmup == "ger_lora":
                # Match train_stage2.py: text-only LoRA warm-up does not load
                # the identity encoders and therefore uses a zero identity.
                z_id = torch.zeros(cfg["identity"]["fused_dim"], device=device)
            else:
                id_q = pool.query(voice_emb, face_emb)
                z_id = id_q.z_id
                if len(pool) == 0:
                    z_id = pool.fuser(voice_emb.unsqueeze(0), face_emb.unsqueeze(0)).squeeze(0)

            f_align = (
                torch.empty(0, cfg["alignment"]["d_model"], device=device)
                if warmup == "ger_lora"
                else aligner(asr_tok_feats=asr_tok, vsr_feats=vsr_features, e_id=z_id)
            )
            ctc_report = ctc(f_align, targets=[batch["target"]]) if w_ctc else None
            l_ctc = ctc_report.loss if ctc_report is not None else torch.zeros((), device=device)

            if w_ger:
                assert ger_ce is not None
                lip_hyp = base._format_nbest(batch.get("lip_nbest")) or batch.get("lip_hyp", "")
                ger_report = ger_ce(
                    z_id=z_id,
                    f_align=f_align,
                    nbest=batch["asr_nbest"],
                    lip_hyp=lip_hyp,
                    target=batch["target"],
                    speaker_id=batch.get("speaker_id"),
                    mode=ger_mode,
                    use_av_context=use_av_context,
                )
                l_ger = ger_report.loss
            else:
                ger_report = None
                l_ger = torch.zeros((), device=device)

            if w_info:
                neg_voice = batch.get("neg_voice_emb")
                neg_face = batch.get("neg_face_emb")
                voice_pair = neg_voice.to(device, non_blocking=True) if neg_voice is not None else torch.zeros_like(voice_emb)
                face_pair = neg_face.to(device, non_blocking=True) if neg_face is not None else torch.zeros_like(face_emb)
                projected_voice = pool.fuser.voice_proj(torch.stack([voice_emb, voice_pair]))
                projected_face = pool.fuser.face_proj(torch.stack([face_emb, face_pair]))
                l_info = info(projected_voice, projected_face).loss
            else:
                l_info = torch.zeros((), device=device)

            loss = w_ctc * l_ctc + w_ger * l_ger + w_info * l_info
            parts = {"ctc": l_ctc, "ger": l_ger, "info": l_info, "total": loss}
            finite = {name: bool(torch.isfinite(value.detach()).item()) for name, value in parts.items()}
            should_debug = debug_loss_every > 0 and (step == 0 or (step + 1) % debug_loss_every == 0)
            if should_debug:
                values = " ".join(f"{name}={float(value.detach()):.6g}" for name, value in parts.items())
                print(f"[pro6000-debug] step={step + 1} epoch={epoch + 1} {values}")
            if fail_on_nonfinite and not all(finite.values()):
                raise FloatingPointError(f"Non-finite Stage-2 loss at step={step + 1}: {finite}")

            optim.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optim.step()
            step += 1

            running["ctc"] += float(l_ctc.detach())
            running["ger"] += float(l_ger.detach())
            running["info"] += float(l_info.detach())
            running["n"] += 1
            wb.log(
                {
                    "stage2/loss/total": float(loss.detach()),
                    "stage2/loss/ctc": float(l_ctc.detach()),
                    "stage2/loss/ger": float(l_ger.detach()),
                    "stage2/loss/info": float(l_info.detach()),
                    "stage2/lr": lr,
                    "stage2/epoch": epoch,
                },
                step=step,
            )

        n = max(1, running["n"])
        print(
            f"[epoch {epoch + 1:02d}] ctc={running['ctc'] / n:.4f} "
            f"ger={running['ger'] / n:.4f} info={running['info'] / n:.4f}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool.save(out / "identity_pool_stage2.pt")
    torch.save(aligner.state_dict(), out / "aligner_stage2.pt")
    torch.save(ctc.state_dict(), out / "ctc_head_stage2.pt")
    if ger is not None:
        ger_dir = out / "ger"
        ger_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"qformer": ger.qformer.state_dict(), "id_proj": ger.id_proj.state_dict()}, ger_dir / "ger_projectors.pt")
        if not stub and ger._llm is not None:
            ger._llm.save_pretrained(ger_dir / "lora_adapter")
            if ger._tok is not None:
                ger._tok.save_pretrained(ger_dir / "tokenizer")
    print(f"[done] Saved cached Stage-2 checkpoints to {out}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pro6000 Stage-2 trainer with reusable frozen-feature shards.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="checkpoints/stage2_pro6000")
    ap.add_argument("--cache-dir", required=True, help="Reusable directory for frozen feature shards.")
    ap.add_argument("--cache-only", action="store_true", help="Build/validate the cache, then exit.")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--cache-shard-size", type=int, default=128)
    ap.add_argument("--warmup", choices=["joint", "align_ctc", "ger_lora", "ger_qformer"], default="joint")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--stage1-pool", default=None)
    ap.add_argument("--aligner-checkpoint", default=None)
    ap.add_argument("--ctc-checkpoint", default=None)
    ap.add_argument("--ger-projectors-checkpoint", default=None)
    ap.add_argument("--ger-mode", choices=["audio_only", "av", "visual_only"], default=None)
    ap.add_argument("--llm-name", default=None)
    ap.add_argument("--llm-quant", choices=["auto", "fp16", "bf16", "int8", "4bit"], default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--asr-backend", choices=["faster-whisper", "openai-whisper"], default=None)
    ap.add_argument("--asr-beam-size", type=int, default=None)
    ap.add_argument("--asr-n-best", type=int, default=None)
    ap.add_argument("--debug-loss-every", type=int, default=0)
    ap.add_argument("--no-fail-on-nonfinite", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    add_wandb_args(ap)
    return ap


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    stage2 = cfg.setdefault("training", {}).setdefault("stage2", {})
    if args.stage1_pool is not None:
        stage2["stage1_pool"] = args.stage1_pool
    if args.epochs is not None:
        stage2["epochs"] = args.epochs
    if args.lr is not None:
        stage2["lr"] = args.lr
        stage1_lr = float(cfg["training"].setdefault("stage1", {}).get("lr", 0.001))
        stage2["lr_ratio_to_stage1"] = float(args.lr) / stage1_lr
    for arg_name, section, key in (
        ("ger_mode", "ger", "mode"),
        ("llm_name", "ger", "llm_name"),
        ("llm_quant", "ger", "llm_quant"),
        ("max_new_tokens", "ger", "max_new_tokens"),
        ("asr_backend", "asr", "backend"),
        ("asr_beam_size", "asr", "beam_size"),
        ("asr_n_best", "asr", "n_best"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg.setdefault(section, {})[key] = value


def main() -> None:
    args = _build_parser().parse_args()
    if args.cache_shard_size <= 0:
        raise ValueError("--cache-shard-size must be positive")
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    index_path = build_feature_cache(
        cfg,
        args.manifest,
        args.cache_dir,
        shard_size=args.cache_shard_size,
        rebuild=args.rebuild_cache,
    )
    if args.cache_only:
        print(f"[done] Feature cache is ready: {index_path}")
        return

    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"stage2-pro6000-{args.warmup}-{Path(args.manifest).stem}",
        job_type="train-stage2",
        config={"stage": "stage2", "cache": str(index_path), "warmup": args.warmup, **cfg},
    )
    try:
        train_cached(
            cfg,
            index_path,
            args.out,
            wb=wb,
            warmup=args.warmup,
            aligner_checkpoint=args.aligner_checkpoint,
            ctc_checkpoint=args.ctc_checkpoint,
            ger_projectors_checkpoint=args.ger_projectors_checkpoint,
            debug_loss_every=args.debug_loss_every,
            fail_on_nonfinite=not args.no_fail_on_nonfinite,
            grad_clip_norm=args.grad_clip,
        )
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
