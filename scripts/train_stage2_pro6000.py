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
import subprocess
import sys
from typing import Any, Iterable

import torch


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from avsd_ger.backbones import AVHubertVSR, WhisperASR
from avsd_ger.c1_identity import FaceEncoder, IdentityPool, VoiceEncoder
from avsd_ger.c2_alignment import GERHead, IDConditionedAligner
from avsd_ger.c2_alignment.model_backend import supported_model_families
from avsd_ger.training import BidirectionalInfoNCE
from avsd_ger.training.ctc_loss import CTCHead
from avsd_ger.training.ger_loss import GERCrossEntropy
from avsd_ger.training.quality import resample_quality_track, token_snr_scores
from avsd_ger.training.run_state import build_provenance, load_run_state, save_run_state
from avsd_ger.eval.metrics import compute_sa_wer
from avsd_ger.eval.session import SessionTurnResult
from avsd_ger.utils import load_config, pool_encoder_to_tokens, resolve_device, seed_all
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args
from scripts import train_stage2 as base


CACHE_VERSION = 3
QUALITY_SCHEMA_VERSION = 2

_FEATURE_SOURCE_FILES = (
    "scripts/train_stage2.py",
    "scripts/train_stage2_pro6000.py",
    "avsd_ger/training/quality.py",
    "avsd_ger/backbones/asr_whisper.py",
    "avsd_ger/backbones/vsr_avhubert.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _artifact_identity(raw: Any) -> dict[str, Any]:
    value = str(raw) if raw is not None else ""
    path = Path(value)
    if not path.is_absolute():
        path = _ROOT / path
    if path.is_file():
        return {
            "configured": value,
            "resolved": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return {"configured": value, "resolved": None, "size": None, "sha256": None}


def _adapter_identity(raw: str | Path) -> dict[str, Any]:
    directory = Path(raw)
    if not directory.is_dir():
        raise FileNotFoundError(f"GER adapter directory not found: {directory}")
    files = []
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        path = directory / name
        if path.is_file():
            files.append({
                "name": name,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
    if not files or not any(item["name"].startswith("adapter_model") for item in files):
        raise FileNotFoundError(f"Incomplete GER adapter directory: {directory}")
    return {"resolved": str(directory.resolve()), "files": files}


def _feature_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in _FEATURE_SOURCE_FILES:
        path = _ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii") if path.is_file() else b"missing")
    return digest.hexdigest()


def _cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(device="cpu").contiguous()


def _cache_signature(cfg: dict[str, Any], manifest: Path) -> str:
    manifest_digest = _sha256_file(manifest) if manifest.is_file() else None
    payload = {
        "version": CACHE_VERSION,
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "manifest": str(manifest.resolve()) if manifest.exists() else str(manifest),
        "manifest_sha256": manifest_digest,
        "git_commit": _git_commit(),
        "feature_source_fingerprint": _feature_source_fingerprint(),
        "stub_backbones": bool(cfg.get("stub_backbones", True)),
        "asr": cfg.get("asr", {}),
        "vsr": cfg.get("vsr", {}),
        "vsr_checkpoint": _artifact_identity(cfg.get("vsr", {}).get("checkpoint")),
        "identity_encoders": {
            "voice": cfg.get("identity", {}).get("voice_encoder"),
            "face": cfg.get("identity", {}).get("face_encoder"),
        },
        "quality": {
            "tau_a_snr_db": cfg.get("identity", {}).get("dual_gate", {}).get("tau_a_snr_db"),
            "snr_soft_scale_db": cfg.get("alignment", {}).get("snr_soft_scale_db", 4.0),
            "lip_resampling": "linear_align_corners_false",
            "snr_frame_hz": 100.0,
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

    cached = {
        "utt_id": str(batch.get("utt_id", "unknown")),
        "start": float(batch.get("start", 0.0)),
        "end": float(batch.get("end", 0.0)),
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
        "participant_id": batch.get("participant_id"),
        "dataset_build_id": batch.get("dataset_build_id", "unknown"),
        "lip_conf_v": _cpu_tensor(resample_quality_track(
            batch.get("lip_conf"), int(vsr_out["vsr_features"].shape[0])
        )),
        "lip_conf_source": str(batch.get("lip_conf_source", "missing")),
        "snr_per_tok": _cpu_tensor(token_snr_scores(
            batch["audio"],
            asr_out.words,
            int(asr_tok.shape[0]),
            tau_snr_db=float(cfg["identity"]["dual_gate"]["tau_a_snr_db"]),
            soft_scale_db=float(cfg["alignment"].get("snr_soft_scale_db", 4.0)),
        )),
        "speaker_mask_v": torch.ones(
            int(vsr_out["vsr_features"].shape[0]), dtype=torch.bool
        ),
    }
    _validate_cached_record(cached)
    return cached


def _validate_cached_record(record: dict[str, Any]) -> None:
    required = {
        "asr_tok", "asr_nbest", "vsr_features", "voice_emb", "face_emb",
        "target", "lip_conf_v", "lip_conf_source", "snr_per_tok",
        "speaker_mask_v", "dataset_build_id", "utt_id", "start", "end",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Cached feature record is missing fields: {missing}")
    asr = record["asr_tok"]
    vsr = record["vsr_features"]
    if not isinstance(asr, torch.Tensor) or asr.ndim != 2:
        raise ValueError("cached asr_tok must be a rank-2 tensor")
    if not isinstance(vsr, torch.Tensor) or vsr.ndim != 2:
        raise ValueError("cached vsr_features must be a rank-2 tensor")
    if int(record["snr_per_tok"].numel()) != int(asr.shape[0]):
        raise ValueError("cached snr_per_tok length does not match asr_tok")
    for key in ("lip_conf_v", "speaker_mask_v"):
        if int(record[key].numel()) != int(vsr.shape[0]):
            raise ValueError(f"cached {key} length does not match vsr_features")
    for key in ("asr_tok", "vsr_features", "voice_emb", "face_emb", "lip_conf_v", "snr_per_tok"):
        value = record[key]
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"cached {key} contains non-finite values")
    if not str(record["target"]).strip():
        raise ValueError("cached target is empty")


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
        if index.get("version") != CACHE_VERSION:
            raise RuntimeError(
                f"Feature cache schema version {index.get('version')} is not "
                f"the required version {CACHE_VERSION}; rebuild it."
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
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "dataset_build_ids": sorted({
            str(record.get("dataset_build_id", "unknown"))
            for record in records if record is not None
        }),
        "required_record_fields": sorted({
            "asr_tok", "asr_nbest", "vsr_features", "voice_emb", "face_emb",
            "target", "lip_conf_v", "lip_conf_source", "snr_per_tok",
            "speaker_mask_v", "dataset_build_id",
            "utt_id", "start", "end",
        }),
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
        for record in records:
            _validate_cached_record(record)
            yield record
        del records


def _load_checkpoint(module: torch.nn.Module, path: str | Path | None, label: str, device: torch.device) -> None:
    if not path:
        return
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    module.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"[pro6000] Loaded {label} checkpoint from {checkpoint}")


def _cache_provenance(index_path: str | Path) -> tuple[str, str, str]:
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    return str(index["manifest"]), str(index["signature"]), str(Path(index_path).resolve())


def _load_lora_adapter(model: torch.nn.Module, adapter_dir: Path, device: torch.device) -> None:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Resume LoRA adapter not found: {adapter_dir}")
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    weights = load_peft_weights(str(adapter_dir), device=str(device))
    result = set_peft_model_state_dict(model, weights, adapter_name="default")
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"Unexpected keys while restoring LoRA adapter: {unexpected[:5]}")


def _checkpoint_modules(
    warmup: str,
    pool: IdentityPool,
    aligner: IDConditionedAligner,
    ctc: CTCHead,
    ger: GERHead | None,
) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {}
    if warmup == "joint":
        modules["fuser"] = pool.fuser
    if warmup in {"joint", "align_ctc", "ger_qformer"}:
        modules["aligner"] = aligner
    if warmup in {"joint", "align_ctc"}:
        modules["ctc"] = ctc
    if ger is not None and warmup in {"joint", "ger_qformer"}:
        modules["ger_bridge"] = ger.bridge
    return modules


@torch.no_grad()
def _evaluate_cached(
    cfg: dict[str, Any],
    index_path: str | Path,
    *,
    warmup: str,
    pool: IdentityPool,
    aligner: IDConditionedAligner,
    ctc: CTCHead,
    ger: GERHead | None,
    ger_ce: GERCrossEntropy | None,
    device: torch.device,
    ger_mode: str,
    use_av_context: bool,
) -> dict[str, float | str]:
    modules = [pool.fuser, aligner, ctc]
    if ger is not None:
        modules.append(ger)
    previous = [module.training for module in modules]
    for module in modules:
        module.eval()

    ctc_total = 0.0
    ger_total = 0.0
    n = 0
    turns: list[SessionTurnResult] = []
    try:
        for batch in iter_cached_records(index_path):
            asr_tok = batch["asr_tok"].to(device, non_blocking=True)
            vsr_features = batch["vsr_features"].to(device, non_blocking=True)
            voice_emb = batch["voice_emb"].to(device, non_blocking=True)
            face_emb = batch["face_emb"].to(device, non_blocking=True)
            identity = pool.query(voice_emb, face_emb)
            if warmup == "ger_lora":
                z_id = torch.zeros(cfg["identity"]["fused_dim"], device=device)
                hyp_speaker = None
            else:
                z_id = identity.z_id
                if len(pool) == 0:
                    z_id = pool.fuser(voice_emb.unsqueeze(0), face_emb.unsqueeze(0)).squeeze(0)
                hyp_speaker = None
                if not identity.is_unknown and identity.top_ids:
                    hyp_speaker = str(identity.top_ids[0])
            f_align = (
                torch.empty(0, cfg["alignment"]["d_model"], device=device)
                if warmup == "ger_lora"
                else aligner(
                    asr_tok_feats=asr_tok,
                    vsr_feats=vsr_features,
                    e_id=z_id,
                    speaker_mask_v=batch["speaker_mask_v"].to(device),
                    snr_per_tok=batch["snr_per_tok"].to(device),
                    lip_conf_v=batch["lip_conf_v"].to(device),
                )
            )
            if warmup in {"joint", "align_ctc"}:
                ctc_total += float(ctc(f_align, targets=[batch["target"]]).loss)
            if ger is not None and ger_ce is not None:
                lip_hyp = base._format_nbest(batch.get("lip_nbest")) or batch.get("lip_hyp", "")
                ger_total += float(ger_ce(
                    z_id=z_id,
                    f_align=f_align,
                    nbest=batch["asr_nbest"],
                    lip_hyp=lip_hyp,
                    target=batch["target"],
                    speaker_id=batch.get("speaker_id"),
                    mode=ger_mode,
                    use_av_context=use_av_context,
                ).loss)
                generated = ger.generate(
                    z_id=z_id,
                    f_align=f_align,
                    nbest=batch["asr_nbest"],
                    lip_hyp=lip_hyp,
                    speaker_id=batch.get("speaker_id"),
                    mode=ger_mode,
                    use_av_context=use_av_context,
                )
                turns.append(SessionTurnResult(
                    turn_id=str(batch["utt_id"]),
                    start=float(batch["start"]),
                    end=float(batch["end"]),
                    hyp_text=str(generated["text"]),
                    hyp_speaker=hyp_speaker,
                    confidence=0.0,
                    s_acoustic=None,
                    iterations=1,
                    pool_updated=False,
                    ref_text=str(batch["target"]),
                    ref_speaker=batch.get("participant_id") or batch.get("speaker_id"),
                ))
            n += 1
    finally:
        for module, was_training in zip(modules, previous):
            module.train(was_training)

    if n == 0:
        raise RuntimeError("Dev feature cache contains no records")
    result: dict[str, float | str] = {
        "ctc_loss": ctc_total / n,
        "ger_loss": ger_total / n,
    }
    if turns:
        language = str(cfg.get("language") or cfg.get("asr", {}).get("language", "en"))
        sa_wer, details = compute_sa_wer(turns, language=language)
        result["sa_wer"] = float(sa_wer)
        result["wer"] = float(details["wer"])
    if warmup == "align_ctc":
        result["selection_name"] = "dev_ctc_loss"
        result["selection_value"] = float(result["ctc_loss"])
    elif warmup == "ger_lora":
        result["selection_name"] = "dev_wer"
        result["selection_value"] = float(result["wer"])
    else:
        result["selection_name"] = "dev_sa_wer"
        result["selection_value"] = float(result["sa_wer"])
    return result


def train_cached(
    cfg: dict[str, Any],
    index_path: str | Path,
    dev_index_path: str | Path,
    out_dir: str | Path,
    *,
    wb: WandbLogger,
    warmup: str,
    aligner_checkpoint: str | Path | None,
    ctc_checkpoint: str | Path | None,
    ger_projectors_checkpoint: str | Path | None,
    ger_adapter_checkpoint: str | Path | None,
    debug_loss_every: int,
    fail_on_nonfinite: bool,
    grad_clip_norm: float,
    resume: str | Path | None,
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
    ctc = CTCHead(
        d_align=cfg["alignment"]["d_model"],
        min_expansion=int(cfg["alignment"].get("ctc_min_expansion", 8)),
        max_expansion=int(cfg["alignment"].get("ctc_max_expansion", 32)),
    ).to(device)
    needs_ger = warmup in {"joint", "ger_lora", "ger_qformer"}
    ger = (
        GERHead(cfg["ger"], z_dim=cfg["identity"]["fused_dim"], d_align=cfg["alignment"]["d_model"], stub=stub, device=device)
        if needs_ger
        else None
    )
    _load_checkpoint(aligner, aligner_checkpoint, "aligner", device)
    _load_checkpoint(ctc, ctc_checkpoint, "CTC", device)
    if ger is not None and ger_projectors_checkpoint:
        ger.load_projector_checkpoint(ger_projectors_checkpoint, map_location=device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resume_path: Path | None = None
    if resume is not None:
        resume_path = Path(resume)
        resume_run_dir = resume_path if resume_path.is_dir() else resume_path.parent
        if resume_run_dir.resolve() != out.resolve():
            raise ValueError(
                "--resume must point to last.pt in the same --out directory; "
                "forking a run requires a new explicit initialization workflow"
            )
        if resume_path.is_dir():
            resume_path = resume_path / "last.pt"
    if ger_adapter_checkpoint is not None and resume is None:
        if ger is None or stub or ger._llm is None:
            raise ValueError("--ger-adapter-checkpoint requires a real GER-enabled warmup")
        _load_lora_adapter(ger._llm, Path(ger_adapter_checkpoint), device)
        print(f"[pro6000] Loaded initial GER LoRA adapter from {ger_adapter_checkpoint}")
    if resume is not None and ger is not None and not stub and ger._llm is not None:
        _load_lora_adapter(ger._llm, out / "ger" / "lora_adapter_last", device)

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

    train_manifest, train_signature, _ = _cache_provenance(index_path)
    dev_manifest, dev_signature, _ = _cache_provenance(dev_index_path)
    combined_signature = hashlib.sha256(
        f"train={train_signature};dev={dev_signature}".encode("utf-8")
    ).hexdigest()
    provenance = build_provenance(
        stage=f"stage2_{warmup}",
        cfg=cfg,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        cache_signature=combined_signature,
    )
    checkpoint_modules = _checkpoint_modules(warmup, pool, aligner, ctc, ger)
    step = 0
    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = -1
    if resume_path is not None:
        state = load_run_state(
            resume_path,
            expected_provenance=provenance,
            modules=checkpoint_modules,
            optimizer=optim,
            map_location=device,
        )
        start_epoch = int(state["epoch"]) + 1
        step = int(state["global_step"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        print(f"[resume] {resume_path}: next_epoch={start_epoch + 1} step={step}")

    for epoch in range(start_epoch, int(stage2_cfg["epochs"])):
        running = {"ctc": 0.0, "ger": 0.0, "info": 0.0, "n": 0, "ctc_zero": 0}
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
                else aligner(
                    asr_tok_feats=asr_tok,
                    vsr_feats=vsr_features,
                    e_id=z_id,
                    speaker_mask_v=batch["speaker_mask_v"].to(device),
                    snr_per_tok=batch["snr_per_tok"].to(device),
                    lip_conf_v=batch["lip_conf_v"].to(device),
                )
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
                ctc_diag = ""
                if ctc_report is not None:
                    ctc_diag = (
                        f" ctc_input={ctc_report.input_lengths.tolist()}"
                        f" ctc_target={ctc_report.target_lengths.tolist()}"
                        f" ctc_minimum={ctc_report.minimum_steps.tolist()}"
                        f" ctc_expansion={ctc_report.expansion}"
                        f" ctc_zero={ctc_report.zero_loss_count}"
                        f" ctc_nonfinite={ctc_report.nonfinite_count}"
                    )
                print(f"[pro6000-debug] step={step + 1} epoch={epoch + 1}{ctc_diag} {values}")
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
            running["ctc_zero"] += ctc_report.zero_loss_count if ctc_report is not None else 0
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
            f"ger={running['ger'] / n:.4f} info={running['info'] / n:.4f} "
            f"ctc_exact_zero={running['ctc_zero']}"
        )

        dev = _evaluate_cached(
            cfg,
            dev_index_path,
            warmup=warmup,
            pool=pool,
            aligner=aligner,
            ctc=ctc,
            ger=ger,
            ger_ce=ger_ce,
            device=device,
            ger_mode=ger_mode,
            use_av_context=use_av_context,
        )
        selection_value = float(dev["selection_value"])
        selection_metric = -selection_value
        improved = selection_metric > best_metric
        if improved:
            best_metric = selection_metric
            best_epoch = epoch
        print(
            f"[dev {epoch + 1:02d}] {dev['selection_name']}={selection_value:.4f} "
            f"best={-best_metric:.4f}@{best_epoch + 1}"
        )
        ger_dir = out / "ger"
        if ger is not None and not stub and ger._llm is not None:
            ger_dir.mkdir(parents=True, exist_ok=True)
            ger._llm.save_pretrained(ger_dir / "lora_adapter_last")
            if improved:
                ger._llm.save_pretrained(ger_dir / "lora_adapter_best")
        save_run_state(
            out / "last.pt",
            provenance=provenance,
            epoch=epoch,
            global_step=step,
            modules=checkpoint_modules,
            optimizer=optim,
            best_metric=best_metric,
            best_epoch=best_epoch,
            extra=dict(dev),
        )
        if improved:
            save_run_state(
                out / "best.pt",
                provenance=provenance,
                epoch=epoch,
                global_step=step,
                modules=checkpoint_modules,
                optimizer=optim,
                best_metric=best_metric,
                best_epoch=best_epoch,
                extra=dict(dev),
            )
        wb.log({
            "stage2/dev/ctc_loss": float(dev["ctc_loss"]),
            "stage2/dev/ger_loss": float(dev["ger_loss"]),
            "stage2/dev/selection_value": selection_value,
            "stage2/dev/is_best": int(improved),
            **({"stage2/dev/wer": float(dev["wer"]), "stage2/dev/sa_wer": float(dev["sa_wer"])} if "wer" in dev else {}),
        }, step=step)

    best_path = out / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Stage-2 produced no best checkpoint; increase epochs or check --resume")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    for name, module in checkpoint_modules.items():
        module.load_state_dict(best_state["modules"][name], strict=True)
    if ger is not None and not stub and ger._llm is not None:
        _load_lora_adapter(ger._llm, out / "ger" / "lora_adapter_best", device)
    print(f"[selection] restored best epoch={best_epoch + 1} metric={-best_metric:.4f}")
    pool.save(out / "identity_pool_stage2.pt")
    torch.save(aligner.state_dict(), out / "aligner_stage2.pt")
    torch.save(ctc.state_dict(), out / "ctc_head_stage2.pt")
    if ger is not None:
        ger_dir = out / "ger"
        ger_dir.mkdir(parents=True, exist_ok=True)
        ger.save_projector_checkpoint(ger_dir / "ger_projectors.pt")
        if not stub and ger._llm is not None:
            ger._llm.save_pretrained(ger_dir / "lora_adapter")
            if ger._tok is not None:
                ger._tok.save_pretrained(ger_dir / "tokenizer")
    print(f"[done] Saved cached Stage-2 checkpoints to {out}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pro6000 Stage-2 trainer with reusable frozen-feature shards.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dev-manifest", required=True, help="Validation JSONL used only for checkpoint selection.")
    ap.add_argument("--out", default="checkpoints/stage2_pro6000")
    ap.add_argument("--cache-dir", required=True, help="Reusable directory for frozen feature shards.")
    ap.add_argument("--dev-cache-dir", required=True, help="Reusable frozen-feature cache for --dev-manifest.")
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
    ap.add_argument(
        "--ger-adapter-checkpoint",
        default=None,
        help="Initialize a new GER-enabled stage from a prior PEFT adapter directory.",
    )
    ap.add_argument("--ger-mode", choices=["audio_only", "av", "visual_only"], default=None)
    ap.add_argument("--model-path", "--llm-name", dest="model_path", default=None)
    ap.add_argument("--model-family", choices=supported_model_families(), default=None)
    ap.add_argument("--ger-dtype", "--llm-quant", dest="ger_dtype", choices=["auto", "fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--asr-backend", choices=["faster-whisper", "openai-whisper"], default=None)
    ap.add_argument("--asr-beam-size", type=int, default=None)
    ap.add_argument("--asr-n-best", type=int, default=None)
    ap.add_argument("--debug-loss-every", type=int, default=0)
    ap.add_argument("--no-fail-on-nonfinite", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--resume", default=None, help="Path to last.pt or a run directory containing last.pt.")
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
    if args.ger_adapter_checkpoint is not None:
        stage2["initial_ger_adapter"] = _adapter_identity(
            args.ger_adapter_checkpoint
        )
    for arg_name, section, key in (
        ("ger_mode", "ger", "mode"),
        ("model_path", "ger", "model_path"),
        ("model_family", "ger", "model_family"),
        ("ger_dtype", "ger", "dtype"),
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
    dev_index_path = build_feature_cache(
        cfg,
        args.dev_manifest,
        args.dev_cache_dir,
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
            dev_index_path,
            args.out,
            wb=wb,
            warmup=args.warmup,
            aligner_checkpoint=args.aligner_checkpoint,
            ctc_checkpoint=args.ctc_checkpoint,
            ger_projectors_checkpoint=args.ger_projectors_checkpoint,
            ger_adapter_checkpoint=args.ger_adapter_checkpoint,
            debug_loss_every=args.debug_loss_every,
            fail_on_nonfinite=not args.no_fail_on_nonfinite,
            grad_clip_norm=args.grad_clip,
            resume=args.resume,
        )
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
