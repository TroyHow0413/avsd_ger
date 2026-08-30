"""Stage-1 C1 identity training entry point (spec §7).

Workflow
--------
1. Iterate over a session corpus yielding (waveform, lip_conf_track, face_crop).
2. Apply :class:`DualGate` to drop low-SNR / occluded frames.
3. Extract ECAPA voice embedding + ArcFace face embedding from surviving
   regions of each utterance.
4. Fuse via :class:`IdentityFuser` into a single z_id per utterance.
5. Cold-start with :class:`AgglomerativeColdStart` → auto-K cluster labels
   + an 'unknown' bucket.
6. Use the cluster labels as pseudo-speaker IDs and optimise
   :class:`BidirectionalInfoNCE` between the audio-projected and
   visual-projected features for each utterance (Stage-1 LR=1e-3, warmup
   500 steps, freeze backbones — per spec §7).

This script is deliberately a skeleton: the real dataset loader is
project-specific (LRS3, VoxCeleb2, AMI, …). The outer loop below is what
Phase-3 evaluation will drive.

CLI
---
    python scripts/train_identity.py --config configs/default.yaml \
        --manifest data/session_manifest.jsonl --out checkpoints/stage1/
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from avsd_ger.c1_identity import FaceEncoder, IdentityPool, VoiceEncoder
from avsd_ger.c1_identity.cold_start import AgglomerativeColdStart
from avsd_ger.c1_identity.gate import DualGate
from avsd_ger.training import BidirectionalInfoNCE
from avsd_ger.training.run_state import (
    build_provenance,
    load_run_state,
    save_run_state,
)
from avsd_ger.utils import load_config, resolve_device, seed_all
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args


# ---------------------------------------------------------------- dataset iface
def _resolve_data_path(path: str | None, *, kind: str) -> Path:
    if not path:
        raise FileNotFoundError(f"Missing {kind} path in manifest record.")
    raw = Path(path)
    candidates = [raw]
    parts = raw.parts
    if "data" in parts:
        data_idx = parts.index("data")
        candidates.append(_ROOT / Path(*parts[data_idx:]))
    if not raw.is_absolute():
        candidates.append(_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{kind} path does not exist: {path!r}")


def iter_manifest(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield one manifest record per line.

    Expected fields per record (all optional but strongly recommended):
        wav_path, face_path, lip_conf (list[float]) — lip-detector scores per video frame.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------- training step
def _warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / max(1, warmup_steps)


def _speaker_balanced_batches(
    labels: torch.Tensor,
    *,
    speakers_per_batch: int,
    turns_per_speaker: int,
) -> list[torch.Tensor]:
    """Build one deterministic-size epoch with speaker-balanced sampling."""
    if speakers_per_batch < 2 or turns_per_speaker < 1:
        raise ValueError(
            "speaker-balanced sampling requires speakers_per_batch >= 2 "
            "and turns_per_speaker >= 1"
        )
    labels_cpu = labels.detach().cpu().long()
    groups = {
        int(label): (labels_cpu == int(label)).nonzero(as_tuple=True)[0]
        for label in torch.unique(labels_cpu).tolist()
        if int(label) >= 0
    }
    if len(groups) < 2:
        raise RuntimeError("Stage-1 requires at least two known speakers")
    speaker_ids = sorted(groups)
    effective_speakers = min(speakers_per_batch, len(speaker_ids))
    batch_size = effective_speakers * turns_per_speaker
    n_known = sum(int(items.numel()) for items in groups.values())
    n_batches = max(1, math.ceil(n_known / batch_size))
    order = torch.randperm(len(speaker_ids)).tolist()
    batches: list[torch.Tensor] = []
    cursor = 0
    for _ in range(n_batches):
        chosen: list[int] = []
        for _ in range(effective_speakers):
            chosen.append(speaker_ids[order[cursor % len(order)]])
            cursor += 1
            if cursor % len(order) == 0:
                order = torch.randperm(len(speaker_ids)).tolist()
        rows: list[torch.Tensor] = []
        for speaker in chosen:
            candidates = groups[speaker]
            if candidates.numel() >= turns_per_speaker:
                pick = candidates[torch.randperm(candidates.numel())[:turns_per_speaker]]
            else:
                pick = candidates[torch.randint(
                    0, candidates.numel(), (turns_per_speaker,)
                )]
            rows.append(pick)
        batches.append(torch.cat(rows))
    return batches


@torch.no_grad()
def _identity_retrieval_metric(
    fuser: torch.nn.Module,
    voice: torch.Tensor,
    face: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, float]:
    """Return speaker-aware A->V, V->A and mean retrieval accuracy."""
    if voice.shape[0] != face.shape[0] or voice.shape[0] != labels.numel():
        raise ValueError("Dev voice/face/label counts do not match")
    if voice.shape[0] < 2 or torch.unique(labels).numel() < 2:
        raise ValueError("Stage-1 dev selection requires at least two speakers")
    a = torch.nn.functional.normalize(fuser.voice_proj(voice), dim=-1)
    v = torch.nn.functional.normalize(fuser.face_proj(face), dim=-1)
    similarity = a @ v.transpose(0, 1)
    acc_av = (labels[similarity.argmax(dim=1)] == labels).float().mean().item()
    acc_va = (labels[similarity.argmax(dim=0)] == labels).float().mean().item()
    return float(acc_av), float(acc_va), float((acc_av + acc_va) / 2.0)


def train(
    cfg: dict[str, Any],
    manifest: str | Path,
    out_dir: str | Path,
    dev_manifest: str | Path | None = None,
    resume: str | Path | None = None,
    wb: "WandbLogger | None" = None,
) -> None:
    if wb is None:
        wb = WandbLogger(None)
    device = resolve_device(cfg.get("device", "cpu"))
    seed_all(int(cfg.get("seed", 1337)))
    stub = bool(cfg.get("stub_backbones", True))
    if not stub and dev_manifest is None:
        raise ValueError("Production Stage-1 training requires --dev-manifest")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Backbones + projections + loss
    voice = VoiceEncoder(cfg["identity"]["voice_encoder"], stub=stub, device=device)
    face = FaceEncoder(cfg["identity"]["face_encoder"], stub=stub, device=device)
    pool = IdentityPool(cfg["identity"], device=device)

    gate = DualGate(cfg["identity"])
    loss_fn = BidirectionalInfoNCE(cfg["training"]["infonce"]).to(device)
    optim = torch.optim.AdamW(pool.fuser.parameters(), lr=float(cfg["training"]["stage1"]["lr"]))
    warmup_steps = int(cfg["training"]["stage1"].get("warmup_steps", 500))
    base_lr = float(cfg["training"]["stage1"]["lr"])

    # First pass: collect per-utterance fused embeddings (frozen fuser)
    # so cold-start sees a stable geometry. Then we train the fuser with
    # InfoNCE pairs drawn from those pseudo-labels.
    # If the manifest is missing (e.g. stub rehearsal), fall back to a
    # synthetic 8-record list so the training loop still runs end-to-end --
    # mirrors the same fallback in train_stage2.py.
    using_stub_records = not Path(manifest).exists()
    if not using_stub_records:
        records: list[dict[str, Any]] = list(iter_manifest(manifest))
    else:
        print(f"[manifest] {manifest} not found -- falling back to 8 stub records")
        records = [{"utt_id": f"stub_{i:02d}"} for i in range(8)]
    fused_list: list[torch.Tensor] = []
    voice_list: list[torch.Tensor] = []
    face_list: list[torch.Tensor] = []
    participant_ids: list[str | None] = []
    face_cache: dict[str, torch.Tensor] = {}
    face_cache_hits = 0
    face_cache_misses = 0

    with torch.no_grad():
        for rec in tqdm(records, desc="[stage1] embedding first pass", unit="utt"):
            # --- load + dual-gate (stubbed here; real loader decodes audio/video) ---
            wav = _load_wav(rec, stub=stub)
            lip_conf = np.asarray(rec.get("lip_conf", []), dtype=np.float32)
            mask = gate.filter(wav, lip_conf).mask
            # In stub mode AND in manifest-missing fallback mode the wav is
            # Gaussian noise: SNR estimator returns ~0 dB across all frames,
            # well below tau_a=8 dB, so the gate rejects every frame. Bypass
            # the kill-switch in those modes so the training loop can still
            # exercise end-to-end. Real training (manifest-present, real
            # backbones) keeps the gate active as the spec mandates.
            input_is_random = stub or using_stub_records
            if (not input_is_random) and mask.size and mask.sum() == 0:
                continue   # everything filtered out — skip utterance

            voice_emb = voice.embed(wav)
            face_key = str(rec.get("face_path") or rec.get("utt_id") or len(face_cache))
            if (not stub) and face_key in face_cache:
                face_emb = face_cache[face_key].to(device)
                face_cache_hits += 1
            else:
                face_emb = face.embed(_load_face(rec, stub=stub))
                if not stub:
                    face_cache[face_key] = face_emb.detach().cpu()
                    face_cache_misses += 1
            z = pool.fuser(voice_emb.unsqueeze(0), face_emb.unsqueeze(0)).squeeze(0)
            fused_list.append(z.detach().cpu())
            voice_list.append(voice_emb.detach().cpu())
            face_list.append(face_emb.detach().cpu())
            participant_ids.append(
                rec.get("participant_id")
                or rec.get("speaker_id")
                or rec.get("ref_speaker")
            )

    if not fused_list:
        raise RuntimeError("No utterances survived the dual gate — check thresholds.")
    if not stub:
        print(
            f"[face_cache] unique={len(face_cache)} hits={face_cache_hits} "
            f"misses={face_cache_misses}"
        )

    # Freeze a separate dev feature set once. Selection never uses test data,
    # and participant IDs remain the only labels used for AMI evaluation.
    dev_voice_t: torch.Tensor | None = None
    dev_face_t: torch.Tensor | None = None
    dev_labels_t: torch.Tensor | None = None
    if dev_manifest is not None:
        dev_records = list(iter_manifest(dev_manifest))
        dev_voice: list[torch.Tensor] = []
        dev_face: list[torch.Tensor] = []
        dev_names: list[str] = []
        dev_face_cache: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for rec in tqdm(dev_records, desc="[stage1] dev embedding pass", unit="utt"):
                speaker = rec.get("participant_id") or rec.get("speaker_id") or rec.get("ref_speaker")
                if not speaker:
                    raise ValueError("Stage-1 dev record is missing participant_id/speaker_id")
                wav = _load_wav(rec, stub=stub)
                lip_conf = np.asarray(rec.get("lip_conf", []), dtype=np.float32)
                mask = gate.filter(wav, lip_conf).mask
                if (not stub) and mask.size and mask.sum() == 0:
                    continue
                voice_emb = voice.embed(wav)
                face_key = str(rec.get("face_path") or rec.get("utt_id") or len(dev_face_cache))
                if (not stub) and face_key in dev_face_cache:
                    face_emb = dev_face_cache[face_key].to(device)
                else:
                    face_emb = face.embed(_load_face(rec, stub=stub))
                    if not stub:
                        dev_face_cache[face_key] = face_emb.detach().cpu()
                dev_voice.append(voice_emb.detach().cpu())
                dev_face.append(face_emb.detach().cpu())
                dev_names.append(str(speaker))
        if not dev_voice:
            raise RuntimeError("No dev utterances survived the Stage-1 dual gate")
        dev_mapping = {name: idx for idx, name in enumerate(sorted(set(dev_names)))}
        dev_voice_t = torch.stack(dev_voice).to(device)
        dev_face_t = torch.stack(dev_face).to(device)
        dev_labels_t = torch.tensor([dev_mapping[name] for name in dev_names], device=device)

    stage1_cfg = cfg["training"]["stage1"]
    supervision = str(stage1_cfg.get("identity_supervision", "auto")).lower()
    if supervision not in {"auto", "participant", "cold_start"}:
        raise ValueError(
            "training.stage1.identity_supervision must be "
            "auto/participant/cold_start"
        )
    all_participants_present = all(participant_ids)
    use_participants = supervision == "participant" or (
        supervision == "auto" and all_participants_present
    )
    if supervision == "participant" and not all_participants_present:
        missing = sum(value is None for value in participant_ids)
        raise ValueError(
            f"Participant-supervised Stage-1 has {missing} surviving records "
            "without participant_id/speaker_id"
        )

    identity_label_names: list[str | None]
    if use_participants:
        names = [str(value) for value in participant_ids]
        mapping = {name: idx for idx, name in enumerate(sorted(set(names)))}
        labels = torch.tensor([mapping[name] for name in names], device=device)
        identity_label_names = names
        print(
            f"[stage1-supervision] participant labels: "
            f"speakers={len(mapping)} records={len(names)}"
        )
        wb.log({
            "stage1/supervision/participant": 1,
            "stage1/supervision/speakers": len(mapping),
            "stage1/supervision/records": len(names),
        })
    else:
        # Cold-start remains an explicit deployment/non-AMI path. It operates
        # in pretrained voice space, never over the complete supervised AMI
        # split when participant labels are available.
        cold = AgglomerativeColdStart(cfg["identity"])
        cold_input = torch.stack(voice_list, dim=0).numpy()
        cs = cold.fit(cold_input)
        labels = torch.from_numpy(cs.labels).to(device)
        identity_label_names = [
            f"cluster_{int(label):04d}" if int(label) >= 0 else None
            for label in cs.labels
        ]
        print(
            f"[cold_start] K={cs.centroids.shape[0]} "
            f"unknown={cs.n_unknown}/{len(identity_label_names)}"
        )
        wb.log({
            "stage1/cold_start/K": int(cs.centroids.shape[0]),
            "stage1/cold_start/n_unknown": int(cs.n_unknown),
            "stage1/cold_start/n_records": len(identity_label_names),
        })

    # --- InfoNCE training on the pseudo-labelled set --------------------
    voice_t = torch.stack(voice_list, dim=0).to(device)
    face_t = torch.stack(face_list, dim=0).to(device)
    known_idx = (labels >= 0).nonzero(as_tuple=True)[0]
    if known_idx.numel() < 2:
        raise RuntimeError("Cold-start yielded < 2 'known' samples; cannot form InfoNCE batches.")

    n_epochs = int(cfg["training"]["stage1"]["epochs"])
    speakers_per_batch = int(stage1_cfg.get("speakers_per_batch", 16))
    turns_per_speaker = int(stage1_cfg.get("turns_per_speaker", 4))
    provenance = build_provenance(
        stage="stage1_identity",
        cfg=cfg,
        train_manifest=manifest,
        dev_manifest=dev_manifest,
    )
    modules = {"fuser": pool.fuser}
    step = 0
    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = -1
    if resume is not None:
        resume_path = Path(resume)
        if resume_path.is_dir():
            resume_path = resume_path / "last.pt"
        state = load_run_state(
            resume_path,
            expected_provenance=provenance,
            modules=modules,
            optimizer=optim,
            map_location=device,
        )
        start_epoch = int(state["epoch"]) + 1
        step = int(state["global_step"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        print(f"[resume] {resume_path}: next_epoch={start_epoch + 1} step={step}")

    for epoch in range(start_epoch, n_epochs):
        epoch_batches = _speaker_balanced_batches(
            labels,
            speakers_per_batch=speakers_per_batch,
            turns_per_speaker=turns_per_speaker,
        )
        for idx_cpu in epoch_batches:
            idx = idx_cpu.to(device)
            v_emb = voice_t[idx]
            f_emb = face_t[idx]
            # Forward through (trainable) fuser projections to get a, v
            a = pool.fuser.voice_proj(v_emb)
            v = pool.fuser.face_proj(f_emb)
            rep = loss_fn(a, v, speaker_labels=labels[idx])

            for g in optim.param_groups:
                g["lr"] = _warmup_lr(step, warmup_steps, base_lr)
            optim.zero_grad()
            rep.loss.backward()
            optim.step()
            step += 1

            wb.log({
                "stage1/loss/total":  float(rep.loss.item()),
                "stage1/loss/A->V":   float(rep.loss_av.item()),
                "stage1/loss/V->A":   float(rep.loss_va.item()),
                "stage1/acc/A->V":    float(rep.acc_av),
                "stage1/acc/V->A":    float(rep.acc_va),
                "stage1/batch/speakers": int(torch.unique(labels[idx]).numel()),
                "stage1/batch/positives_per_anchor": float(rep.mean_positives),
                "stage1/lr":          float(optim.param_groups[0]["lr"]),
                "stage1/epoch":       int(epoch),
            }, step=step)

        print(
            f"[epoch {epoch+1:02d}] loss={rep.loss.item():.4f} "
            f"(A→V={rep.loss_av.item():.4f} acc={rep.acc_av:.3f} | "
            f"V→A={rep.loss_va.item():.4f} acc={rep.acc_va:.3f})"
        )
        wb.log({
            "stage1/epoch_end/loss":   float(rep.loss.item()),
            "stage1/epoch_end/acc_av": float(rep.acc_av),
            "stage1/epoch_end/acc_va": float(rep.acc_va),
        }, step=step)

        if dev_voice_t is not None and dev_face_t is not None and dev_labels_t is not None:
            pool.fuser.eval()
            dev_acc_av, dev_acc_va, selection_metric = _identity_retrieval_metric(
                pool.fuser, dev_voice_t, dev_face_t, dev_labels_t
            )
            pool.fuser.train()
            print(
                f"[dev {epoch+1:02d}] retrieval={selection_metric:.4f} "
                f"(A→V={dev_acc_av:.4f}, V→A={dev_acc_va:.4f})"
            )
        else:
            # Stub rehearsals keep their historical no-dev behaviour; real
            # training is guarded above and always takes the dev branch.
            dev_acc_av = float(rep.acc_av)
            dev_acc_va = float(rep.acc_va)
            selection_metric = (dev_acc_av + dev_acc_va) / 2.0

        improved = selection_metric > best_metric
        if improved:
            best_metric = selection_metric
            best_epoch = epoch
        state_extra = {
            "selection_metric": "mean_speaker_retrieval_accuracy",
            "dev_acc_av": dev_acc_av,
            "dev_acc_va": dev_acc_va,
        }
        save_run_state(
            out_dir / "last.pt",
            provenance=provenance,
            epoch=epoch,
            global_step=step,
            modules=modules,
            optimizer=optim,
            best_metric=best_metric,
            best_epoch=best_epoch,
            extra=state_extra,
        )
        if improved:
            save_run_state(
                out_dir / "best.pt",
                provenance=provenance,
                epoch=epoch,
                global_step=step,
                modules=modules,
                optimizer=optim,
                best_metric=best_metric,
                best_epoch=best_epoch,
                extra=state_extra,
            )
        wb.log({
            "stage1/dev/acc_av": dev_acc_av,
            "stage1/dev/acc_va": dev_acc_va,
            "stage1/dev/selection_metric": selection_metric,
            "stage1/dev/best_metric": best_metric,
            "stage1/dev/is_best": int(improved),
        }, step=step)

    best_path = out_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError(
            "Stage-1 produced no best checkpoint; increase epochs or check --resume"
        )
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    pool.fuser.load_state_dict(best_state["modules"]["fuser"], strict=True)
    print(f"[selection] restored best epoch={best_epoch + 1} metric={best_metric:.4f}")

    # Persist an actual identity pool, not only the fuser weights. AMI JSONL
    # records include speaker_id, so prefer supervised prototypes when present;
    # otherwise fall back to cold-start cluster labels.
    grouped: dict[str, list[int]] = defaultdict(list)
    labelled = 0
    for i, sid in enumerate(identity_label_names):
        if sid:
            grouped[str(sid)].append(i)
            labelled += 1
    source = "manifest_participant_id" if use_participants else "cold_start_cluster"
    if not grouped:
        raise RuntimeError("Stage-1 produced no known identity groups")

    for sid, idxs in sorted(grouped.items()):
        v_proto = torch.stack([voice_list[i] for i in idxs], dim=0).mean(dim=0)
        f_proto = torch.stack([face_list[i] for i in idxs], dim=0).mean(dim=0)
        v_proto = v_proto / (v_proto.norm() + 1e-8)
        f_proto = f_proto / (f_proto.norm() + 1e-8)
        pool.enroll(
            speaker_id=sid,
            voice_emb=v_proto,
            face_emb=f_proto,
            meta={"n_utterances": len(idxs), "source": source},
        )
    print(f"[pool] enrolled={len(pool)} source={source} labelled_records={labelled}/{len(identity_label_names)}")
    wb.log({
        "stage1/pool/enrolled": len(pool),
        "stage1/pool/labelled_records": labelled,
    }, step=step)

    pool.save(out_dir / "identity_pool_stage1.pt")
    print(f"[done] saved stage-1 fuser + enrollees → {out_dir}")


# ---------------------------------------------------------------- I/O stubs
def _load_wav(rec: dict[str, Any], stub: bool) -> torch.Tensor:
    if stub or not rec.get("wav_path"):
        return torch.randn(16000 * 3)
    import soundfile as sf
    data, _ = sf.read(_resolve_data_path(rec["wav_path"], kind="audio"))
    return torch.from_numpy(np.asarray(data, dtype=np.float32))


def _load_face(rec: dict[str, Any], stub: bool) -> np.ndarray:
    if stub or not rec.get("face_path"):
        return (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    from PIL import Image
    return np.array(Image.open(_resolve_data_path(rec["face_path"], kind="face")).convert("RGB"))


def _resolve_manifest_arg(manifest: str | None, manifest_dir: str | None) -> str:
    if manifest_dir:
        directory = Path(manifest_dir)
        sibling_jsonl = directory.with_suffix(".jsonl")
        if sibling_jsonl.exists():
            print(f"[train_identity] Resolved --manifest-dir {directory} -> {sibling_jsonl}")
            return str(sibling_jsonl)
        raise FileNotFoundError(
            f"--manifest-dir was provided, but converted JSONL was not found: {sibling_jsonl}. "
            "Create it with scripts/ami_visual_to_jsonl.py --manifest-dir <dir> --out <dir>.jsonl"
        )
    if not manifest:
        raise ValueError("Either --manifest or --manifest-dir is required.")
    return manifest


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1 C1 identity training (spec section 7).")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default=None, help="JSONL manifest: one utterance per line.")
    ap.add_argument(
        "--manifest-dir",
        default=None,
        help=(
            "Directory of AMI visual per-meeting manifests. The trainer expects "
            "a JSONL file, so this resolves <dir>.jsonl when present."
        ),
    )
    ap.add_argument("--out", default="checkpoints/stage1/")
    ap.add_argument("--dev-manifest", default=None, help="Validation JSONL used only for checkpoint selection.")
    ap.add_argument("--resume", default=None, help="Path to last.pt or a run directory containing last.pt.")
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.stage1.epochs without editing the YAML config.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override training.stage1.lr without editing the YAML config.",
    )
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override training.stage1.warmup_steps without editing the YAML config.",
    )
    add_wandb_args(ap)
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.manifest = _resolve_manifest_arg(args.manifest, args.manifest_dir)
    if args.epochs is not None:
        cfg.setdefault("training", {}).setdefault("stage1", {})["epochs"] = args.epochs
        print(f"[train_identity] Override stage1.epochs -> {args.epochs}")
    if args.lr is not None:
        cfg.setdefault("training", {}).setdefault("stage1", {})["lr"] = args.lr
        print(f"[train_identity] Override stage1.lr -> {args.lr}")
    if args.warmup_steps is not None:
        cfg.setdefault("training", {}).setdefault("stage1", {})["warmup_steps"] = args.warmup_steps
        print(f"[train_identity] Override stage1.warmup_steps -> {args.warmup_steps}")
    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"stage1-{Path(args.manifest).stem}",
        job_type="train-stage1",
        config={"stage": "stage1", "config_path": args.config, "manifest": args.manifest, **cfg},
    )
    try:
        train(
            cfg,
            args.manifest,
            args.out,
            dev_manifest=args.dev_manifest,
            resume=args.resume,
            wb=wb,
        )
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
