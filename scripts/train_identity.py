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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from avsd_ger.c1_identity import FaceEncoder, IdentityPool, VoiceEncoder
from avsd_ger.c1_identity.cold_start import AgglomerativeColdStart
from avsd_ger.c1_identity.gate import DualGate
from avsd_ger.training import BidirectionalInfoNCE
from avsd_ger.utils import load_config, resolve_device, seed_all
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args


# ---------------------------------------------------------------- dataset iface
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


def train(
    cfg: dict[str, Any],
    manifest: str | Path,
    out_dir: str | Path,
    wb: "WandbLogger | None" = None,
) -> None:
    if wb is None:
        wb = WandbLogger(None)
    device = resolve_device(cfg.get("device", "cpu"))
    seed_all(int(cfg.get("seed", 1337)))
    stub = bool(cfg.get("stub_backbones", True))

    # Backbones + projections + loss
    voice = VoiceEncoder(cfg["identity"]["voice_encoder"], stub=stub, device=device)
    face = FaceEncoder(cfg["identity"]["face_encoder"], stub=stub, device=device)
    pool = IdentityPool(cfg["identity"], device=device)

    gate = DualGate(cfg["identity"])
    cold = AgglomerativeColdStart(cfg["identity"])

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

    with torch.no_grad():
        for rec in records:
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
            face_emb = face.embed(_load_face(rec, stub=stub))
            z = pool.fuser(voice_emb.unsqueeze(0), face_emb.unsqueeze(0)).squeeze(0)
            fused_list.append(z.detach().cpu())
            voice_list.append(voice_emb.detach().cpu())
            face_list.append(face_emb.detach().cpu())

    if not fused_list:
        raise RuntimeError("No utterances survived the dual gate — check thresholds.")

    # --- Cold-start: data-driven K with unknown bucket ------------------
    fused = torch.stack(fused_list, dim=0).numpy()
    cs = cold.fit(fused)
    print(f"[cold_start] K={cs.centroids.shape[0]}  unknown={cs.n_unknown}/{len(records)}")
    wb.log({
        "stage1/cold_start/K": int(cs.centroids.shape[0]),
        "stage1/cold_start/n_unknown": int(cs.n_unknown),
        "stage1/cold_start/n_records": len(records),
    })

    # --- InfoNCE training on the pseudo-labelled set --------------------
    voice_t = torch.stack(voice_list, dim=0).to(device)
    face_t = torch.stack(face_list, dim=0).to(device)
    labels = torch.from_numpy(cs.labels).to(device)
    known_idx = (labels >= 0).nonzero(as_tuple=True)[0]
    if known_idx.numel() < 2:
        raise RuntimeError("Cold-start yielded < 2 'known' samples; cannot form InfoNCE batches.")

    n_epochs = int(cfg["training"]["stage1"]["epochs"])
    batch = min(64, known_idx.numel())
    step = 0
    for epoch in range(n_epochs):
        perm = known_idx[torch.randperm(known_idx.numel(), device=device)]
        for s in range(0, perm.numel() - batch + 1, batch):
            idx = perm[s : s + batch]
            v_emb = voice_t[idx]
            f_emb = face_t[idx]
            # Forward through (trainable) fuser projections to get a, v
            a = pool.fuser.voice_proj(v_emb)
            v = pool.fuser.face_proj(f_emb)
            rep = loss_fn(a, v)

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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool.save(out_dir / "identity_pool_stage1.pt")
    print(f"[done] saved stage-1 fuser + enrollees → {out_dir}")


# ---------------------------------------------------------------- I/O stubs
def _load_wav(rec: dict[str, Any], stub: bool) -> torch.Tensor:
    if stub or not rec.get("wav_path"):
        return torch.randn(16000 * 3)
    import soundfile as sf
    data, _ = sf.read(rec["wav_path"])
    return torch.from_numpy(np.asarray(data, dtype=np.float32))


def _load_face(rec: dict[str, Any], stub: bool) -> np.ndarray:
    if stub or not rec.get("face_path"):
        return (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    from PIL import Image
    return np.array(Image.open(rec["face_path"]).convert("RGB"))


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
        train(cfg, args.manifest, args.out, wb=wb)
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
