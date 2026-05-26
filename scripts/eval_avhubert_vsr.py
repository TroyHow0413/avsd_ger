"""Standalone AV-HuBERT visual-only / features-only evaluator.

This script is intentionally outside the full AVSD-GER pipeline:
  - no Whisper / ASR
  - no IdentityFuser / identity pool
  - no GER / C2
  - no C3 feedback

It reads pre-extracted mouth ROI clips from an AMI visual manifest and runs
AV-HuBERT VSR. If the configured checkpoint has a VSR decoder, it reports
visual-only WER from lip_hyp. If text decoding is unavailable, it still reports
feature-extraction coverage and marks the WER path as unavailable via
empty_hyp_rate.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.backbones.vsr_avhubert import AVHubertVSR  # noqa: E402
from avsd_ger.utils import load_config, resolve_device  # noqa: E402
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args  # noqa: E402


def _resolve_paths(spec: str) -> list[Path]:
    p = Path(spec)
    if p.is_dir():
        paths = sorted(p.glob("*.json")) + sorted(p.glob("*.jsonl"))
    elif any(ch in spec for ch in "*?[]"):
        paths = sorted(Path(x) for x in glob.glob(spec))
    else:
        paths = [p]
    paths = [x for x in paths if x.exists() and x.is_file()]
    if not paths:
        raise FileNotFoundError(f"No manifest files matched: {spec}")
    return paths


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from data
    else:
        yield from data.get("turns", data.get("utterances", []))


def _project_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    p2 = ROOT / path
    if p2.exists():
        return p2
    if "\\" in path:
        p3 = ROOT / path.replace("\\", "/")
        if p3.exists():
            return p3
    return None


def _row_ref(row: dict[str, Any]) -> str:
    return str(row.get("ref_text") or row.get("target") or row.get("text") or "")


def _row_id(row: dict[str, Any], i: int) -> str:
    return str(row.get("turn_id") or row.get("utt_id") or row.get("id") or f"utt_{i:06d}")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            diag = dp[i - 1][j - 1] + sub_cost
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            best = diag
            op = 0
            if delete < best:
                best = delete
                op = 1
            if insert < best:
                best = insert
                op = 2
            dp[i][j] = best
            back[i][j] = op

    sub = delete = insert = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if i > 0 and j > 0 and op == 0:
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i -= 1
            j -= 1
        elif i > 0 and op == 1:
            delete += 1
            i -= 1
        else:
            insert += 1
            j -= 1
    return sub, delete, insert


def _load_video(path: Path) -> torch.Tensor:
    arr = np.load(path)
    t = torch.from_numpy(arr).float()
    if t.ndim == 3:
        t = t.unsqueeze(1)
    if t.ndim != 4:
        raise ValueError(f"Expected mouth ROI [T,H,W] or [T,1,H,W], got shape {tuple(t.shape)}")
    if t.shape[1] != 1 and t.shape[-1] == 1:
        t = t.permute(0, 3, 1, 2).contiguous()
    if t.shape[1] != 1:
        raise ValueError(f"Expected grayscale channel dimension 1, got shape {tuple(t.shape)}")
    return t / 255.0 if float(t.max()) > 1.5 else t


@dataclass
class VSRResult:
    manifest: str
    utt_id: str
    mouth_roi: str
    ref_text: str
    hyp_text: str
    n_ref_words: int
    n_sub: int
    n_del: int
    n_ins: int
    wer: float
    n_feature_frames: int
    decode_error: str | None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--manifest", required=True, help="Visual manifest .json/.jsonl, directory, or glob.")
    p.add_argument("--device", default=None, help="cuda, cpu, mps; defaults to config device.")
    p.add_argument("--limit", type=int, default=None, help="Optional quick smoke-test limit.")
    p.add_argument("--stub", action="store_true", help="Use AV-HuBERT stub outputs.")
    p.add_argument("--out", default=str(ROOT / "out/avhubert_vsr_eval.json"))
    add_wandb_args(p)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device or str(cfg.get("device", "cpu")))
    manifests = _resolve_paths(args.manifest)
    vsr_cfg = dict(cfg.get("vsr", {}))
    stub = bool(args.stub or cfg.get("stub_backbones", False))

    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"avhubert-vsr-{Path(args.manifest).stem}",
        job_type="eval-avhubert-vsr",
        config={
            "baseline": "avhubert_visual_only",
            "pipeline": "none",
            "uses_asr": False,
            "uses_identity_fuser": False,
            "uses_ger": False,
            "uses_c3": False,
            "config": args.config,
            "manifest": args.manifest,
            "n_manifests": len(manifests),
            "device": str(device),
            "checkpoint": vsr_cfg.get("checkpoint"),
            "emit_text_requested": bool(vsr_cfg.get("emit_text", True)),
            "stub": stub,
            "limit": args.limit,
            "out": args.out,
        },
    )

    runner = AVHubertVSR(vsr_cfg, stub=stub, device=device)
    results: list[VSRResult] = []
    skipped_no_visual = 0
    total_ref = total_sub = total_del = total_ins = 0
    n_empty_hyp = 0
    n_decode_errors = 0
    seen = 0

    for manifest_path in manifests:
        for i, row in enumerate(_iter_rows(manifest_path)):
            if args.limit is not None and seen >= args.limit:
                break
            roi_path = _project_path(row.get("mouth_roi") or row.get("video_frames") or row.get("video_path"))
            if roi_path is None:
                skipped_no_visual += 1
                continue

            video = _load_video(roi_path)
            out = runner.extract(video)
            hyp_text = str(out.get("lip_hyp") or "")
            ref_text = _row_ref(row)
            ref_toks = _tokens(ref_text)
            hyp_toks = _tokens(hyp_text)
            n_sub, n_del, n_ins = _edit_counts(ref_toks, hyp_toks)
            n_ref = len(ref_toks)
            wer = (n_sub + n_del + n_ins) / n_ref if n_ref else 0.0
            feats = out.get("vsr_features")
            n_feature_frames = int(feats.shape[0]) if hasattr(feats, "shape") and len(feats.shape) > 0 else 0
            decode_error = getattr(runner, "last_decode_error", None)
            if not hyp_text.strip():
                n_empty_hyp += 1
            if decode_error:
                n_decode_errors += 1

            results.append(VSRResult(
                manifest=str(manifest_path),
                utt_id=_row_id(row, i),
                mouth_roi=str(roi_path),
                ref_text=ref_text,
                hyp_text=hyp_text,
                n_ref_words=n_ref,
                n_sub=n_sub,
                n_del=n_del,
                n_ins=n_ins,
                wer=wer,
                n_feature_frames=n_feature_frames,
                decode_error=decode_error,
            ))
            total_ref += n_ref
            total_sub += n_sub
            total_del += n_del
            total_ins += n_ins
            seen += 1
            if seen % 25 == 0:
                cur_wer = (total_sub + total_del + total_ins) / max(1, total_ref)
                empty_rate = n_empty_hyp / max(1, seen)
                print(
                    f"[avhubert-vsr] {seen} clips, WER={cur_wer:.4f}, empty_hyp={empty_rate:.2%}",
                    flush=True,
                )
                wb.log({
                    "running/wer": cur_wer,
                    "running/word_accuracy": 1.0 - cur_wer,
                    "running/empty_hyp_rate": empty_rate,
                    "running/n_clips": seen,
                    "running/n_ref_words": total_ref,
                    "running/skipped_no_visual": skipped_no_visual,
                }, step=seen)
        if args.limit is not None and seen >= args.limit:
            break

    total_err = total_sub + total_del + total_ins
    wer = total_err / total_ref if total_ref else 0.0
    empty_hyp_rate = n_empty_hyp / max(1, len(results))
    payload = {
        "baseline": "avhubert_visual_only",
        "config": args.config,
        "manifest": args.manifest,
        "n_manifests": len(manifests),
        "device": str(device),
        "checkpoint": vsr_cfg.get("checkpoint"),
        "emit_text_requested": bool(vsr_cfg.get("emit_text", True)),
        "emit_text_active_after_load": bool(getattr(runner, "emit_text", False)),
        "generator_available": bool(getattr(runner, "_generator", None) is not None),
        "stub": stub,
        "metrics": {
            "wer": wer,
            "word_accuracy": 1.0 - wer,
            "n_ref_words": total_ref,
            "n_sub": total_sub,
            "n_del": total_del,
            "n_ins": total_ins,
            "n_clips": len(results),
            "skipped_no_visual": skipped_no_visual,
            "n_empty_hyp": n_empty_hyp,
            "empty_hyp_rate": empty_hyp_rate,
            "n_decode_errors": n_decode_errors,
            "mean_feature_frames": (
                sum(x.n_feature_frames for x in results) / len(results) if results else 0.0
            ),
        },
        "utterances": [asdict(x) for x in results],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2), flush=True)
    print(f"[wrote] {out_path}", flush=True)
    wb.summary({
        "summary/wer": wer,
        "summary/word_accuracy": 1.0 - wer,
        "summary/n_ref_words": total_ref,
        "summary/n_clips": len(results),
        "summary/skipped_no_visual": skipped_no_visual,
        "summary/empty_hyp_rate": empty_hyp_rate,
        "summary/n_decode_errors": n_decode_errors,
        "summary/emit_text_active_after_load": bool(getattr(runner, "emit_text", False)),
        "summary/generator_available": bool(getattr(runner, "_generator", None) is not None),
        "summary/out": str(out_path),
    })
    wb.log({
        "final/wer": wer,
        "final/word_accuracy": 1.0 - wer,
        "final/n_clips": len(results),
        "final/empty_hyp_rate": empty_hyp_rate,
        "final/skipped_no_visual": skipped_no_visual,
    }, step=max(1, seen))
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
