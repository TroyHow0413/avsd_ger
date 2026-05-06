"""Ablation evaluator (spec section 10, Table 2).

Runs the full AVSD-GER pipeline in five configurations against one session
manifest and reports the five primary metrics (SA-WER, SCR, AV-SID Acc, DER,
JER) plus the power-monitor energy number for each run:

    1. Full Model              -- all flags off
    2. w/o C1                  -- disable_c1: true   (ID conditioning off)
    3. w/o C2                  -- disable_c2: true   (skip GER, return ASR 1-best)
    4. w/o C3                  -- disable_c3: true   (no closed-loop retries,
                                                      frozen identity pool)
    5. C3 w/o Conf. Gate       -- disable_conf_gate: true
                                  (must perform *worse* than #4 per spec; this
                                   is the structural-safety proof of the gate)

The manifest format is intentionally permissive -- see SessionRunner and
SessionTurn docstrings. At minimum each turn needs: turn_id, start, end,
audio_wav, video_frames; ref_text + ref_speaker are required for metrics.

    python scripts/eval_ablations.py \
        --config configs/default.yaml \
        --manifest data/session_manifest.json \
        --pool checkpoints/identity_pool.pt \
        --out out/ablation_report.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.pipeline import AVSDGERPipeline            # noqa: E402
from avsd_ger.eval.session import SessionRunner, SessionTurn  # noqa: E402
from avsd_ger.eval.metrics import evaluate_session, MetricsReport  # noqa: E402
from avsd_ger.eval.power import PowerMonitor             # noqa: E402
from avsd_ger.frontend import get_frontend_profile, list_frontend_profiles  # noqa: E402
from avsd_ger.utils import load_config                   # noqa: E402
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args  # noqa: E402


ABLATION_MATRIX = [
    ("full_model",           {}),
    ("wo_c1",                {"disable_c1": True}),
    ("wo_c2",                {"disable_c2": True}),
    ("wo_c3",                {"disable_c3": True}),
    ("c3_wo_conf_gate",      {"disable_conf_gate": True}),
]


def _frontend_choices() -> list[str]:
    return [p.key for p in list_frontend_profiles()]


def _resolve_manifest_paths(spec: str) -> list[Path]:
    """Resolve one manifest file, a directory of JSON manifests, or a glob."""

    p = Path(spec)
    if p.is_dir():
        paths = sorted(p.glob("*.json"))
    elif any(ch in spec for ch in "*?[]"):
        paths = sorted(Path(x) for x in glob.glob(spec))
    else:
        paths = [p]

    paths = [x for x in paths if x.exists() and x.is_file()]
    if not paths:
        raise FileNotFoundError(f"No manifest JSON files matched: {spec}")
    return paths


def _output_path_for_manifest(out_arg: str, manifest_path: Path, multi: bool) -> Path:
    """Back compatible single-file output; directory-style output for batches."""

    out = Path(out_arg)
    if not multi:
        return out
    out_dir = out.parent / out.stem if out.suffix.lower() == ".json" else out
    return out_dir / f"{manifest_path.stem}.json"


def _apply_frontend_profile(cfg: dict[str, Any], frontend_profile: str | None) -> None:
    if frontend_profile is not None:
        cfg.setdefault("frontend", {})["profile"] = frontend_profile


def _frontend_meta_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    frontend_cfg = cfg.get("frontend", {}) or {}
    frontend_key = str(frontend_cfg.get("profile", "oracle_turns"))
    try:
        frontend_profile = get_frontend_profile(frontend_key)
        return {
            "profile": frontend_profile.key,
            "label": frontend_profile.label,
            "tier": frontend_profile.tier,
            "diarization": frontend_profile.diarization,
            "active_speaker": frontend_profile.active_speaker,
            "claim": frontend_profile.claim,
        }
    except KeyError:
        return {"profile": frontend_key, "label": "custom", "tier": "custom"}


def _resolve_manifest_path(path: str | None, *, kind: str) -> Path:
    """Resolve manifest paths, accepting Windows separators on Linux hosts."""

    if not path:
        raise FileNotFoundError(f"Missing {kind} path in manifest.")

    raw = Path(path)
    if raw.exists():
        return raw

    if "\\" in path:
        normalized = Path(path.replace("\\", "/"))
        if normalized.exists():
            return normalized

    raise FileNotFoundError(
        f"{kind} path does not exist: {path!r}. "
        "If this manifest was generated on Windows, replace backslashes with '/'."
    )


def _load_audio(
    path: str | None,
    *,
    kind: str = "audio",
    allow_synthetic_audio: bool = False,
) -> np.ndarray | torch.Tensor:
    if allow_synthetic_audio:
        try:
            resolved = _resolve_manifest_path(path, kind=kind)
        except FileNotFoundError:
            return torch.randn(16000 * 3)
    else:
        resolved = _resolve_manifest_path(path, kind=kind)

    try:
        import soundfile as sf
        wav, sr = sf.read(resolved)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        return wav.astype(np.float32)
    except Exception as exc:
        if allow_synthetic_audio:
            return torch.randn(16000 * 3)
        raise RuntimeError(f"Failed to read {kind} file {str(resolved)!r}: {exc}") from exc


def _load_face(path: str | None) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _load_video(path: str | None, num_frames: int = 75) -> tuple[torch.Tensor | None, bool]:
    if path is None or not Path(path).exists():
        return None, False
    arr = np.load(path)
    t = torch.from_numpy(arr).float()
    if t.ndim == 3:
        t = t.unsqueeze(1)
    return (t / 255.0 if t.max() > 1.5 else t), True


def _maybe_tensor(x) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return torch.as_tensor(x)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def _load_turns(manifest: dict[str, Any], *, allow_synthetic_audio: bool = False) -> list[SessionTurn]:
    turns: list[SessionTurn] = []
    for i, row in enumerate(manifest.get("turns", manifest.get("utterances", []))):
        video_frames, has_visual = _load_video(row.get("mouth_roi") or row.get("video"))
        turns.append(SessionTurn(
            turn_id=str(row.get("turn_id", row.get("utt_id", f"t{i:04d}"))),
            start=float(row.get("start", i)),
            end=float(row.get("end", i + 1)),
            audio_wav=_load_audio(
                row.get("audio"),
                kind=f"turn audio for {row.get('turn_id', row.get('utt_id', f't{i:04d}'))}",
                allow_synthetic_audio=allow_synthetic_audio,
            ),
            video_frames=video_frames,
            has_visual=has_visual,
            face_image=None,
            speaker_mask_v=_maybe_tensor(row.get("speaker_mask_v")),
            snr_per_tok=_maybe_tensor(row.get("snr_per_tok")),
            lip_conf_v=_maybe_tensor(row.get("lip_conf_v")),
            ref_text=row.get("ref_text"),
            ref_speaker=row.get("ref_speaker"),
        ))
    return turns


def _enroll_pool_from_manifest(
    pipe: AVSDGERPipeline,
    manifest: dict[str, Any],
    *,
    allow_synthetic_audio: bool = False,
) -> None:
    speakers = manifest.get("speakers", [])
    if not speakers:
        return
    for spk in speakers:
        pipe.enroll(
            spk["speaker_id"],
            _load_audio(
                spk.get("enrollment_audio"),
                kind=f"enrollment audio for {spk.get('speaker_id', '<unknown>')}",
                allow_synthetic_audio=allow_synthetic_audio,
            ),
            _load_face(spk.get("enrollment_face")),
            meta=spk.get("meta"),
        )


def _run_one(
    cfg: dict[str, Any],
    ablation_name: str,
    flags: dict[str, bool],
    manifest: dict[str, Any],
    pool_path: str | None,
    monitor: PowerMonitor | None,
) -> dict[str, Any]:
    cfg_run = copy.deepcopy(cfg)
    cfg_run.setdefault("ablation", {})
    # Clear any caller-supplied ablation flags for a clean baseline, then apply.
    cfg_run["ablation"] = {
        "disable_c1": False,
        "disable_c2": False,
        "disable_c3": False,
        "disable_conf_gate": False,
    }
    cfg_run["ablation"].update(flags)
    if cfg_run.get("ger", {}).get("mode") == "visual_only":
        cfg_run["ablation"]["disable_c2"] = False

    allow_synthetic_audio = bool(cfg_run.get("stub_backbones", False))
    pipe = AVSDGERPipeline(cfg_run)
    if pool_path and Path(pool_path).exists():
        pipe.load_pool(pool_path)
    else:
        _enroll_pool_from_manifest(
            pipe,
            manifest,
            allow_synthetic_audio=allow_synthetic_audio,
        )

    turns = _load_turns(manifest, allow_synthetic_audio=allow_synthetic_audio)
    runner = SessionRunner(pipe)

    if monitor is not None:
        with monitor.measure(ablation_name):
            session = runner.run(turns)
        pwr = monitor.last_report()
    else:
        session = runner.run(turns)
        pwr = None

    report = evaluate_session(session.turns)
    frontend_meta = _frontend_meta_from_cfg(cfg_run)

    return {
        "ablation": ablation_name,
        "flags": flags,
        "frontend": frontend_meta,
        "metrics": {
            "sa_wer": report.sa_wer,
            "wer": report.wer,
            "scr": report.scr,
            "av_sid_acc": report.av_sid_acc,
            "der": report.der,
            "jer": report.jer,
            "n_ref_words": report.n_ref_words,
            "n_turns": report.n_turns,
        },
        "power": ({
            "label": pwr.label,
            "duration_s": pwr.duration_s,
            "n_samples": pwr.n_samples,
            "avg_power_w": pwr.avg_power_w,
            "peak_power_w": pwr.peak_power_w,
            "energy_j": pwr.energy_j,
            "energy_wh": pwr.energy_wh,
            "idle_baseline_w": pwr.idle_baseline_w,
            "degraded": pwr.degraded,
        } if pwr is not None else None),
        "transcript": session.transcript,
        "speaker_order": session.speaker_order,
    }


def _run_manifest(
    cfg: dict[str, Any],
    manifest_path: Path,
    pool_path: str | None,
    monitor: PowerMonitor | None,
    only: list[str] | None,
    wb: WandbLogger,
    step_offset: int = 0,
) -> tuple[list[dict[str, Any]], bool | None]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results: list[dict[str, Any]] = []
    for i, (name, flags) in enumerate(ABLATION_MATRIX):
        if only is not None and name not in only:
            continue
        print(f"\n=== {manifest_path.name} :: running ablation: {name}  flags={flags} ===")
        r = _run_one(cfg, name, flags, manifest, pool_path, monitor)
        print(json.dumps(r["metrics"], indent=2))
        results.append(r)

        prefix = f"manifest/{manifest_path.stem}/ablation/{name}"
        wb.log({
            f"{prefix}/sa_wer":     r["metrics"]["sa_wer"],
            f"{prefix}/wer":        r["metrics"]["wer"],
            f"{prefix}/scr":        r["metrics"]["scr"],
            f"{prefix}/av_sid_acc": r["metrics"]["av_sid_acc"],
            f"{prefix}/der":        r["metrics"]["der"],
            f"{prefix}/jer":        r["metrics"]["jer"],
            f"{prefix}/frontend":   r["frontend"]["profile"],
            **({f"{prefix}/energy_wh": r["power"]["energy_wh"],
                f"{prefix}/avg_power_w": r["power"]["avg_power_w"]} if r["power"] else {}),
        }, step=step_offset + i)

    by_name = {r["ablation"]: r for r in results}
    spec_check_pass: bool | None = None
    if "wo_c3" in by_name and "c3_wo_conf_gate" in by_name:
        a = by_name["wo_c3"]["metrics"]["sa_wer"]
        b = by_name["c3_wo_conf_gate"]["metrics"]["sa_wer"]
        spec_check_pass = b >= a
        print(
            f"\n[spec check] {manifest_path.name}: C3-w/o-gate SA-WER ({b:.4f}) "
            f"{'>=' if spec_check_pass else '<'} w/o-C3 SA-WER ({a:.4f}): "
            f"{'PASS' if spec_check_pass else 'FAIL -- gate-removed variant should degrade'}"
        )

    return results, spec_check_pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument(
        "--manifest",
        required=True,
        help="Manifest JSON file, directory containing *.json manifests, or a glob pattern.",
    )
    p.add_argument("--pool", default=str(ROOT / "checkpoints/identity_pool.pt"))
    p.add_argument("--out", default=str(ROOT / "out/ablation_report.json"))
    p.add_argument("--no-power", action="store_true", help="skip PowerMonitor")
    p.add_argument("--idle-calibrate-s", type=float, default=1.0)
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict to a subset of {full_model,wo_c1,wo_c2,wo_c3,c3_wo_conf_gate}",
    )
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
        help=(
            "Override cfg.ger.mode. audio_only ignores mouth ROI/lip_hyp/<AV_CTX>; "
            "av uses them only when a real mouth_roi/video path exists; "
            "visual_only is reserved for future VSR-only GER experiments."
        ),
    )
    p.add_argument(
        "--frontend-profile",
        default=None,
        choices=_frontend_choices(),
        help="Override cfg.frontend.profile for reporting/experiment grouping.",
    )
    add_wandb_args(p)
    args = p.parse_args()

    cfg = load_config(args.config)
    # Apply --llm-quant override before any per-row pipeline gets built.
    if args.llm_quant is not None:
        cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant
        print(f"[eval_ablations] Override llm_quant -> {args.llm_quant}")
    if args.ger_mode is not None:
        cfg.setdefault("ger", {})["mode"] = args.ger_mode
        print(f"[eval_ablations] Override ger.mode -> {args.ger_mode}")
    _apply_frontend_profile(cfg, args.frontend_profile)
    manifest_paths = _resolve_manifest_paths(args.manifest)
    multi = len(manifest_paths) > 1
    frontend_meta = _frontend_meta_from_cfg(cfg)
    print(
        f"[eval_ablations] frontend={frontend_meta['profile']} "
        f"({frontend_meta.get('tier', 'custom')}); manifests={len(manifest_paths)}"
    )

    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=(
            f"ablation-{Path(args.manifest).stem}"
            if not multi else f"ablation-batch-{Path(args.manifest).stem}"
        ),
        job_type="eval-ablations",
        config={
            "config_path": args.config,
            "manifest": args.manifest,
            "n_manifests": len(manifest_paths),
            "frontend_profile": frontend_meta["profile"],
            "frontend_tier": frontend_meta.get("tier"),
            **cfg,
        },
    )

    monitor = None
    if not args.no_power:
        monitor = PowerMonitor()
        monitor.calibrate_idle(duration_s=args.idle_calibrate_s)

    all_runs: list[dict[str, Any]] = []
    spec_checks: dict[str, bool | None] = {}
    ablations_per_manifest = len(args.only) if args.only is not None else len(ABLATION_MATRIX)
    for m_idx, manifest_path in enumerate(manifest_paths):
        results, spec_check_pass = _run_manifest(
            cfg,
            manifest_path,
            args.pool,
            monitor,
            args.only,
            wb,
            step_offset=m_idx * max(1, ablations_per_manifest),
        )
        out_path = _output_path_for_manifest(args.out, manifest_path, multi=multi)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": str(manifest_path),
            "frontend": frontend_meta,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[wrote] {out_path}")
        all_runs.append(payload)
        spec_checks[manifest_path.stem] = spec_check_pass

    if multi:
        out = Path(args.out)
        out_dir = out.parent / out.stem if out.suffix.lower() == ".json" else out
        summary_path = out_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "frontend": frontend_meta,
                "n_manifests": len(manifest_paths),
                "spec_checks": spec_checks,
                "runs": all_runs,
            }, f, indent=2)
        print(f"\n[wrote] {summary_path}")

    # Pin the headline numbers to the run summary so they're sortable in the W&B UI.
    summary: dict[str, Any] = {}
    summary["summary/frontend_profile"] = frontend_meta["profile"]
    summary["summary/frontend_tier"] = frontend_meta.get("tier")
    for run in all_runs:
        stem = Path(run["manifest"]).stem
        for r in run["results"]:
            for k, v in r["metrics"].items():
                summary[f"summary/{stem}/{r['ablation']}/{k}"] = v
    for stem, passed in spec_checks.items():
        if passed is not None:
            summary[f"summary/{stem}/spec_check_c3_gate_pass"] = bool(passed)
    wb.summary(summary)
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
