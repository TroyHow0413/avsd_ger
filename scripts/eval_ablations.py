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
from avsd_ger.utils import load_config                   # noqa: E402
from avsd_ger.wandb_logger import WandbLogger, add_wandb_args  # noqa: E402


ABLATION_MATRIX = [
    ("full_model",           {}),
    ("wo_c1",                {"disable_c1": True}),
    ("wo_c2",                {"disable_c2": True}),
    ("wo_c3",                {"disable_c3": True}),
    ("c3_wo_conf_gate",      {"disable_conf_gate": True}),
]


def _load_audio(path: str | None) -> np.ndarray | torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.randn(16000 * 3)
    try:
        import soundfile as sf
        wav, sr = sf.read(path)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        return wav.astype(np.float32)
    except Exception:
        return torch.randn(16000 * 3)


def _load_video(path: str | None, num_frames: int = 75) -> torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.rand(num_frames, 1, 96, 96)
    arr = np.load(path)
    t = torch.from_numpy(arr).float()
    if t.ndim == 3:
        t = t.unsqueeze(1)
    return t / 255.0 if t.max() > 1.5 else t


def _maybe_tensor(x) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return torch.as_tensor(x)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def _load_turns(manifest: dict[str, Any]) -> list[SessionTurn]:
    turns: list[SessionTurn] = []
    for i, row in enumerate(manifest.get("turns", manifest.get("utterances", []))):
        turns.append(SessionTurn(
            turn_id=str(row.get("turn_id", row.get("utt_id", f"t{i:04d}"))),
            start=float(row.get("start", i)),
            end=float(row.get("end", i + 1)),
            audio_wav=_load_audio(row.get("audio")),
            video_frames=_load_video(row.get("mouth_roi") or row.get("video")),
            face_image=None,
            speaker_mask_v=_maybe_tensor(row.get("speaker_mask_v")),
            snr_per_tok=_maybe_tensor(row.get("snr_per_tok")),
            lip_conf_v=_maybe_tensor(row.get("lip_conf_v")),
            ref_text=row.get("ref_text"),
            ref_speaker=row.get("ref_speaker"),
        ))
    return turns


def _enroll_pool_from_manifest(pipe: AVSDGERPipeline, manifest: dict[str, Any]) -> None:
    speakers = manifest.get("speakers", [])
    if not speakers:
        return
    for spk in speakers:
        pipe.enroll(
            spk["speaker_id"],
            _load_audio(spk.get("enrollment_audio")),
            # enrol_face: stub with random RGB if missing
            spk.get("enrollment_face")
            and _load_audio(spk.get("enrollment_face"))
            or (np.random.rand(224, 224, 3) * 255).astype(np.uint8),
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

    pipe = AVSDGERPipeline(cfg_run)
    if pool_path and Path(pool_path).exists():
        pipe.load_pool(pool_path)
    else:
        _enroll_pool_from_manifest(pipe, manifest)

    turns = _load_turns(manifest)
    runner = SessionRunner(pipe)

    if monitor is not None:
        with monitor.measure(ablation_name):
            session = runner.run(turns)
        pwr = monitor.last_report()
    else:
        session = runner.run(turns)
        pwr = None

    report = evaluate_session(session.turns)

    return {
        "ablation": ablation_name,
        "flags": flags,
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--manifest", required=True)
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
    add_wandb_args(p)
    args = p.parse_args()

    cfg = load_config(args.config)
    # Apply --llm-quant override before any per-row pipeline gets built.
    if args.llm_quant is not None:
        cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant
        print(f"[eval_ablations] Override llm_quant -> {args.llm_quant}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"ablation-{Path(args.manifest).stem}",
        job_type="eval-ablations",
        config={"config_path": args.config, "manifest": args.manifest, **cfg},
    )

    monitor = None
    if not args.no_power:
        monitor = PowerMonitor()
        monitor.calibrate_idle(duration_s=args.idle_calibrate_s)

    results: list[dict[str, Any]] = []
    for i, (name, flags) in enumerate(ABLATION_MATRIX):
        if args.only is not None and name not in args.only:
            continue
        print(f"\n=== running ablation: {name}  flags={flags} ===")
        r = _run_one(cfg, name, flags, manifest, args.pool, monitor)
        print(json.dumps(r["metrics"], indent=2))
        results.append(r)

        # One W&B "step" per ablation row, namespaced by row name so charts
        # can show all 5 metrics x 5 rows on the same dashboard.
        wb.log({
            f"ablation/{name}/sa_wer":     r["metrics"]["sa_wer"],
            f"ablation/{name}/wer":        r["metrics"]["wer"],
            f"ablation/{name}/scr":        r["metrics"]["scr"],
            f"ablation/{name}/av_sid_acc": r["metrics"]["av_sid_acc"],
            f"ablation/{name}/der":        r["metrics"]["der"],
            f"ablation/{name}/jer":        r["metrics"]["jer"],
            **({f"ablation/{name}/energy_wh": r["power"]["energy_wh"],
                f"ablation/{name}/avg_power_w": r["power"]["avg_power_w"]} if r["power"] else {}),
        }, step=i)

    # Spec-mandated sanity: C3 w/o Conf. Gate must be worse than w/o C3.
    by_name = {r["ablation"]: r for r in results}
    spec_check_pass: bool | None = None
    if "wo_c3" in by_name and "c3_wo_conf_gate" in by_name:
        a = by_name["wo_c3"]["metrics"]["sa_wer"]
        b = by_name["c3_wo_conf_gate"]["metrics"]["sa_wer"]
        spec_check_pass = b >= a  # higher SA-WER = worse
        print(
            f"\n[spec check] C3-w/o-gate SA-WER ({b:.4f}) "
            f"{'>=' if spec_check_pass else '<'} w/o-C3 SA-WER ({a:.4f}): "
            f"{'PASS' if spec_check_pass else 'FAIL -- gate-removed variant should degrade'}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\n[wrote] {out_path}")

    # Pin the headline numbers to the run summary so they're sortable in the W&B UI.
    summary: dict[str, Any] = {}
    for r in results:
        for k, v in r["metrics"].items():
            summary[f"summary/{r['ablation']}/{k}"] = v
    if spec_check_pass is not None:
        summary["summary/spec_check_c3_gate_pass"] = bool(spec_check_pass)
    wb.summary(summary)
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
