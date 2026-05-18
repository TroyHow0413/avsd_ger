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
from collections import Counter
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


def _apply_safe_core_preset(cfg: dict[str, Any], preset: str | None) -> None:
    if preset is None:
        return

    feedback = cfg.setdefault("feedback", {})
    feedback["enable_pool_update"] = False
    if preset == "no_ger_gate":
        feedback["enable_ger_safety_gate"] = False
    elif preset == "artifact_only":
        feedback.update({
            "enable_ger_safety_gate": True,
            "enable_ger_artifact_gate": True,
            "enable_ger_length_gate": False,
            "enable_ger_overlap_gate": False,
            "enable_ger_acoustic_fallback": False,
        })
    elif preset == "text_gates":
        feedback.update({
            "enable_ger_safety_gate": True,
            "enable_ger_artifact_gate": True,
            "enable_ger_length_gate": True,
            "enable_ger_overlap_gate": True,
            "enable_ger_acoustic_fallback": False,
        })
    elif preset == "full":
        feedback.update({
            "enable_ger_safety_gate": True,
            "enable_ger_artifact_gate": True,
            "enable_ger_length_gate": True,
            "enable_ger_overlap_gate": True,
            "enable_ger_acoustic_fallback": True,
        })
    else:
        raise ValueError(f"Unknown safe-core preset: {preset}")


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
            audio_path=row.get("audio"),
            mouth_roi_path=row.get("mouth_roi"),
            video_path=row.get("video"),
            manifest_row=dict(row),
        ))
    return turns


def _manifest_has_visual_turns(manifest: dict[str, Any]) -> bool:
    for row in manifest.get("turns", manifest.get("utterances", [])):
        path = row.get("mouth_roi") or row.get("video")
        if path and Path(path).exists():
            return True
    return False


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


def _load_stage2_checkpoints(
    pipe: AVSDGERPipeline,
    *,
    aligner_ckpt: str | None = None,
    ger_ckpt: str | None = None,
) -> None:
    if aligner_ckpt:
        path = Path(aligner_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"Missing aligner checkpoint: {path}")
        pipe.aligner.load_state_dict(torch.load(path, map_location=pipe.device))
        print(f"[stage2] loaded aligner checkpoint: {path}")

    if not ger_ckpt:
        return
    root = Path(ger_ckpt)
    if not root.exists():
        raise FileNotFoundError(f"Missing GER checkpoint path: {root}")

    projectors = root / "ger_projectors.pt" if root.is_dir() else root
    if projectors.exists() and projectors.is_file():
        state = torch.load(projectors, map_location=pipe.device)
        pipe.ger.qformer.load_state_dict(state["qformer"])
        pipe.ger.id_proj.load_state_dict(state["id_proj"])
        print(f"[stage2] loaded GER projectors: {projectors}")

    adapter_dir = root / "lora_adapter" if root.is_dir() else None
    if adapter_dir is not None and adapter_dir.exists() and pipe.ger._llm is not None:
        if hasattr(pipe.ger._llm, "load_adapter"):
            adapter_name = "stage2_eval"
            pipe.ger._llm.load_adapter(
                str(adapter_dir), adapter_name=adapter_name, is_trainable=False
            )
            pipe.ger._llm.set_adapter(adapter_name)
        else:
            from peft import PeftModel
            pipe.ger._llm = PeftModel.from_pretrained(
                pipe.ger._llm, str(adapter_dir), is_trainable=False
            )
        print(f"[stage2] loaded GER LoRA adapter: {adapter_dir}")


def _run_one(
    cfg: dict[str, Any],
    ablation_name: str,
    flags: dict[str, bool],
    manifest: dict[str, Any],
    pool_path: str | None,
    fresh_pool: bool,
    monitor: PowerMonitor | None,
    aligner_ckpt: str | None = None,
    ger_ckpt: str | None = None,
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
    _load_stage2_checkpoints(pipe, aligner_ckpt=aligner_ckpt, ger_ckpt=ger_ckpt)
    loaded_pool = False
    if pool_path and Path(pool_path).exists() and not fresh_pool:
        pipe.load_pool(pool_path)
        loaded_pool = True
    elif pool_path and Path(pool_path).exists() and fresh_pool:
        pipe.load_pool(pool_path)
        loaded_pool = True
        print(f"[pool] loaded fuser from {pool_path}; re-enrolling speakers from manifest")

    if fresh_pool or not loaded_pool:
        if fresh_pool:
            print(f"[pool] fresh-pool enabled; enrolling {ablation_name} from manifest")
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
    trace_summary = _summarize_traces(session.turns)

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
        "turn_debug": _build_turn_debug(session.turns),
        "trace_summary": trace_summary,
    }


def _build_turn_debug(turns) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in turns:
        dbg = dict(turn.debug or {})
        pipe_dbg = dbg.get("pipeline", {}) or {}
        final_dbg = pipe_dbg.get("final", {}) or {}
        asr_dbg = pipe_dbg.get("asr", {}) or {}
        visual_dbg = pipe_dbg.get("visual", {}) or {}
        c1_dbg = pipe_dbg.get("c1_effective", {}) or {}
        last_trace = turn.trace[-1] if turn.trace else {}
        rows.append({
            "summary": {
                "turn_id": turn.turn_id,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start,
                "ref_speaker": turn.ref_speaker,
                "hyp_speaker": turn.hyp_speaker,
                "speaker_correct": (
                    turn.ref_speaker == turn.hyp_speaker
                    if turn.ref_speaker is not None and turn.hyp_speaker is not None
                    else None
                ),
                "ref_text": turn.ref_text,
                "asr_top": asr_dbg.get("top"),
                "lip_hyp": visual_dbg.get("lip_hyp"),
                "final_text": turn.hyp_text,
                "confidence": turn.confidence,
                "s_acoustic": turn.s_acoustic,
                "iterations": turn.iterations,
                "pool_updated": turn.pool_updated,
                "fallback_applied": bool(last_trace.get("fallback_applied")),
                "fallback_reason": last_trace.get("fallback_reason"),
                "ger_mode": last_trace.get("ger_mode") or visual_dbg.get("effective_ger_mode"),
                "has_visual": last_trace.get("has_visual", visual_dbg.get("has_visual")),
                "top_ids": c1_dbg.get("top_ids"),
                "top_scores": c1_dbg.get("top_scores"),
                "av_consistency_raw": c1_dbg.get("av_consistency_raw"),
                "is_unknown": c1_dbg.get("is_unknown"),
            },
            "turn": dbg.get("turn", {}),
            "input": pipe_dbg.get("input", {}),
            "asr": asr_dbg,
            "visual": visual_dbg,
            "embeddings": pipe_dbg.get("embeddings", {}),
            "c1_initial": pipe_dbg.get("c1_initial", {}),
            "c1_effective": c1_dbg,
            "c2": pipe_dbg.get("c2", {}),
            "c3": pipe_dbg.get("c3", {}),
            "final": final_dbg,
            "trace": list(turn.trace or []),
        })
    return rows


def _summarize_traces(turns) -> dict[str, Any]:
    fallback_reasons: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    raw_artifacts: Counter[str] = Counter()
    fallback_turns = 0
    pool_updates = 0
    total_iters = 0
    total_turns = len(turns)
    artifact_terms = [
        "please provide",
        "i'm happy to help",
        "here is the corrected transcript",
        "audio hypothesis:",
        "corrected transcript:",
        "transcript provided",
        "speaker label",
    ]

    for turn in turns:
        if turn.pool_updated:
            pool_updates += 1
        total_iters += int(turn.iterations or 0)
        turn_had_fallback = False
        for item in turn.trace:
            decision = item.get("decision")
            if decision:
                decisions[str(decision)] += 1
            if item.get("fallback_applied"):
                turn_had_fallback = True
                reason = item.get("fallback_reason") or "<unknown>"
                fallback_reasons[str(reason)] += 1
            raw = str(item.get("raw_ger_text") or item.get("raw_generation") or "").lower()
            for term in artifact_terms:
                if term in raw:
                    raw_artifacts[term] += 1
        if turn_had_fallback:
            fallback_turns += 1

    return {
        "n_turns": total_turns,
        "fallback_turns": fallback_turns,
        "fallback_rate": fallback_turns / max(1, total_turns),
        "fallback_reasons": dict(fallback_reasons.most_common()),
        "decisions": dict(decisions.most_common()),
        "pool_updates": pool_updates,
        "mean_iterations": total_iters / max(1, total_turns),
        "raw_artifact_hits": dict(raw_artifacts.most_common()),
    }


def _write_debug_sidecars(
    results: list[dict[str, Any]],
    out_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    debug_dir = out_path.parent / f"{out_path.stem}_debug"
    compact_results: list[dict[str, Any]] = []
    for r in results:
        r_main = dict(r)
        turn_debug = r_main.pop("turn_debug", [])
        debug_path = debug_dir / f"{manifest_path.stem}.{r['ablation']}.debug.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({
                "manifest": str(manifest_path),
                "ablation": r["ablation"],
                "flags": r["flags"],
                "frontend": r["frontend"],
                "metrics": r["metrics"],
                "trace_summary": r["trace_summary"],
                "speaker_order": r["speaker_order"],
                "turns": turn_debug,
            }, f, indent=2, ensure_ascii=False)
        r_main["debug_path"] = str(debug_path)
        compact_results.append(r_main)
        print(f"[debug] wrote {debug_path}")
    return compact_results


def _run_manifest(
    cfg: dict[str, Any],
    manifest_path: Path,
    pool_path: str | None,
    fresh_pool: bool,
    monitor: PowerMonitor | None,
    only: list[str] | None,
    wb: WandbLogger,
    step_offset: int = 0,
    aligner_ckpt: str | None = None,
    ger_ckpt: str | None = None,
) -> tuple[list[dict[str, Any]], bool | None]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if str(cfg.get("ger", {}).get("mode", "audio_only")).lower() == "av" and not _manifest_has_visual_turns(manifest):
        print(
            f"[warning] {manifest_path.name}: no valid mouth_roi/video paths found; "
            "pipeline will run these turns as audio_only. Do not report this run as AV."
        )

    results: list[dict[str, Any]] = []
    for i, (name, flags) in enumerate(ABLATION_MATRIX):
        if only is not None and name not in only:
            continue
        print(f"\n=== {manifest_path.name} :: running ablation: {name}  flags={flags} ===")
        r = _run_one(
            cfg,
            name,
            flags,
            manifest,
            pool_path,
            fresh_pool,
            monitor,
            aligner_ckpt=aligner_ckpt,
            ger_ckpt=ger_ckpt,
        )
        print(json.dumps(r["metrics"], indent=2))
        results.append(r)

        prefix = f"manifest/{manifest_path.stem}/ablation/{name}"
        metric_prefix = f"metric/{name}"
        wb.log({
            f"{prefix}/sa_wer":     r["metrics"]["sa_wer"],
            f"{prefix}/wer":        r["metrics"]["wer"],
            f"{prefix}/scr":        r["metrics"]["scr"],
            f"{prefix}/av_sid_acc": r["metrics"]["av_sid_acc"],
            f"{prefix}/der":        r["metrics"]["der"],
            f"{prefix}/jer":        r["metrics"]["jer"],
            f"{prefix}/frontend":   r["frontend"]["profile"],
            f"{prefix}/fallback_rate": r["trace_summary"]["fallback_rate"],
            f"{prefix}/fallback_turns": r["trace_summary"]["fallback_turns"],
            f"{prefix}/pool_updates": r["trace_summary"]["pool_updates"],
            f"{metric_prefix}/sa_wer/{manifest_path.stem}": r["metrics"]["sa_wer"],
            f"{metric_prefix}/wer/{manifest_path.stem}": r["metrics"]["wer"],
            f"{metric_prefix}/scr/{manifest_path.stem}": r["metrics"]["scr"],
            f"{metric_prefix}/av_sid_acc/{manifest_path.stem}": r["metrics"]["av_sid_acc"],
            f"{metric_prefix}/der/{manifest_path.stem}": r["metrics"]["der"],
            f"{metric_prefix}/jer/{manifest_path.stem}": r["metrics"]["jer"],
            f"{metric_prefix}/fallback_rate/{manifest_path.stem}": r["trace_summary"]["fallback_rate"],
            f"{metric_prefix}/fallback_turns/{manifest_path.stem}": r["trace_summary"]["fallback_turns"],
            f"{metric_prefix}/pool_updates/{manifest_path.stem}": r["trace_summary"]["pool_updates"],
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
    p.add_argument(
        "--aligner-ckpt",
        default=None,
        help="Optional Stage-2 aligner checkpoint, e.g. checkpoints/stage2/aligner_stage2.pt.",
    )
    p.add_argument(
        "--ger-ckpt",
        default=None,
        help="Optional Stage-2 GER directory containing ger_projectors.pt and lora_adapter/.",
    )
    p.add_argument(
        "--fresh-pool",
        action="store_true",
        help=(
            "Re-enroll manifest speakers for every ablation run. If --pool exists, "
            "its trained fuser is loaded first, then speakers are enrolled from the manifest."
        ),
    )
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
        "--asr-backend",
        default=None,
        choices=["faster-whisper", "openai-whisper", "transformers"],
        help="Override cfg.asr.backend for eval without editing the YAML config.",
    )
    p.add_argument(
        "--frontend-profile",
        default=None,
        choices=_frontend_choices(),
        help="Override cfg.frontend.profile for reporting/experiment grouping.",
    )
    p.add_argument(
        "--safe-core-preset",
        default=None,
        choices=["no_ger_gate", "artifact_only", "text_gates", "full"],
        help=(
            "Apply a safe-core feedback preset on top of --config without "
            "changing backbone/model settings."
        ),
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
    if args.asr_backend is not None:
        cfg.setdefault("asr", {})["backend"] = args.asr_backend
        print(f"[eval_ablations] Override asr.backend -> {args.asr_backend}")
    if args.safe_core_preset is not None:
        _apply_safe_core_preset(cfg, args.safe_core_preset)
        print(f"[eval_ablations] Apply safe-core preset -> {args.safe_core_preset}")
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
            "fresh_pool": bool(args.fresh_pool),
            "safe_core_preset": args.safe_core_preset,
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
            args.fresh_pool,
            monitor,
            args.only,
            wb,
            step_offset=m_idx * max(1, ablations_per_manifest),
            aligner_ckpt=args.aligner_ckpt,
            ger_ckpt=args.ger_ckpt,
        )
        out_path = _output_path_for_manifest(args.out, manifest_path, multi=multi)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_for_payload = _write_debug_sidecars(results, out_path, manifest_path)
        payload = {
            "manifest": str(manifest_path),
            "frontend": frontend_meta,
            "results": results_for_payload,
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
                summary[f"summary_metric/{r['ablation']}/{k}/{stem}"] = v
            ts = r.get("trace_summary", {}) or {}
            for k in ("fallback_rate", "fallback_turns", "pool_updates", "mean_iterations"):
                if k in ts:
                    summary[f"summary/{stem}/{r['ablation']}/{k}"] = ts[k]
                    summary[f"summary_metric/{r['ablation']}/{k}/{stem}"] = ts[k]
    for stem, passed in spec_checks.items():
        if passed is not None:
            summary[f"summary/{stem}/spec_check_c3_gate_pass"] = bool(passed)
            summary[f"summary_metric/spec_check/c3_gate_pass/{stem}"] = bool(passed)
    wb.summary(summary)
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
