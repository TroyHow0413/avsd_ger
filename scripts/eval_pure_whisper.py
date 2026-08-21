"""Pure Whisper dev/test evaluator.

This is intentionally outside AVSD-GER's pipeline:
  - no C1 identity pool
  - no C2 GER
  - no C3 closed-loop feedback
  - no AV-HuBERT / Llama loading

It runs Whisper 1-best transcription on manifest audio and reports text WER
plus word accuracy (1 - WER).

Examples:
    python scripts/eval_pure_whisper.py \
        --manifest data/ami_dev/manifests \
        --backend faster-whisper \
        --model large-v3 \
        --out out/dev_pure_whisper_faster.json

    python scripts/eval_pure_whisper.py \
        --manifest data/ami_dev/manifests \
        --backend openai-whisper \
        --model large-v3 \
        --out out/dev_pure_whisper_openai.json
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.wandb_logger import WandbLogger, add_wandb_args  # noqa: E402
from avsd_ger.eval.metrics import compute_sa_wer, compute_scr  # noqa: E402
from avsd_ger.eval.session import SessionTurnResult  # noqa: E402


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


def _project_path(path: str | None) -> Path:
    if not path:
        raise FileNotFoundError("Missing audio path in manifest row.")
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
    raise FileNotFoundError(f"Audio path does not exist: {path!r}")


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


def _row_audio(row: dict[str, Any]) -> str | None:
    return row.get("audio") or row.get("wav_path") or row.get("audio_wav")


def _row_ref(row: dict[str, Any]) -> str:
    return str(row.get("ref_text") or row.get("target") or row.get("text") or "")


def _row_id(row: dict[str, Any], i: int) -> str:
    return str(row.get("turn_id") or row.get("utt_id") or row.get("id") or f"utt_{i:06d}")


def _row_speaker(row: dict[str, Any], source: str) -> str | None:
    if source == "none":
        return None
    if source == "ref_speaker":
        return row.get("ref_speaker") or row.get("speaker_id")
    if source == "frontend_speaker":
        # Degraded manifests only set frontend_speaker on flipped turns; unflipped
        # turns inherit the reference/original frontend speaker.
        return row.get("frontend_speaker") or row.get("ref_speaker") or row.get("speaker_id")
    return row.get(source)


def _tokens(text: str) -> list[str]:
    # AMI references contain punctuation as separated tokens. For a Whisper-only
    # ASR baseline, score lexical words case-insensitively and ignore punctuation.
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


@dataclass
class UtteranceResult:
    manifest: str
    utt_id: str
    audio: str
    ref_text: str
    hyp_text: str
    ref_speaker: str | None
    hyp_speaker: str | None
    n_ref_words: int
    n_sub: int
    n_del: int
    n_ins: int
    wer: float


class PureWhisper:
    def __init__(
        self,
        backend: str,
        model: str,
        device: str,
        compute_type: str,
        language: str | None,
        openai_fp16: bool,
    ):
        self.backend = backend
        self.language = language
        self.openai_fp16 = openai_fp16
        if backend == "faster-whisper":
            from faster_whisper import WhisperModel

            self.model = WhisperModel(model, device=device, compute_type=compute_type)
        elif backend == "openai-whisper":
            import whisper

            self.model = whisper.load_model(model, device=device)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def transcribe(self, audio_path: Path, beam_size: int) -> str:
        if self.backend == "faster-whisper":
            segments, _info = self.model.transcribe(
                str(audio_path),
                beam_size=beam_size,
                temperature=0.0,
                language=self.language,
                condition_on_previous_text=False,
            )
            return "".join(seg.text for seg in segments).strip()

        kwargs: dict[str, Any] = {
            "beam_size": beam_size,
            "temperature": 0.0,
            "language": self.language,
            "fp16": self.openai_fp16,
            "verbose": False,
            "condition_on_previous_text": False,
        }
        result = self.model.transcribe(str(audio_path), **kwargs)
        return str(result.get("text", "")).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Manifest .json/.jsonl, directory, or glob.")
    p.add_argument("--backend", choices=["faster-whisper", "openai-whisper"], default="faster-whisper")
    p.add_argument("--model", default="large-v3", help="Model name or local model path.")
    p.add_argument("--device", default="cuda", help="cuda, cpu, or backend-supported device.")
    p.add_argument("--compute-type", default="int8_float16", help="faster-whisper compute type.")
    p.add_argument(
        "--openai-fp16",
        action="store_true",
        help="Use fp16=True for openai-whisper. Ignored by faster-whisper.",
    )
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--language", default=None)
    p.add_argument("--limit", type=int, default=None, help="Optional quick smoke-test limit.")
    p.add_argument(
        "--diarization-source",
        default="none",
        help=(
            "Optional manifest field used as hypothesis speaker labels for "
            "Whisper+diarization SA-WER. Use 'ref_speaker' for oracle diarization, "
            "'frontend_speaker' for degraded/frontend labels, or 'none' for text-only WER."
        ),
    )
    p.add_argument("--out", default=str(ROOT / "out/pure_whisper_eval.json"))
    add_wandb_args(p)
    args = p.parse_args()

    manifests = _resolve_paths(args.manifest)
    wb = WandbLogger.from_args(
        args,
        default_project="avsd-ger",
        default_run_name=f"pure-whisper-{args.backend}-{Path(args.manifest).stem}",
        job_type="eval-pure-whisper",
        config={
            "baseline": "pure_whisper",
            "pipeline": "none",
            "uses_c1": False,
            "uses_c2": False,
            "uses_c3": False,
            "backend": args.backend,
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type if args.backend == "faster-whisper" else None,
            "openai_fp16": bool(args.openai_fp16) if args.backend == "openai-whisper" else None,
            "beam_size": args.beam_size,
            "language": args.language,
            "manifest": args.manifest,
            "n_manifests": len(manifests),
            "limit": args.limit,
            "diarization_source": args.diarization_source,
            "out": args.out,
        },
    )
    runner = PureWhisper(
        backend=args.backend,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        openai_fp16=bool(args.openai_fp16),
    )

    results: list[UtteranceResult] = []
    speaker_turns: list[SessionTurnResult] = []
    total_ref = total_sub = total_del = total_ins = 0
    seen = 0
    for manifest_path in manifests:
        for i, row in enumerate(_iter_rows(manifest_path)):
            if args.limit is not None and seen >= args.limit:
                break
            audio_path = _project_path(_row_audio(row))
            ref_text = _row_ref(row)
            ref_speaker = row.get("ref_speaker") or row.get("speaker_id")
            hyp_speaker = _row_speaker(row, args.diarization_source)
            hyp_text = runner.transcribe(audio_path, beam_size=args.beam_size)
            ref_toks = _tokens(ref_text)
            hyp_toks = _tokens(hyp_text)
            n_sub, n_del, n_ins = _edit_counts(ref_toks, hyp_toks)
            n_ref = len(ref_toks)
            wer = (n_sub + n_del + n_ins) / n_ref if n_ref else 0.0
            results.append(UtteranceResult(
                manifest=str(manifest_path),
                utt_id=_row_id(row, i),
                audio=str(audio_path),
                ref_text=ref_text,
                hyp_text=hyp_text,
                ref_speaker=str(ref_speaker) if ref_speaker is not None else None,
                hyp_speaker=str(hyp_speaker) if hyp_speaker is not None else None,
                n_ref_words=n_ref,
                n_sub=n_sub,
                n_del=n_del,
                n_ins=n_ins,
                wer=wer,
            ))
            if args.diarization_source != "none":
                speaker_turns.append(SessionTurnResult(
                    turn_id=_row_id(row, i),
                    start=float(row.get("start", seen)),
                    end=float(row.get("end", seen + 1)),
                    hyp_text=hyp_text,
                    hyp_speaker=str(hyp_speaker) if hyp_speaker is not None else None,
                    confidence=0.0,
                    s_acoustic=None,
                    iterations=1,
                    pool_updated=False,
                    ref_text=ref_text,
                    ref_speaker=str(ref_speaker) if ref_speaker is not None else None,
                ))
            total_ref += n_ref
            total_sub += n_sub
            total_del += n_del
            total_ins += n_ins
            seen += 1
            if seen % 25 == 0:
                cur_wer = (total_sub + total_del + total_ins) / max(1, total_ref)
                print(f"[pure-whisper] {seen} utterances, WER={cur_wer:.4f}", flush=True)
                wb.log({
                    "running/wer": cur_wer,
                    "running/word_accuracy": 1.0 - cur_wer,
                    "running/n_utterances": seen,
                    "running/n_ref_words": total_ref,
                    "running/n_sub": total_sub,
                    "running/n_del": total_del,
                    "running/n_ins": total_ins,
                }, step=seen)
        if args.limit is not None and seen >= args.limit:
            break

    total_err = total_sub + total_del + total_ins
    wer = total_err / total_ref if total_ref else 0.0
    speaker_metrics: dict[str, Any] = {}
    if args.diarization_source != "none":
        metric_language = args.language or "en"
        sa_wer, sa_details = compute_sa_wer(
            speaker_turns, language=metric_language
        )
        scr, scr_details = compute_scr(speaker_turns, language=metric_language)
        speaker_metrics = {
            "sa_wer": sa_wer,
            "speaker_attributed_word_accuracy": 1.0 - sa_wer,
            "scr": scr,
            "diarization_source": args.diarization_source,
            "n_speaker_turns": len(speaker_turns),
            "details": {
                "sa_wer": sa_details,
                "scr": scr_details,
            },
        }

    payload = {
        "backend": args.backend,
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type if args.backend == "faster-whisper" else None,
        "openai_fp16": bool(args.openai_fp16) if args.backend == "openai-whisper" else None,
        "beam_size": args.beam_size,
        "language": args.language,
        "manifest": args.manifest,
        "n_manifests": len(manifests),
        "n_utterances": len(results),
        "diarization_source": args.diarization_source,
        "metrics": {
            "wer": wer,
            "word_accuracy": 1.0 - wer,
            "n_ref_words": total_ref,
            "n_sub": total_sub,
            "n_del": total_del,
            "n_ins": total_ins,
            **{k: v for k, v in speaker_metrics.items() if k != "details"},
        },
        "speaker_metrics": speaker_metrics,
        "utterances": [asdict(x) for x in results],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2), flush=True)
    print(f"[wrote] {out}", flush=True)
    wb.summary({
        "summary/wer": wer,
        "summary/word_accuracy": 1.0 - wer,
        "summary/n_ref_words": total_ref,
        "summary/n_utterances": len(results),
        "summary/n_sub": total_sub,
        "summary/n_del": total_del,
        "summary/n_ins": total_ins,
        **({
            "summary/sa_wer": speaker_metrics["sa_wer"],
            "summary/scr": speaker_metrics["scr"],
            "summary/speaker_attributed_word_accuracy": speaker_metrics["speaker_attributed_word_accuracy"],
        } if speaker_metrics else {}),
        "summary/out": str(out),
    })
    wb.log({
        "final/wer": wer,
        "final/word_accuracy": 1.0 - wer,
        "final/n_ref_words": total_ref,
        "final/n_utterances": len(results),
        **({
            "final/sa_wer": speaker_metrics["sa_wer"],
            "final/scr": speaker_metrics["scr"],
        } if speaker_metrics else {}),
    }, step=seen)
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
