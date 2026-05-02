"""One-go feasibility runner for AVSD-GER.

This is a thin orchestration layer over the repo's existing scripts. By
default it runs in stub mode so the full C1 -> C2 -> C3 wiring can be checked
without downloading Whisper, AV-HuBERT, ECAPA, InsightFace, or Llama weights.
Use --real only after the real-model prerequisites are ready.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ONE_GO = Path(__file__).resolve().parent
RUN_DIR = ONE_GO / "runs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def make_runtime_config(args: argparse.Namespace) -> Path:
    cfg = _load_yaml(ROOT / "configs" / "default.yaml")
    if args.frontend_profile:
        cfg.setdefault("frontend", {})["profile"] = args.frontend_profile
    if not args.real:
        cfg["stub_backbones"] = True
        cfg["device"] = args.device or "cpu"
        cfg.setdefault("ger", {})["llm_quant"] = "auto"
    else:
        if args.device:
            cfg["device"] = args.device
        if args.llm_quant:
            cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant

    out = RUN_DIR / ("config_real.yaml" if args.real else "config_stub.yaml")
    _write_yaml(out, cfg)
    return out


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"\n[one_go] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Run AVSD-GER feasibility checks from one folder.")
    p.add_argument(
        "--mode",
        choices=["smoke", "eval", "all"],
        default="all",
        help="smoke=enroll+single utterance, eval=ablation table, all=both.",
    )
    p.add_argument("--real", action="store_true", help="Use real backbones from configs/default.yaml.")
    p.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    p.add_argument("--llm-quant", choices=["auto", "fp16", "int8", "4bit"], default=None)
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run repo scripts with. Use the avsdger env python if needed.",
    )
    p.add_argument("--manifest", default=str(ROOT / "data" / "sample_manifest.json"))
    p.add_argument("--session-manifest", default=str(ROOT / "data" / "sample_session_manifest.json"))
    p.add_argument("--utt", default="utt_0001")
    p.add_argument("--pool", default=str(RUN_DIR / "identity_pool.pt"))
    p.add_argument("--ablation-out", default=str(RUN_DIR / "ablation_report.json"))
    p.add_argument(
        "--frontend-profile",
        choices=[
            "oracle_turns",
            "common_pyannote_lightasd",
            "strong_sortformer_talknet",
            "degraded_pyannote",
        ],
        default=None,
        help="Frontend profile metadata for eval reports/W&B.",
    )
    p.add_argument("--wandb-project", default=None, help="Forwarded to eval_ablations.py.")
    p.add_argument("--wandb-run-name", default=None, help="Forwarded to eval_ablations.py.")
    p.add_argument("--no-wandb", action="store_true", help="Forwarded to eval_ablations.py.")
    p.add_argument("--with-power", action="store_true", help="Enable power monitor during eval.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = p.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config = make_runtime_config(args)

    py = args.python
    pool = Path(args.pool)

    if args.mode in {"smoke", "all"}:
        run_cmd([
            py,
            "one_go/c1_enroll.py",
            "--config",
            str(config),
            "--manifest",
            args.manifest,
            "--out-pool",
            str(pool),
        ], dry_run=args.dry_run)
        run_cmd([
            py,
            "scripts/run_sample.py",
            "--config",
            str(config),
            "--manifest",
            args.manifest,
            "--utt",
            args.utt,
            "--pool",
            str(pool),
        ], dry_run=args.dry_run)

    if args.mode in {"eval", "all"}:
        if not pool.exists() and not args.dry_run:
            run_cmd([
                py,
                "one_go/c1_enroll.py",
                "--config",
                str(config),
                "--manifest",
                args.manifest,
                "--out-pool",
                str(pool),
            ], dry_run=False)
        eval_cmd = [
            py,
            "scripts/eval_ablations.py",
            "--config",
            str(config),
            "--manifest",
            args.session_manifest,
            "--pool",
            str(pool),
            "--out",
            args.ablation_out,
        ]
        if args.frontend_profile:
            eval_cmd.extend(["--frontend-profile", args.frontend_profile])
        if args.wandb_project:
            eval_cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_run_name:
            eval_cmd.extend(["--wandb-run-name", args.wandb_run_name])
        if args.no_wandb:
            eval_cmd.append("--no-wandb")
        if not args.with_power:
            eval_cmd.append("--no-power")
        run_cmd(eval_cmd, dry_run=args.dry_run)

    print("\n[one_go] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
