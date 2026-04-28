"""One-go training launcher for AVSD-GER.

This wraps the repo's Stage-1 and Stage-2 training scripts and writes a runtime
config under one_go/runs/. In stub mode the scripts exercise the training
control flow with synthetic tensors; in real mode they expect a real JSONL
manifest and working backbone weights.
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
    if not args.real:
        cfg["stub_backbones"] = True
        cfg["device"] = args.device or "cpu"
        cfg["training"]["stage1"]["epochs"] = args.stub_epochs
        cfg["training"]["stage2"]["epochs"] = args.stub_epochs
    else:
        if args.device:
            cfg["device"] = args.device
        if args.stage1_epochs is not None:
            cfg["training"]["stage1"]["epochs"] = args.stage1_epochs
        if args.stage2_epochs is not None:
            cfg["training"]["stage2"]["epochs"] = args.stage2_epochs
        if args.llm_quant:
            cfg.setdefault("ger", {})["llm_quant"] = args.llm_quant

    out = RUN_DIR / ("train_config_real.yaml" if args.real else "train_config_stub.yaml")
    _write_yaml(out, cfg)
    return out


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n[one_go/train] {' '.join(str(x) for x in cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Run AVSD-GER Stage-1/Stage-2 training.")
    p.add_argument("--stage", choices=["stage1", "stage2", "all"], default="stage1")
    p.add_argument("--real", action="store_true", help="Use real backbones and a real manifest.")
    p.add_argument("--device", default=None)
    p.add_argument("--llm-quant", choices=["auto", "fp16", "int8", "4bit"], default=None)
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run repo scripts with. Use the avsdger env python if needed.",
    )
    p.add_argument("--manifest", default=str(ROOT / "data" / "train_manifest.jsonl"))
    p.add_argument("--stage1-out", default=str(RUN_DIR / "stage1"))
    p.add_argument("--stage2-out", default=str(RUN_DIR / "stage2"))
    p.add_argument("--stub-epochs", type=int, default=1)
    p.add_argument("--stage1-epochs", type=int, default=None)
    p.add_argument("--stage2-epochs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config = make_runtime_config(args)
    py = args.python

    if args.stage in {"stage1", "all"}:
        run_cmd([
            py,
            "scripts/train_identity.py",
            "--config",
            str(config),
            "--manifest",
            args.manifest,
            "--out",
            args.stage1_out,
        ], dry_run=args.dry_run)

    if args.stage in {"stage2", "all"}:
        run_cmd([
            py,
            "scripts/train_stage2.py",
            "--config",
            str(config),
            "--manifest",
            args.manifest,
            "--out",
            args.stage2_out,
        ], dry_run=args.dry_run)

    print("\n[one_go/train] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
