"""Thin Weights & Biases adapter.

Goals:
    * One place to read CLI flags from any script (`add_wandb_args`).
    * Graceful no-op when `wandb` is not installed or `--no-wandb` is set.
    * Identical call surface (`init`, `log`, `summary`, `finish`) regardless
      of whether logging is live, so callers don't need their own try/except.

Usage:
    from avsd_ger.wandb_logger import add_wandb_args, WandbLogger

    parser = argparse.ArgumentParser()
    ...
    add_wandb_args(parser)
    args = parser.parse_args()

    wb = WandbLogger.from_args(args, default_project="avsd-ger", config={"lr": 1e-4})
    wb.log({"loss/total": 0.42}, step=10)
    wb.summary({"final/sa_wer": 0.18})
    wb.finish()
"""
from __future__ import annotations

import argparse
from typing import Any


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add a uniform set of W&B flags to any script's argparse.

    Flags:
        --wandb-project   (default: avsd-ger)
        --wandb-run-name  (default: scripts auto-derive from manifest/stage if unset)
        --wandb-entity    (default: $WANDB_ENTITY env or wandb's default)
        --wandb-tags      (space-separated; e.g. --wandb-tags stub stage1)
        --no-wandb        kill switch -- disables logging without removing other flags
    """
    g = parser.add_argument_group("Weights & Biases")
    g.add_argument("--wandb-project", default="avsd-ger",
                   help="W&B project name (default: avsd-ger)")
    g.add_argument("--wandb-run-name", default=None,
                   help="W&B run name; if omitted, auto-derived from script + timestamp")
    g.add_argument("--wandb-entity", default=None,
                   help="W&B team/user (default: WANDB_ENTITY env or your default)")
    g.add_argument("--wandb-tags", nargs="+", default=None,
                   help="space-separated tags, e.g. --wandb-tags stub stage1")
    g.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging entirely")


class WandbLogger:
    """No-op if wandb missing or --no-wandb; otherwise a thin wandb.* wrapper."""

    def __init__(self, run: Any | None):
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        default_project: str = "avsd-ger",
        default_run_name: str | None = None,
        config: dict[str, Any] | None = None,
        job_type: str | None = None,
    ) -> "WandbLogger":
        if getattr(args, "no_wandb", False):
            return cls(None)
        try:
            import wandb
        except ImportError:
            print("[wandb] package not installed -- logging disabled "
                  "(install with `pip install wandb` to enable)")
            return cls(None)

        run = wandb.init(
            project=getattr(args, "wandb_project", None) or default_project,
            name=getattr(args, "wandb_run_name", None) or default_run_name,
            entity=getattr(args, "wandb_entity", None),
            tags=getattr(args, "wandb_tags", None),
            config=config or {},
            job_type=job_type,
        )
        return cls(run)

    # ------------------------------------------------------------------ logging
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as e:
            print(f"[wandb] log failed (step={step}): {e}")

    def summary(self, metrics: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            for k, v in metrics.items():
                self._run.summary[k] = v
        except Exception as e:
            print(f"[wandb] summary failed: {e}")

    def watch(self, model, log: str = "gradients", log_freq: int = 200) -> None:
        if self._run is None:
            return
        try:
            import wandb
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            print(f"[wandb] watch failed: {e}")

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")
        self._run = None
