"""Pure statistical checks for C3 ablation results."""
from __future__ import annotations

from typing import Any

import numpy as np


def c3_cluster_bootstrap_spec_check(
    runs: list[dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 1337,
) -> dict[str, Any]:
    """Paired manifest-cluster bootstrap for the C3 structural-safety claim."""
    pairs: list[dict[str, Any]] = []
    for run in runs:
        by_name = {result["ablation"]: result for result in run.get("results", [])}
        if "wo_c3" not in by_name or "c3_wo_conf_gates" not in by_name:
            continue
        base = float(by_name["wo_c3"]["metrics"]["sa_wer"])
        ungated = float(by_name["c3_wo_conf_gates"]["metrics"]["sa_wer"])
        pairs.append({
            "manifest": str(run.get("manifest", "")),
            "wo_c3_sa_wer": base,
            "c3_wo_conf_gates_sa_wer": ungated,
            "delta": ungated - base,
        })
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "c3_wo_conf_gates - wo_c3 canonical SA-WER",
        "bootstrap_unit": "manifest",
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }
    if len(pairs) < 2:
        return {**result, "status": "insufficient", "pass": None}
    diffs = np.asarray([pair["delta"] for pair in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    if low > 0.0:
        status, passed = "degraded", True
    elif high < 0.0:
        status, passed = "improved", False
    else:
        status, passed = "inconclusive", None
    return {
        **result,
        "mean_delta": float(diffs.mean()),
        "ci95": [float(low), float(high)],
        "status": status,
        "pass": passed,
    }

