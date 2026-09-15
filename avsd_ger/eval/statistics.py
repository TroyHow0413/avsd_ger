"""Cluster-aware uncertainty and paired comparisons for formal evaluation.

Point estimates remain corpus-micro scores. Confidence intervals resample
whole sessions, never turns. A meeting-series sensitivity analysis clusters
AMI's a/b/c/d sessions (which share participants) under the common prefix.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np


METRIC_ORDER = (
    "tcpwer_5s", "cpwer", "wer", "der", "jer",
    "sa_wer_custom", "scr", "av_sid_acc",
)


def meeting_series_id(meeting_id: str) -> str:
    """Collapse AMI sessions such as ES2011a-d to their participant series."""
    return re.sub(r"[a-d]$", "", str(meeting_id), flags=re.IGNORECASE)


def _nested_number(payload: dict[str, Any], *names: str) -> float | None:
    lowered = {str(key).lower().replace("_", " "): value for key, value in payload.items()}
    for name in names:
        value = lowered.get(name.lower().replace("_", " "))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _ratio(value: Any, numerator: Any, denominator: Any) -> dict[str, Any] | None:
    if not isinstance(value, (int, float)):
        return None
    if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
        return {
            "value": float(value), "numerator": float(numerator),
            "denominator": float(denominator), "aggregation": "ratio_of_sums",
        }
    return {
        "value": float(value), "numerator": float(value),
        "denominator": 1.0, "aggregation": "cluster_mean",
    }


def _meeteval(result: dict[str, Any], key: str) -> dict[str, Any] | None:
    score = (
        result.get("standard_metrics", {}).get("meeteval", {})
        .get("scores", {}).get(key, {})
    )
    return _ratio(score.get("error_rate"), score.get("errors"), score.get("length"))


def _wer(result: dict[str, Any]) -> dict[str, Any] | None:
    score = result.get("standard_metrics", {}).get("jiwer", {})
    errors = sum(float(score.get(key, 0)) for key in ("substitutions", "deletions", "insertions"))
    return _ratio(score.get("wer"), errors, score.get("reference_words"))


def _der(result: dict[str, Any]) -> dict[str, Any] | None:
    score = (
        result.get("standard_metrics", {}).get("pyannote", {})
        .get("scores", {}).get("der_collar_0s", {})
    )
    components = score.get("components", {}) if isinstance(score, dict) else {}
    total = _nested_number(components, "total")
    errors = sum(filter(lambda x: x is not None, [
        _nested_number(components, "false alarm"),
        _nested_number(components, "missed detection", "miss"),
        _nested_number(components, "confusion"),
    ]))
    value = score.get("value") if isinstance(score, dict) else None
    if value is not None and total is not None:
        return _ratio(value, errors, total)
    details = result.get("metric_details", {}).get("der", {})
    fallback_errors = sum(float(details.get(key, 0)) for key in ("miss", "false_alarm", "confusion"))
    return _ratio(result.get("metrics", {}).get("der"), fallback_errors, details.get("total_ref"))


def _jer(result: dict[str, Any]) -> dict[str, Any] | None:
    score = (
        result.get("standard_metrics", {}).get("pyannote", {})
        .get("scores", {}).get("jer_collar_0s", {})
    )
    components = score.get("components", {}) if isinstance(score, dict) else {}
    speaker_error = _nested_number(components, "speaker error", "speaker_error")
    speaker_count = _nested_number(components, "speaker count", "speaker_count")
    value = score.get("value") if isinstance(score, dict) else None
    if value is not None and speaker_error is not None and speaker_count is not None:
        return _ratio(value, speaker_error, speaker_count)
    per_speaker = result.get("metric_details", {}).get("jer", {}).get("per_speaker", {})
    if per_speaker:
        return _ratio(
            result.get("metrics", {}).get("jer"),
            sum(float(value) for value in per_speaker.values()),
            len(per_speaker),
        )
    return _ratio(result.get("metrics", {}).get("jer"), None, None)


def _task_ratio(
    result: dict[str, Any],
    *,
    value_key: str,
    detail_key: str,
    numerator: Callable[[dict[str, Any]], float],
    denominator_key: str,
) -> dict[str, Any] | None:
    details = result.get("metric_details", {}).get(detail_key, {})
    return _ratio(
        result.get("metrics", {}).get(value_key),
        numerator(details),
        details.get(denominator_key),
    )


def metric_observation(result: dict[str, Any], metric: str) -> dict[str, Any] | None:
    if metric == "tcpwer_5s":
        return _meeteval(result, "tcpwer_collar_5s")
    if metric == "cpwer":
        return _meeteval(result, "cpwer")
    if metric == "wer":
        return _wer(result)
    if metric == "der":
        return _der(result)
    if metric == "jer":
        return _jer(result)
    if metric == "sa_wer_custom":
        return _task_ratio(
            result, value_key="sa_wer", detail_key="sa_wer",
            numerator=lambda d: sum(float(d.get(k, 0)) for k in ("n_sub", "n_del", "n_ins", "n_spk_err")),
            denominator_key="n_ref_words",
        )
    if metric == "scr":
        return _task_ratio(
            result, value_key="scr", detail_key="scr",
            numerator=lambda d: float(d.get("n_spk_err", 0)),
            denominator_key="n_matched",
        )
    if metric == "av_sid_acc":
        return _task_ratio(
            result, value_key="av_sid_acc", detail_key="av_sid",
            numerator=lambda d: float(d.get("n_correct", 0)),
            denominator_key="n",
        )
    raise KeyError(metric)


def _aggregate(observations: list[dict[str, Any]], indices: np.ndarray | None = None) -> float:
    selected = observations if indices is None else [observations[int(i)] for i in indices]
    numerator = sum(item["numerator"] for item in selected)
    denominator = sum(item["denominator"] for item in selected)
    return float(numerator / denominator) if denominator else 0.0


def _bootstrap(
    observations: list[dict[str, Any]], *, samples: int, seed: int,
) -> dict[str, Any]:
    result = {
        "estimate": _aggregate(observations),
        "n_clusters": len(observations),
        "bootstrap_samples": int(samples), "seed": int(seed),
        "aggregation": observations[0]["aggregation"] if observations else None,
    }
    if len(observations) < 2:
        return {**result, "status": "insufficient", "ci95": None}
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        indices = rng.integers(0, len(observations), size=len(observations))
        values[draw] = _aggregate(observations, indices)
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        **result, "status": "ok", "ci95": [float(low), float(high)],
        "bootstrap_mean": float(values.mean()),
    }


def _collapse_clusters(
    observations: list[dict[str, Any]], cluster_ids: list[str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster, observation in zip(cluster_ids, observations):
        groups[cluster].append(observation)
    return [{
        "value": _aggregate(items),
        "numerator": sum(item["numerator"] for item in items),
        "denominator": sum(item["denominator"] for item in items),
        "aggregation": items[0]["aggregation"],
    } for _, items in sorted(groups.items())]


def _collect(raw_runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    collected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in raw_runs:
        meeting = Path(str(run.get("manifest", ""))).stem
        for result in run.get("results", []):
            ablation = str(result.get("ablation"))
            collected[ablation][meeting] = {
                metric: observation
                for metric in METRIC_ORDER
                if (observation := metric_observation(result, metric)) is not None
            }
    return collected


def build_statistics_report(
    raw_runs: list[dict[str, Any]], *, samples: int = 10_000, seed: int = 1337,
) -> dict[str, Any]:
    collected = _collect(raw_runs)
    report: dict[str, Any] = {
        "schema_version": 1,
        "independent_unit": "session/manifest",
        "primary_metric": "tcpwer_5s",
        "primary_comparison": "wo_c3 - full_model",
        "inference": "two-sided paired session-cluster bootstrap",
        "secondary_metrics": ["cpwer", "wer"],
        "exploratory_metrics": ["der", "jer", "sa_wer_custom", "scr", "av_sid_acc"],
        "bootstrap_samples": int(samples), "seed": int(seed),
        "ablations": {},
    }
    for ablation, meetings in collected.items():
        report["ablations"][ablation] = {}
        for metric in METRIC_ORDER:
            ids = sorted(mid for mid, values in meetings.items() if metric in values)
            observations = [meetings[mid][metric] for mid in ids]
            if not observations:
                report["ablations"][ablation][metric] = {
                    "status": "unavailable", "n_clusters": 0,
                }
                continue
            session = _bootstrap(observations, samples=samples, seed=seed)
            series_ids = [meeting_series_id(mid) for mid in ids]
            series_observations = _collapse_clusters(observations, series_ids)
            series = _bootstrap(series_observations, samples=samples, seed=seed + 1)
            if len(series_observations) < 5:
                series["warning"] = "fewer than five meeting-series clusters; sensitivity only"
            report["ablations"][ablation][metric] = {
                "session_cluster": session,
                "meeting_series_sensitivity": series,
                "n_sessions": len(ids),
                "n_meeting_series": len(set(series_ids)),
            }
    return report


COMPARISON_SPECS = (
    ("c3_topology", "full_model", "wo_c3"),
    ("identity_zero", "identity_normal", "zero_z_id"),
    ("identity_shuffle", "identity_normal", "shuffled_z_id"),
)


def _paired_bootstrap(
    reference: list[dict[str, Any]], challenger: list[dict[str, Any]],
    *, samples: int, seed: int,
) -> dict[str, Any]:
    ref_estimate = _aggregate(reference)
    challenger_estimate = _aggregate(challenger)
    result = {
        "reference_estimate": ref_estimate,
        "challenger_estimate": challenger_estimate,
        "delta_challenger_minus_reference": challenger_estimate - ref_estimate,
        "n_pairs": len(reference), "bootstrap_samples": samples, "seed": seed,
    }
    if len(reference) < 2:
        return {**result, "status": "insufficient", "ci95": None, "p_value_two_sided": None}
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        indices = rng.integers(0, len(reference), size=len(reference))
        deltas[draw] = _aggregate(challenger, indices) - _aggregate(reference, indices)
    low, high = np.percentile(deltas, [2.5, 97.5])
    less_equal = (np.count_nonzero(deltas <= 0.0) + 1) / (samples + 1)
    greater_equal = (np.count_nonzero(deltas >= 0.0) + 1) / (samples + 1)
    return {
        **result, "status": "ok", "ci95": [float(low), float(high)],
        "p_value_two_sided": float(min(1.0, 2.0 * min(less_equal, greater_equal))),
        "direction": (
            "challenger_worse" if low > 0 else
            "challenger_better" if high < 0 else "inconclusive"
        ),
    }


def _paired_series_observations(
    ids: list[str],
    reference: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    series: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for meeting, ref_item, challenger_item in zip(ids, reference, challenger):
        key = meeting_series_id(meeting)
        ref_group, challenger_group = series.setdefault(key, ([], []))
        ref_group.append(ref_item)
        challenger_group.append(challenger_item)
    collapsed_reference: list[dict[str, Any]] = []
    collapsed_challenger: list[dict[str, Any]] = []
    for key in sorted(series):
        ref_group, challenger_group = series[key]
        collapsed_reference.extend(_collapse_clusters(ref_group, [key] * len(ref_group)))
        collapsed_challenger.extend(
            _collapse_clusters(challenger_group, [key] * len(challenger_group))
        )
    return collapsed_reference, collapsed_challenger


def build_paired_comparisons(
    raw_runs: list[dict[str, Any]], *, samples: int = 10_000, seed: int = 1337,
) -> dict[str, Any]:
    collected = _collect(raw_runs)
    output: dict[str, Any] = {
        "schema_version": 1,
        "delta_definition": "challenger - reference; positive is worse for error metrics",
        "bootstrap_unit": "paired session/manifest",
        "bootstrap_samples": samples, "seed": seed,
        "comparisons": {},
    }
    for name, reference_name, challenger_name in COMPARISON_SPECS:
        if reference_name not in collected or challenger_name not in collected:
            output["comparisons"][name] = {
                "status": "not_run", "reference": reference_name,
                "challenger": challenger_name,
            }
            continue
        result: dict[str, Any] = {
            "reference": reference_name, "challenger": challenger_name,
            "metrics": {},
        }
        common = sorted(set(collected[reference_name]) & set(collected[challenger_name]))
        for metric in METRIC_ORDER:
            ids = [
                mid for mid in common
                if metric in collected[reference_name][mid]
                and metric in collected[challenger_name][mid]
            ]
            if not ids:
                result["metrics"][metric] = {"status": "unavailable", "n_pairs": 0}
                continue
            reference = [collected[reference_name][mid][metric] for mid in ids]
            challenger = [collected[challenger_name][mid][metric] for mid in ids]
            session_result = _paired_bootstrap(
                reference, challenger, samples=samples, seed=seed,
            )
            series_reference, series_challenger = _paired_series_observations(
                ids, reference, challenger,
            )
            series_result = _paired_bootstrap(
                series_reference, series_challenger, samples=samples, seed=seed + 1,
            )
            if len(series_reference) < 5:
                series_result["warning"] = (
                    "fewer than five meeting-series pairs; sensitivity only"
                )
            result["metrics"][metric] = {
                "session_cluster": session_result,
                "meeting_series_sensitivity": series_result,
                "n_sessions": len(ids),
                "n_meeting_series": len(series_reference),
            }
        if name == "c3_topology":
            primary = result["metrics"].get("tcpwer_5s", {}).get("session_cluster", {})
            direction = primary.get("direction")
            if direction is not None:
                result["selection_gate"] = {
                    "metric": "tcpwer_5s",
                    "rule": (
                        "select a topology only when the paired session-cluster 95% CI "
                        "for wo_c3 - full_model excludes zero"
                    ),
                    "decision": (
                        "full_model" if direction == "challenger_worse" else
                        "wo_c3" if direction == "challenger_better" else
                        "inconclusive"
                    ),
                }
        elif name in {"identity_zero", "identity_shuffle"}:
            primary = result["metrics"].get("tcpwer_5s", {}).get("session_cluster", {})
            direction = primary.get("direction")
            if direction is not None:
                result["causal_gate"] = {
                    "metric": "tcpwer_5s",
                    "rule": (
                        "identity use is supported when the paired 95% CI for "
                        "intervention - identity_normal is strictly above zero"
                    ),
                    "decision": (
                        "supports_identity_conditioning"
                        if direction == "challenger_worse" else
                        "intervention_better_review_identity_path"
                        if direction == "challenger_better" else
                        "inconclusive"
                    ),
                }
        output["comparisons"][name] = result
    return output
