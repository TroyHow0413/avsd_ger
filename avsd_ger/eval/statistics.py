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

# Deltas are always reported as challenger - reference.  Most metrics are
# error rates and therefore improve when the delta is negative.  AV-SID is an
# accuracy, so its interpretation is intentionally reversed without changing
# the numeric delta, confidence interval, or randomization test.
METRIC_OPTIMIZATION = {
    metric: ("maximize" if metric == "av_sid_acc" else "minimize")
    for metric in METRIC_ORDER
}

ABLATION_ALIASES = {
    # Raw eval summaries use the compatibility ID while formal artifacts use
    # the canonical spelling.  Normalize at ingestion so both paths produce
    # identical comparisons.  The historical singular c3_wo_conf_gate is
    # deliberately not aliased because it had update-gate-only semantics.
    "c3_wo_conf_gates": "c3_wo_confidence_gates",
}


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
        if not meeting:
            raise ValueError("Every scoring run must have a non-empty manifest/meeting ID")
        for result in run.get("results", []):
            raw_ablation = str(result.get("ablation"))
            if not raw_ablation or raw_ablation == "None":
                raise ValueError(f"{meeting}: every result must have an ablation ID")
            ablation = ABLATION_ALIASES.get(raw_ablation, raw_ablation)
            if meeting in collected[ablation]:
                raise ValueError(
                    f"Duplicate meeting/ablation result after canonicalization: "
                    f"{meeting}/{ablation} (input={raw_ablation}); "
                    "refusing to overwrite paired scoring input"
                )
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
    ("c3_decision_gate", "full_model", "c3_wo_decision_gate"),
    ("c3_update_gate", "full_model", "c3_wo_update_gate"),
    ("c3_both_gates", "full_model", "c3_wo_confidence_gates"),
    (
        "c3_decision_gate_incremental",
        "c3_wo_update_gate",
        "c3_wo_confidence_gates",
    ),
    (
        "c3_update_gate_incremental",
        "c3_wo_decision_gate",
        "c3_wo_confidence_gates",
    ),
    ("identity_zero", "identity_normal", "zero_z_id"),
    ("identity_shuffle", "identity_normal", "shuffled_z_id"),
    (
        "visual_lip_hyp",
        "selected_visual_baseline",
        "selected_wo_lip_hyp",
    ),
    (
        "visual_av_context",
        "selected_visual_baseline",
        "selected_wo_av_context",
    ),
)


def _paired_bootstrap(
    reference: list[dict[str, Any]], challenger: list[dict[str, Any]],
    *, samples: int, seed: int, optimization: str,
) -> dict[str, Any]:
    if optimization not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported metric optimization direction: {optimization!r}")
    ref_estimate = _aggregate(reference)
    challenger_estimate = _aggregate(challenger)
    result = {
        "reference_estimate": ref_estimate,
        "challenger_estimate": challenger_estimate,
        "delta_challenger_minus_reference": challenger_estimate - ref_estimate,
        "n_pairs": len(reference), "bootstrap_samples": samples, "seed": seed,
        "optimization": optimization,
    }
    if len(reference) < 2:
        return {**result, "status": "insufficient", "ci95": None, "p_value_two_sided": None}
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        indices = rng.integers(0, len(reference), size=len(reference))
        deltas[draw] = _aggregate(challenger, indices) - _aggregate(reference, indices)
    low, high = np.percentile(deltas, [2.5, 97.5])
    # The percentile bootstrap is used for the confidence interval only.  A
    # bootstrap distribution is centered on the observed effect, so counting
    # how often it crosses zero is not a valid null-hypothesis p-value.
    observed = abs(challenger_estimate - ref_estimate)
    permutation_rng = np.random.default_rng(seed + 2)
    extreme = 0
    for _ in range(samples):
        swap = permutation_rng.integers(0, 2, size=len(reference), dtype=np.int8)
        permuted_reference = [
            challenger[index] if swap[index] else reference[index]
            for index in range(len(reference))
        ]
        permuted_challenger = [
            reference[index] if swap[index] else challenger[index]
            for index in range(len(reference))
        ]
        delta = _aggregate(permuted_challenger) - _aggregate(permuted_reference)
        extreme += int(abs(delta) >= observed - 1e-15)
    if low > 0:
        direction = "challenger_better" if optimization == "maximize" else "challenger_worse"
    elif high < 0:
        direction = "challenger_worse" if optimization == "maximize" else "challenger_better"
    else:
        direction = "inconclusive"
    return {
        **result, "status": "ok", "ci95": [float(low), float(high)],
        "p_value_two_sided": float((extreme + 1) / (samples + 1)),
        "p_value_method": "paired randomization by within-session label swap",
        "permutation_samples": int(samples),
        "direction": direction,
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


def _factorial_interaction_bootstrap(
    full: list[dict[str, Any]],
    decision_off: list[dict[str, Any]],
    update_off: list[dict[str, Any]],
    both_off: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    optimization: str,
) -> dict[str, Any]:
    """Paired 2x2 interaction for disabling the decision and update gates."""
    if optimization not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported metric optimization direction: {optimization!r}")

    def estimate(indices: np.ndarray | None = None) -> float:
        return float(
            _aggregate(both_off, indices)
            - _aggregate(decision_off, indices)
            - _aggregate(update_off, indices)
            + _aggregate(full, indices)
        )

    observed = estimate()
    result = {
        "estimate": observed,
        "formula": "both_off - decision_off - update_off + full_model",
        "n_pairs": len(full),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "optimization": optimization,
    }
    if len(full) < 2:
        return {**result, "status": "insufficient", "ci95": None}

    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        indices = rng.integers(0, len(full), size=len(full))
        values[draw] = estimate(indices)
    low, high = np.percentile(values, [2.5, 97.5])
    if low > 0:
        direction = "positive_interaction"
        interpretation = (
            "joint_disable_more_favorable_than_additive"
            if optimization == "maximize"
            else "joint_disable_less_favorable_than_additive"
        )
    elif high < 0:
        direction = "negative_interaction"
        interpretation = (
            "joint_disable_less_favorable_than_additive"
            if optimization == "maximize"
            else "joint_disable_more_favorable_than_additive"
        )
    else:
        direction = "inconclusive"
        interpretation = "inconclusive"
    return {
        **result,
        "status": "ok",
        "ci95": [float(low), float(high)],
        "bootstrap_mean": float(values.mean()),
        "direction": direction,
        "performance_interpretation": interpretation,
    }


def _build_c3_gate_interaction(
    collected: dict[str, dict[str, dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    conditions = {
        "full_model": "full_model",
        "decision_off": "c3_wo_decision_gate",
        "update_off": "c3_wo_update_gate",
        "both_off": "c3_wo_confidence_gates",
    }
    missing = [name for name in conditions.values() if name not in collected]
    if missing:
        return {
            "status": "not_run",
            "conditions": conditions,
            "missing_ablations": missing,
        }

    meeting_sets = {
        label: set(collected[ablation])
        for label, ablation in conditions.items()
    }
    if len({frozenset(meetings) for meetings in meeting_sets.values()}) != 1:
        return {
            "status": "invalid_unpaired_meetings",
            "conditions": conditions,
            "meetings_by_condition": {
                label: sorted(meetings)
                for label, meetings in meeting_sets.items()
            },
        }

    common = sorted(meeting_sets["full_model"])
    result: dict[str, Any] = {
        "status": "ok",
        "conditions": conditions,
        "formula": "both_off - decision_off - update_off + full_model",
        "metrics": {},
    }
    for metric in METRIC_ORDER:
        ids = [
            meeting for meeting in common
            if all(
                metric in collected[ablation][meeting]
                for ablation in conditions.values()
            )
        ]
        if len(ids) != len(common):
            result["metrics"][metric] = {
                "status": "invalid_incomplete_metric_pairs",
                "n_pairs": len(ids),
                "expected_pairs": len(common),
                "missing_meetings": sorted(set(common) - set(ids)),
            }
            continue
        observations = {
            label: [collected[ablation][meeting][metric] for meeting in ids]
            for label, ablation in conditions.items()
        }
        session = _factorial_interaction_bootstrap(
            observations["full_model"],
            observations["decision_off"],
            observations["update_off"],
            observations["both_off"],
            samples=samples,
            seed=seed,
            optimization=METRIC_OPTIMIZATION[metric],
        )
        series_ids = [meeting_series_id(meeting) for meeting in ids]
        series = {
            label: _collapse_clusters(values, series_ids)
            for label, values in observations.items()
        }
        series_result = _factorial_interaction_bootstrap(
            series["full_model"],
            series["decision_off"],
            series["update_off"],
            series["both_off"],
            samples=samples,
            seed=seed + 1,
            optimization=METRIC_OPTIMIZATION[metric],
        )
        if len(series["full_model"]) < 5:
            series_result["warning"] = (
                "fewer than five meeting-series pairs; sensitivity only"
            )
        result["metrics"][metric] = {
            "session_cluster": session,
            "meeting_series_sensitivity": series_result,
            "n_sessions": len(ids),
            "n_meeting_series": len(series["full_model"]),
        }
    return result


def build_paired_comparisons(
    raw_runs: list[dict[str, Any]], *, samples: int = 10_000, seed: int = 1337,
) -> dict[str, Any]:
    collected = _collect(raw_runs)
    output: dict[str, Any] = {
        "schema_version": 1,
        "delta_definition": "challenger - reference",
        "metric_optimization": dict(METRIC_OPTIMIZATION),
        "bootstrap_unit": "paired session/manifest",
        "bootstrap_samples": samples, "seed": seed,
        "comparisons": {},
        "interactions": {},
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
        reference_meetings = set(collected[reference_name])
        challenger_meetings = set(collected[challenger_name])
        if reference_meetings != challenger_meetings:
            result.update({
                "status": "invalid_unpaired_meetings",
                "missing_from_reference": sorted(challenger_meetings - reference_meetings),
                "missing_from_challenger": sorted(reference_meetings - challenger_meetings),
            })
            output["comparisons"][name] = result
            continue
        common = sorted(reference_meetings)
        for metric in METRIC_ORDER:
            ids = [
                mid for mid in common
                if metric in collected[reference_name][mid]
                and metric in collected[challenger_name][mid]
            ]
            if len(ids) != len(common):
                result["metrics"][metric] = {
                    "status": "invalid_incomplete_metric_pairs",
                    "n_pairs": len(ids),
                    "expected_pairs": len(common),
                    "missing_meetings": sorted(set(common) - set(ids)),
                }
                continue
            reference = [collected[reference_name][mid][metric] for mid in ids]
            challenger = [collected[challenger_name][mid][metric] for mid in ids]
            session_result = _paired_bootstrap(
                reference, challenger, samples=samples, seed=seed,
                optimization=METRIC_OPTIMIZATION[metric],
            )
            series_reference, series_challenger = _paired_series_observations(
                ids, reference, challenger,
            )
            series_result = _paired_bootstrap(
                series_reference, series_challenger, samples=samples, seed=seed + 1,
                optimization=METRIC_OPTIMIZATION[metric],
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
    output["interactions"]["c3_gates"] = _build_c3_gate_interaction(
        collected, samples=samples, seed=seed,
    )
    return output
