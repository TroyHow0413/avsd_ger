"""Primary evaluation metrics (spec section 13).

Implemented:
    compute_sa_wer           -- Speaker-Attributed WER (a match counts only if
                                both the word and the predicted speaker are right)
    compute_scr              -- Speaker Confusion Rate (fraction of aligned-correct
                                reference words attributed to the wrong speaker)
    compute_av_sid_accuracy  -- turn-level top-1 speaker ID accuracy
    compute_der              -- Diarization Error Rate (miss + false alarm + confusion)
    compute_jer              -- Jaccard Error Rate (mean 1 - Jaccard over speakers)

All metrics operate on `SessionTurnResult` lists (what SessionRunner returns)
plus reference turns. Reference speaker labels and hypothesis speaker labels
almost never share a namespace (reference uses human names; hypothesis uses
enrolled-pool IDs), so DER / JER / AV-SID-Acc compute the optimal hyp->ref
mapping via Hungarian assignment before scoring.

We deliberately avoid pulling in `pyannote.metrics` at runtime -- this module
is self-contained and easy to audit against the spec. If a team later wants
the pyannote implementations they can swap them in behind the same API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .session import SessionTurnResult


# ------------------------------------------------------------------ helpers

def _tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return text.strip().split()


def _pairs_from_turns(
    turns: Sequence[SessionTurnResult],
    which: str,  # "ref" or "hyp"
) -> list[tuple[str, str]]:
    """Return a flat list of (speaker_label, word) tuples across turns."""
    out: list[tuple[str, str]] = []
    for t in turns:
        if which == "ref":
            spk = t.ref_speaker if t.ref_speaker is not None else "__NONE__"
            text = t.ref_text
        elif which == "hyp":
            spk = t.hyp_speaker if t.hyp_speaker is not None else "__NONE__"
            text = t.hyp_text
        else:
            raise ValueError(which)
        for w in _tokens(text):
            out.append((spk, w))
    return out


def _hungarian_label_mapping(
    ref_labels: Sequence[str],
    hyp_labels: Sequence[str],
    cost: np.ndarray,
) -> dict[str, str]:
    """Return hyp->ref mapping minimizing cost. Falls back to greedy if scipy missing.

    cost shape: [n_ref, n_hyp]. Lower cost = better match.
    Unmatched hyp labels map to themselves (passthrough).
    """
    if len(ref_labels) == 0 or len(hyp_labels) == 0:
        return {h: h for h in hyp_labels}
    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment handles rectangular matrices.
        r_idx, h_idx = linear_sum_assignment(cost)
        mapping: dict[str, str] = {}
        used_ref: set[int] = set()
        for r, h in zip(r_idx, h_idx):
            mapping[hyp_labels[h]] = ref_labels[r]
            used_ref.add(r)
        # Any unmatched hyp label passes through as itself (will count as confusion).
        for h in hyp_labels:
            mapping.setdefault(h, h)
        return mapping
    except Exception:
        # Greedy fallback -- deterministic, good enough for small label sets.
        mapping = {}
        used: set[int] = set()
        for j, h in enumerate(hyp_labels):
            best_i, best_c = -1, float("inf")
            for i, _ in enumerate(ref_labels):
                if i in used:
                    continue
                c = float(cost[i, j])
                if c < best_c:
                    best_c, best_i = c, i
            if best_i >= 0:
                mapping[h] = ref_labels[best_i]
                used.add(best_i)
            else:
                mapping[h] = h
        return mapping


# ------------------------------------------------------------------ WER / SA-WER / SCR

def _word_levenshtein_align(
    ref: Sequence[tuple[str, str]],
    hyp: Sequence[tuple[str, str]],
) -> list[tuple[str, int, int]]:
    """Standard word-level Levenshtein; returns edit ops as (op, i_ref, j_hyp).

    op in {"match", "sub", "del", "ins"}. Indices are -1 when not applicable.
    Cost is computed on *words only*; speaker tags are carried through for the
    caller to score attribution after alignment.
    """
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=match/sub, 1=del, 2=ins

    for i in range(1, n + 1):
        dp[i, 0] = i
        back[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = j
        back[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1][1] == hyp[j - 1][1] else 1
            diag = dp[i - 1, j - 1] + cost
            up = dp[i - 1, j] + 1
            left = dp[i, j - 1] + 1
            best = diag
            b = 0
            if up < best:
                best, b = up, 1
            if left < best:
                best, b = left, 2
            dp[i, j] = best
            back[i, j] = b

    # Trace back
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i, j] == 0:
            op = "match" if ref[i - 1][1] == hyp[j - 1][1] else "sub"
            ops.append((op, i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and back[i, j] == 1:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            ops.append(("ins", -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


@dataclass
class MetricsReport:
    """Container for a single evaluation run."""

    sa_wer: float = 0.0
    wer: float = 0.0
    scr: float = 0.0
    av_sid_acc: float = 0.0
    der: float = 0.0
    jer: float = 0.0
    n_ref_words: int = 0
    n_turns: int = 0
    details: dict = field(default_factory=dict)


def compute_sa_wer(turns: Sequence[SessionTurnResult]) -> tuple[float, dict]:
    """Speaker-Attributed WER.

    Alignment is done on words (standard Levenshtein). A ref word is "correct"
    only if the alignment yielded a match AND the predicted speaker label
    agrees with the reference speaker label AFTER optimal hyp->ref label
    mapping.

    Returns (sa_wer, details). details includes raw WER, counts, and mapping.
    """
    ref_pairs = _pairs_from_turns(turns, "ref")
    hyp_pairs = _pairs_from_turns(turns, "hyp")
    n_ref = len(ref_pairs)
    if n_ref == 0:
        return 0.0, {"n_ref_words": 0, "wer": 0.0}

    # Build label confusion matrix from turn-level alignment for mapping.
    ref_labels = sorted({p[0] for p in ref_pairs if p[0] != "__NONE__"})
    hyp_labels = sorted({p[0] for p in hyp_pairs if p[0] != "__NONE__"})
    cost = np.zeros((max(len(ref_labels), 1), max(len(hyp_labels), 1)), dtype=np.float64)
    # Cost = -co-occurrence time weighted by shared-word count from turn-level pairs.
    ref_idx = {l: i for i, l in enumerate(ref_labels)}
    hyp_idx = {l: j for j, l in enumerate(hyp_labels)}
    for t in turns:
        if t.ref_speaker in ref_idx and t.hyp_speaker in hyp_idx:
            dur = max(1e-6, float(t.end) - float(t.start))
            cost[ref_idx[t.ref_speaker], hyp_idx[t.hyp_speaker]] -= dur
    mapping = _hungarian_label_mapping(ref_labels, hyp_labels, cost)

    ops = _word_levenshtein_align(ref_pairs, hyp_pairs)
    n_sub = n_del = n_ins = 0
    n_spk_err = 0  # additional errors when word matched but speaker disagreed
    for op, i, j in ops:
        if op == "sub":
            n_sub += 1
        elif op == "del":
            n_del += 1
        elif op == "ins":
            n_ins += 1
        elif op == "match":
            ref_spk = ref_pairs[i][0]
            hyp_spk_mapped = mapping.get(hyp_pairs[j][0], hyp_pairs[j][0])
            if ref_spk != hyp_spk_mapped:
                n_spk_err += 1

    wer = (n_sub + n_del + n_ins) / n_ref
    sa_wer = (n_sub + n_del + n_ins + n_spk_err) / n_ref
    details = {
        "wer": wer,
        "n_ref_words": n_ref,
        "n_sub": n_sub,
        "n_del": n_del,
        "n_ins": n_ins,
        "n_spk_err": n_spk_err,
        "mapping": mapping,
    }
    return sa_wer, details


def compute_scr(turns: Sequence[SessionTurnResult]) -> tuple[float, dict]:
    """Speaker Confusion Rate.

    Among reference words that the hypothesis got *textually right* (matched in
    alignment), what fraction were attributed to the wrong speaker?
    """
    ref_pairs = _pairs_from_turns(turns, "ref")
    hyp_pairs = _pairs_from_turns(turns, "hyp")
    if not ref_pairs or not hyp_pairs:
        return 0.0, {"n_matched": 0, "n_spk_err": 0}

    ref_labels = sorted({p[0] for p in ref_pairs if p[0] != "__NONE__"})
    hyp_labels = sorted({p[0] for p in hyp_pairs if p[0] != "__NONE__"})
    cost = np.zeros((max(len(ref_labels), 1), max(len(hyp_labels), 1)), dtype=np.float64)
    ref_idx = {l: i for i, l in enumerate(ref_labels)}
    hyp_idx = {l: j for j, l in enumerate(hyp_labels)}
    for t in turns:
        if t.ref_speaker in ref_idx and t.hyp_speaker in hyp_idx:
            dur = max(1e-6, float(t.end) - float(t.start))
            cost[ref_idx[t.ref_speaker], hyp_idx[t.hyp_speaker]] -= dur
    mapping = _hungarian_label_mapping(ref_labels, hyp_labels, cost)

    ops = _word_levenshtein_align(ref_pairs, hyp_pairs)
    n_matched = 0
    n_spk_err = 0
    for op, i, j in ops:
        if op == "match":
            n_matched += 1
            ref_spk = ref_pairs[i][0]
            hyp_spk_mapped = mapping.get(hyp_pairs[j][0], hyp_pairs[j][0])
            if ref_spk != hyp_spk_mapped:
                n_spk_err += 1

    scr = (n_spk_err / n_matched) if n_matched else 0.0
    return scr, {"n_matched": n_matched, "n_spk_err": n_spk_err, "mapping": mapping}


# ------------------------------------------------------------------ AV-SID

def compute_av_sid_accuracy(turns: Sequence[SessionTurnResult]) -> tuple[float, dict]:
    """Turn-level top-1 Audio-Visual Speaker ID accuracy.

    Uses the Hungarian-mapped hypothesis labels so arbitrary enrolled-ID
    strings (e.g. "spk_02") are scored against reference names (e.g. "alice").
    Turns with no ref_speaker are skipped.
    """
    labeled = [t for t in turns if t.ref_speaker is not None]
    if not labeled:
        return 0.0, {"n": 0, "n_correct": 0}

    ref_labels = sorted({t.ref_speaker for t in labeled})
    hyp_labels_set = {t.hyp_speaker for t in labeled if t.hyp_speaker is not None}
    hyp_labels = sorted(hyp_labels_set)
    cost = np.zeros((max(len(ref_labels), 1), max(len(hyp_labels), 1)), dtype=np.float64)
    ri = {l: i for i, l in enumerate(ref_labels)}
    hi = {l: j for j, l in enumerate(hyp_labels)}
    for t in labeled:
        if t.hyp_speaker in hi:
            dur = max(1e-6, float(t.end) - float(t.start))
            cost[ri[t.ref_speaker], hi[t.hyp_speaker]] -= dur
    mapping = _hungarian_label_mapping(ref_labels, hyp_labels, cost)

    n_correct = 0
    for t in labeled:
        if t.hyp_speaker is None:
            continue
        if mapping.get(t.hyp_speaker, t.hyp_speaker) == t.ref_speaker:
            n_correct += 1
    acc = n_correct / len(labeled)
    return acc, {"n": len(labeled), "n_correct": n_correct, "mapping": mapping}


# ------------------------------------------------------------------ DER / JER

def _total_time(segments: Iterable[tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in segments)


def _intersect(a: tuple[float, float], b: tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def _per_speaker_segments(
    turns: Sequence[SessionTurnResult], which: str,
) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        if which == "ref":
            lbl = t.ref_speaker
        else:
            lbl = t.hyp_speaker
        if lbl is None:
            continue
        out.setdefault(lbl, []).append((float(t.start), float(t.end)))
    return out


def compute_der(turns: Sequence[SessionTurnResult]) -> tuple[float, dict]:
    """Diarization Error Rate (spec section 13).

    DER = (miss + false_alarm + confusion) / total_ref_speech_time,
    where confusion is computed under the optimal hyp->ref label mapping.

    Turn-level granularity: we treat each turn as a contiguous [start, end)
    speaker region. No overlap handling -- consistent with single-speaker
    pipeline execution shape.
    """
    ref_segs = _per_speaker_segments(turns, "ref")
    hyp_segs = _per_speaker_segments(turns, "hyp")
    total_ref = sum(_total_time(v) for v in ref_segs.values())
    total_hyp = sum(_total_time(v) for v in hyp_segs.values())
    if total_ref <= 0.0:
        return 0.0, {"total_ref": 0.0}

    ref_labels = sorted(ref_segs.keys())
    hyp_labels = sorted(hyp_segs.keys())
    # Cost = negative overlap. Hungarian minimizes -> maximizes overlap.
    cost = np.zeros((max(len(ref_labels), 1), max(len(hyp_labels), 1)), dtype=np.float64)
    for i, rl in enumerate(ref_labels):
        for j, hl in enumerate(hyp_labels):
            ov = 0.0
            for rs in ref_segs[rl]:
                for hs in hyp_segs[hl]:
                    ov += _intersect(rs, hs)
            cost[i, j] = -ov
    mapping = _hungarian_label_mapping(ref_labels, hyp_labels, cost)
    # Invert mapping: mapped ref label -> hyp label (for scoring)
    hyp_to_ref = mapping  # hyp_label -> ref_label (its best-matched ref)

    # Aggregate correct / confusion at turn granularity
    correct = 0.0
    confusion = 0.0
    for t in turns:
        if t.ref_speaker is None:
            continue
        dur = max(0.0, float(t.end) - float(t.start))
        if t.hyp_speaker is None:
            # Miss handled below via total_hyp accounting
            continue
        mapped = hyp_to_ref.get(t.hyp_speaker, t.hyp_speaker)
        if mapped == t.ref_speaker:
            correct += dur
        else:
            confusion += dur

    # Miss/False-alarm at the coarse level: amount of ref time with no hyp overlap
    # and hyp time with no ref overlap. For non-overlapping turn segmentation the
    # simple form below is exact.
    miss = 0.0
    false_alarm = 0.0
    # Ref time that doesn't overlap any hyp segment at all (any speaker):
    all_hyp = [seg for segs in hyp_segs.values() for seg in segs]
    for rl, segs in ref_segs.items():
        for rs in segs:
            covered = sum(_intersect(rs, hs) for hs in all_hyp)
            miss += max(0.0, (rs[1] - rs[0]) - covered)
    all_ref = [seg for segs in ref_segs.values() for seg in segs]
    for hl, segs in hyp_segs.items():
        for hs in segs:
            covered = sum(_intersect(hs, rs) for rs in all_ref)
            false_alarm += max(0.0, (hs[1] - hs[0]) - covered)

    der = (miss + false_alarm + confusion) / total_ref
    return der, {
        "total_ref": total_ref,
        "total_hyp": total_hyp,
        "miss": miss,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "correct": correct,
        "mapping": mapping,
    }


def compute_jer(turns: Sequence[SessionTurnResult]) -> tuple[float, dict]:
    """Jaccard Error Rate -- mean over ref speakers of (1 - Jaccard).

    Jaccard(ref_i, hyp_matched) = overlap / (ref_dur + hyp_dur - overlap).
    Speakers with no matching hyp contribute 1.0 (maximum error).
    """
    ref_segs = _per_speaker_segments(turns, "ref")
    hyp_segs = _per_speaker_segments(turns, "hyp")
    if not ref_segs:
        return 0.0, {"per_speaker": {}}

    ref_labels = sorted(ref_segs.keys())
    hyp_labels = sorted(hyp_segs.keys())
    cost = np.zeros((max(len(ref_labels), 1), max(len(hyp_labels), 1)), dtype=np.float64)
    for i, rl in enumerate(ref_labels):
        for j, hl in enumerate(hyp_labels):
            ov = 0.0
            for rs in ref_segs[rl]:
                for hs in hyp_segs[hl]:
                    ov += _intersect(rs, hs)
            cost[i, j] = -ov
    mapping = _hungarian_label_mapping(ref_labels, hyp_labels, cost)
    # Build ref->hyp mapping from hyp->ref mapping
    ref_to_hyp: dict[str, str | None] = {rl: None for rl in ref_labels}
    for h, r in mapping.items():
        if r in ref_to_hyp and ref_to_hyp[r] is None:
            ref_to_hyp[r] = h

    per_speaker: dict[str, float] = {}
    total = 0.0
    for rl in ref_labels:
        hl = ref_to_hyp.get(rl)
        ref_dur = _total_time(ref_segs[rl])
        if hl is None or hl not in hyp_segs:
            per_speaker[rl] = 1.0
            total += 1.0
            continue
        hyp_dur = _total_time(hyp_segs[hl])
        ov = 0.0
        for rs in ref_segs[rl]:
            for hs in hyp_segs[hl]:
                ov += _intersect(rs, hs)
        union = ref_dur + hyp_dur - ov
        jac = (ov / union) if union > 0 else 0.0
        per_speaker[rl] = 1.0 - jac
        total += per_speaker[rl]
    jer = total / len(ref_labels)
    return jer, {"per_speaker": per_speaker, "mapping": mapping}


# ------------------------------------------------------------------ bundled runner

def evaluate_session(turns: Sequence[SessionTurnResult]) -> MetricsReport:
    """Compute all five spec metrics on one session's turn results."""
    sa_wer, sa_det = compute_sa_wer(turns)
    scr, scr_det = compute_scr(turns)
    av_sid, sid_det = compute_av_sid_accuracy(turns)
    der, der_det = compute_der(turns)
    jer, jer_det = compute_jer(turns)

    return MetricsReport(
        sa_wer=sa_wer,
        wer=sa_det.get("wer", 0.0),
        scr=scr,
        av_sid_acc=av_sid,
        der=der,
        jer=jer,
        n_ref_words=sa_det.get("n_ref_words", 0),
        n_turns=len(turns),
        details={
            "sa_wer": sa_det,
            "scr": scr_det,
            "av_sid": sid_det,
            "der": der_det,
            "jer": jer_det,
        },
    )
