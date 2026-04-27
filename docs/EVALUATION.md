# Evaluation

Lives under `avsd_ger/eval/` and is driven from `scripts/eval_ablations.py`.

```
avsd_ger/eval/
├── session.py     # SessionRunner: fan out single-speaker pipeline across turns
├── metrics.py     # SA-WER, SCR, AV-SID Acc, DER, JER (spec §13)
└── power.py       # PowerMonitor: pynvml + RAPL/psutil, 500 ms idle-corrected (spec §5.10)
```

---

## Session manifest

A *session* is a list of turns. Each turn is one utterance by one (assumed) speaker:

```json
{
  "speakers": [
    {"speaker_id": "alice", "enrollment_audio": "data/enrol/alice.wav"},
    {"speaker_id": "bob",   "enrollment_audio": "data/enrol/bob.wav"}
  ],
  "turns": [
    {
      "turn_id":      "t0001",
      "start":        0.0,
      "end":          3.4,
      "audio":        "data/sess/t0001.wav",
      "mouth_roi":    "data/sess/t0001.npy",
      "ref_text":     "the quick brown fox",
      "ref_speaker":  "alice"
    },
    ...
  ]
}
```

Optional per-turn fields: `speaker_mask_v` (bool [T_v]), `snr_per_tok` (float [N_tok]), `lip_conf_v` (float [T_v]).

`SessionRunner` sorts turns by `start` time, runs `pipeline.run()` per turn, stitches the outputs into a `[Speaker: ID] text\n...` transcript, and returns `SessionResult` with per-turn `SessionTurnResult` objects ready for the metrics module.

> **Diarization is a precondition**, not an output. The runner takes turn boundaries as given. This matches spec §13's SA-WER definition ("text correctness + speaker attribution, given segmentation") and keeps the two concerns cleanly separated.

---

## Metrics (spec §13)

All five operate on `list[SessionTurnResult]` and use **Hungarian assignment** (scipy's `linear_sum_assignment`, with a deterministic greedy fallback) to map hypothesis labels (e.g. enrolled `spk_02`) to reference labels (e.g. `alice`) before scoring.

| Function | What it returns | What it measures |
|---|---|---|
| `compute_sa_wer(turns)` | `(sa_wer, details)` | Speaker-Attributed WER. A reference word counts as correct only if it aligns AND its predicted speaker (after Hungarian) matches. Errors = `sub + del + ins + speaker-mismatched matches`. |
| `compute_scr(turns)` | `(scr, details)` | Speaker Confusion Rate. Of reference words that aligned to a correct word, the fraction attributed to the wrong speaker. |
| `compute_av_sid_accuracy(turns)` | `(acc, details)` | Turn-level top-1 ID accuracy after Hungarian. |
| `compute_der(turns)` | `(der, details)` | Diarization Error Rate = `(miss + false_alarm + confusion) / total_ref_speech`. Confusion uses the optimal hyp→ref mapping. **Computed at turn granularity** — assumes turns don't overlap, which matches the single-speaker-per-utterance pipeline shape. |
| `compute_jer(turns)` | `(jer, details)` | Jaccard Error Rate = mean over reference speakers of `1 − Jaccard(ref_time, hyp_time)`. Speakers without a hyp match contribute `1.0`. |

Bundled:

```python
from avsd_ger.eval.metrics import evaluate_session
report = evaluate_session(session.turns)
print(report.sa_wer, report.scr, report.av_sid_acc, report.der, report.jer)
```

---

## Power monitor (spec §5.10)

```python
from avsd_ger.eval.power import PowerMonitor

mon = PowerMonitor(sample_interval_s=0.5)   # spec-mandated 500 ms
mon.calibrate_idle(duration_s=2.0)          # baseline subtracted from every sample
with mon.measure("stage2_epoch"):
    ... run workload ...
report = mon.last_report()
print(report.energy_wh, report.avg_power_w, report.degraded)
```

* **GPU**: pynvml, summed across all visible devices (mW → W).
* **CPU**: Linux RAPL when available (preferred — reads package energy in μJ); otherwise `psutil.cpu_percent × SDP_WATTS` as an approximation. The `report.degraded` flag is set when neither is reachable so eval pipelines never crash in CPU-only or container environments.
* **Idle correction**: a baseline window is averaged before the workload; `(total_w − baseline_w)` is integrated trapezoidally.

---

## Ablation runner

`scripts/eval_ablations.py` runs the spec §10 Table 2 rows in one shot:

```bash
python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/session_manifest.json \
    --pool checkpoints/identity_pool.pt \
    --out out/ablation_report.json
```

Rows (controlled via `cfg.ablation` overrides):

| Row | Flag flipped | What it isolates |
|---|---|---|
| `full_model` | (none) | baseline |
| `wo_c1` | `disable_c1: true` | contribution of cross-modal identity conditioning |
| `wo_c2` | `disable_c2: true` | contribution of the GER head over ASR 1-best |
| `wo_c3` | `disable_c3: true` | contribution of the closed loop |
| `c3_wo_conf_gate` | `disable_conf_gate: true` | contribution of the **gate** itself, not C3 |

**Spec-mandated structural-safety check** (printed by the script):

```
[spec check] C3-w/o-gate SA-WER (X.XXXX) >= w/o-C3 SA-WER (Y.YYYY): PASS / FAIL
```

If `FAIL`, the gate isn't doing what the spec says it does — investigate before claiming the framework's safety property.

The output JSON (`out/ablation_report.json`) is one record per ablation row: metrics, energy report, transcript, speaker order, flags. Easy to diff across experiments.

---

## Subset / debugging

Restrict to a few rows:

```bash
python scripts/eval_ablations.py --only full_model wo_c1 c3_wo_conf_gate ...
```

Skip the power monitor (e.g. on a CI runner without NVML):

```bash
python scripts/eval_ablations.py --no-power ...
```
