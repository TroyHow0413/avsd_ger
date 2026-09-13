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
report = evaluate_session(session.turns, language="auto")
print(report.sa_wer, report.scr, report.av_sid_acc, report.der, report.jer)
```

`language="auto"` uses the per-turn Whisper detector metadata. Legacy debug
files do not contain that field and must be analysed with an explicit ISO
language such as `language="en"`. English uses Whisper's
`EnglishTextNormalizer`; other languages use `BasicTextNormalizer`.

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

### Public-library appendix metrics

Each evaluation now records a `standard_metrics` block in both the meeting
result and its `*.debug.json` sidecar.  These scores use public packages rather
than the project's backwards-compatible metric implementations:

* **JiWER 4.0.0**: WER, MER, WIL, WIP, CER, hits, substitutions, deletions,
  insertions, and reference/hypothesis lengths.
* **MeetEval 0.4.3+**: SISO-WER, cpWER, ORC-WER, tcpWER, tcORC-WER, greedy
  ORC/tcORC, DI-cpWER, MIMO-WER, and tcMIMO-WER when supported by the installed
  version.  Time-constrained scores use a recorded 5-second collar and
  MeetEval's character-based pseudo-word timing.
* **scikit-learn**: meeting-mapped speaker accuracy, balanced accuracy,
  macro/weighted F1, macro precision/recall, MCC, Cohen's kappa, plus Brier,
  log loss, ROC-AUC and average precision for C3 confidence against exact
  normalized turn correctness.
* **pyannote.metrics 3.2.1**: DER and JER with 0 and 0.25 second collars,
  overlap included.  These are explicitly labelled `oracle_turns` because this
  evaluator receives reference turn boundaries rather than detecting segments.

Missing or incompatible optional scorers are represented by
`status: unavailable/failed` with the exception message.  They never discard a
completed model inference.  The debug sidecars retain every reference,
hypothesis, speaker, time boundary and confidence required for offline
rescoring.

For already completed runs, add or refresh public scores without loading the
LLM, checkpoints, audio, or video:

```bash
python scripts/rescore_standard_metrics.py \
  out/ami_full_v4_llama3_8b_dev \
  --language en \
  --in-place
```

The scorer is pinned to pyannote.metrics 3.2.1/core<6 because pyannote 4.x/core
6.x requires NumPy 2.x while this project's AV stack pins NumPy below 2.  This
is an environment-compatibility pin, and every result records the actual scorer
version.  A separate NumPy-2 scoring environment can rescore the retained debug
sidecars later without model inference.  Do not compare these oracle-turn
DER/JER values with end-to-end
diarization results unless the segmentation protocol, collar and overlap policy
are identical.

```bash
python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/session_manifest.json \
    --pool checkpoints/identity_pool.pt \
    --out out/ablation_report.json
```

After Stage-2 training, do not point eval directly at `checkpoints/stage2/identity_pool_stage2.pt` without either `--fresh-pool` or a separately enrolled pool. That file contains the trained fuser state, but it does not automatically contain enrolled evaluation speakers.

Recommended for per-meeting AMI eval: use `--fresh-pool`. The script loads the trained fuser from `--pool`, then enrolls the `speakers` block from each session manifest before running the ablation row:

```bash
python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/your_real_test_session_manifest.json \
    --pool checkpoints/stage2/identity_pool_stage2.pt \
    --fresh-pool \
    --out out/ablation_report_real.json
```

Alternative for debugging or a fixed deployment-style enrollment: pre-enroll once, then evaluate without `--fresh-pool`:

```bash
python scripts/enroll_identity.py \
    --manifest data/your_real_test_speakers.json \
    --in-pool checkpoints/stage2/identity_pool_stage2.pt \
    --out-pool checkpoints/stage2/identity_pool_stage2_enrolled.pt

python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/your_real_test_session_manifest.json \
    --pool checkpoints/stage2/identity_pool_stage2_enrolled.pt \
    --out out/ablation_report_real.json
```

At startup, verify the pool/enrollment log shows a non-zero speaker count. If the pool has `0` speakers, AV-SID/DER/JER will collapse because every turn is effectively unknown.

Rows (controlled via `cfg.ablation` overrides):

| Row | Flag flipped | What it isolates |
|---|---|---|
| `full_model` | (none) | baseline |
| `wo_c1` | `disable_c1: true` | contribution of cross-modal identity conditioning |
| `wo_c2` | `disable_c2: true` | contribution of the GER head over ASR 1-best |
| `wo_c3` | `disable_c3: true` | contribution of the closed loop |
| `c3_wo_conf_gates` | `disable_c3_decision_gate: true`, `disable_c3_update_gate: true` | both C3 confidence gates; GER safety remains enabled |

The legacy `disable_conf_gate` key remains accepted, emits a deprecation
warning, and maps to both new switches. Historical `c3_wo_conf_gate` results
only disabled the update gate and must not be mixed with the corrected row.

**Structural-safety check** is computed across manifests using a paired
manifest-cluster bootstrap. Equality and a 95% interval crossing zero are
`inconclusive`, never `PASS`; a single manifest is `insufficient`.

```
c3_wo_conf_gates - wo_c3 canonical SA-WER: mean delta, 95% CI, status
```

If `FAIL`, the gate isn't doing what the spec says it does — investigate before claiming the framework's safety property.

The output JSON (`out/ablation_report.json`) is one record per ablation row: metrics, energy report, transcript, speaker order, flags. Easy to diff across experiments.

---

## Subset / debugging

Restrict to a few rows:

```bash
python scripts/eval_ablations.py --only full_model wo_c1 c3_wo_conf_gates ...
```

Skip the power monitor (e.g. on a CI runner without NVML):

```bash
python scripts/eval_ablations.py --no-power ...
```

---

## Raw-Video Frontend Profiles

For raw meeting-video experiments, record which diarization / active-speaker
frontend produced the turn manifest. The recommended profiles are documented
in [`AVSD_FRONTENDS.md`](AVSD_FRONTENDS.md), and the canonical registry can be
printed with:

```bash
python scripts/frontend_profiles.py --format markdown
```

Use `oracle_turns` as the upper-bound condition, `common_pyannote_lightasd` as
the reproducible open-source backbone, `strong_sortformer_talknet` as the
strong/SOTA-ish frontend reference, and `degraded_pyannote` as the robustness
side proof.

`scripts/eval_ablations.py` accepts either one manifest, a directory of
manifests, or a glob pattern:

```powershell
python scripts\eval_ablations.py `
  --config configs\default.yaml `
  --manifest data\ami_test\manifests `
  --pool checkpoints\identity_pool.pt `
  --out out\ami_ablation `
  --frontend-profile common_pyannote_lightasd `
  --no-power
```

When multiple manifests are matched, `--out` is treated as an output directory
and the script writes one report per manifest plus `summary.json`.

## Formal artifact bundle

Formal artifacts are enabled by default. With `--out out/ami_test.json`, the
legacy JSON remains at that path and the reproducible bundle is written under
`out/ami_test/`:

```text
ami_test/
├── run_manifest.json
├── records.schema.json
├── records/<ablation>.jsonl
├── metrics/
│   ├── main_table.json
│   ├── appendix_sdi.json
│   ├── appendix_correction.json
│   ├── appendix_sid.json
│   ├── appendix_calibration.json
│   ├── per_ablation/<ablation>.json
│   ├── per_meeting/<ablation>.jsonl
│   ├── groups/{by_snr,by_lip_conf,by_turn_length,by_duration,by_visual_availability}.json
│   └── scoring_protocol.json
├── scoring_inputs/
│   ├── reference/{reference.stm,reference.rttm,reference.seglst.json}
│   └── hypothesis/<ablation>/{hypothesis.stm,hypothesis.rttm,hypothesis.seglst.json}
├── profiles/{efficiency.json,power.json,latency_per_turn.jsonl}
└── debug/<ablation>/<meeting>.debug.json
```

`run_manifest.json` is the authoritative ablation list and records canonical
IDs, legacy IDs, flags, source/checkpoint hashes, Git state, package versions,
seed and timestamps. The internal compatibility ID `c3_wo_conf_gates` is
written as canonical `c3_wo_confidence_gates` in the formal bundle.

Each JSONL turn includes the reference, ASR hypothesis, raw/clean/final GER
hypotheses, fallback metadata, Top-5 identity ranking, confidence signals,
raw estimated SNR dB, normalized SNR score, visual quality, synchronized
latency and memory observations. C1 still makes decisions with Top-3; Top-5
is logging-only. CUDA peak memory is reset once per ablation, not per turn.
`gpu_peak_mb` is therefore a meeting-ablation run-level peak repeated on each
turn, while `gpu_memory_allocated_mb` is the instantaneous allocation sampled
after that turn.

`appendix_calibration.json` keeps GER confidence calibration and C1 identity
calibration separately. C1 correctness uses the meeting-level Hungarian label
mapping. Because `av_consistency_raw` is a cosine similarity rather than a
learned probability, ECE/Brier use `clip(av_consistency_raw, 0, 1)` as an
explicitly documented diagnostic proxy; ranking metrics retain the raw score.

The scoring protocol fixes the text normalizer, public scorer versions,
diarization collars, aggregation rules, group bins and definitions of the few
project-specific metrics. Use `--no-formal-artifacts` only when a lightweight
legacy/debug run is explicitly desired.

## Offline canonical debug audit

Recompute ASR, raw-GER, final and lip WER without loading any model:

```bash
python scripts/analyze_debug_outputs.py \
  out/eval_qwen/summary.json out/eval_llama/summary.json \
  --language en --out-dir out/phase_a_analysis
```

The output includes `report.json`, `report.md`, `sessions.csv` and `turns.csv`.
It reports GER candidate/acceptance coverage, per-turn improvement or harm,
iteration versus final fallback, gate failure reasons, C1 raw top-1 accuracy,
UNKNOWN coverage and runtime/throughput metadata. `jiwer==4.0.0` independently
checks every edit-distance result.
