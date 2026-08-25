# Repaired AMI Data Pipeline

The production AMI path uses complete eligible turns, AMI global participant
IDs, official meeting-specific Closeup mappings, measured visual tracking
quality, explicit IHM metadata, and fixed disjoint split outputs.

## Why old visual manifests must be rebuilt

The previous `data/ami_*_visual` files are diagnostic smoke data, not a valid
full-corpus training set:

- the builder capped most meetings at 50 turns and roughly 13 turns per speaker;
- it assumed `A=Closeup1`, `B=Closeup2`, `C=Closeup3`, `D=Closeup4`, although
  AMI camera assignments vary by meeting;
- it treated meeting-local labels such as `ES2002a_A` as if they were stable
  identities;
- JSONL conversion invented an all-ones lip-confidence track when none existed;
- generating full data with the old converter used an O(N^2) negative pool;
- eight dev meetings also exist in the legacy test-manifest directory.

Do not mix old and rebuilt JSON/JSONL files in one experiment. The repaired
pipeline never overwrites the legacy directories. Its default version is:

```text
data/ami_full_v2/
  train/base/       # base manifests, turn WAVs, quality enrollment
  train/visual/     # repaired visual manifests and mouth ROIs
  train/train.jsonl
  dev/...
  test/...
```

For a later pipeline change, use a new immutable directory such as
`--run-dir data/ami_full_v3` instead of modifying `ami_full_v2`.

## One-time metadata setup

```bash
python scripts/fetch_ami_metadata.py
```

This installs the official AMI `corpusResources/meetings.xml`, which supplies
the authoritative `nxt_agent`, headset channel, Closeup camera, role, and global
participant ID mapping.

## Rebuild complete visual manifests

Use the project training environment containing OpenCV, dlib, ffmpeg, NumPy,
and the configured dlib/AV-HuBERT model files:

```bash
python scripts/rebuild_ami_base_manifests.py \
  --run-dir data/ami_full_v2 \
  --splits train dev test

python scripts/build_ami_visual_manifests.py \
  --run-dir data/ami_full_v2 \
  --splits train dev test \
  --jobs 2
```

The base rebuild reuses existing per-turn WAVs by default, regenerates
quality-selected enrollment clips, attaches corpus-global identities, and
records IHM/oracle-turn conditions. Pass `--overwrite-turn-audio` only when the
existing turn WAVs themselves must be re-sliced. If a long build is interrupted,
rerun the same command with `--resume`; completed meeting manifests are skipped.
Without `--resume`, a non-empty version directory is rejected rather than
overwritten.

There is no turn cap by default. `--max-turns` and
`--max-turns-per-speaker` are retained only for smoke tests. Legacy manifests
are used only to recover the agreed split membership. Rebuilt artifacts are
written exclusively inside `data/ami_full_v2`, and duplicated legacy test
meetings are excluded before the new base manifests are created.

The extraction confidence is an auditable tracking signal:

- `1.0`: face/landmarks detected directly on the frame;
- `0.5`: missing frame interpolated between direct detections;
- `0.25`: leading/trailing frame extrapolated from the nearest detection;
- all frames missing: reject the clip instead of using a centre crop.

## Visual failure diagnosis and logging

The first partial `ami_full_v2` server build reported 1,526 failed turns out of
12,309 attempts (12.40%) across 57 completed train manifests. This is too large
and too meeting-dependent to treat as a harmless preprocessing detail. The
worst observed meetings included `ES2006d` (66.8%), `ES2006c` (42.1%),
`IS1004a` (37.9%) and `ES2006a` (36.0%). These failures also occurred in
manifests completed before the 48-worker run, so high concurrency may amplify
decoder pressure but cannot explain the full pattern.

Observed failure classes are:

- `landmark_all_frames`: the generated turn clip is readable, but neither the
  dlib HOG detector nor its CNN fallback produced landmarks on any frame;
- `clip_unreadable`: ffmpeg returned successfully but OpenCV could not decode a
  frame, commonly requiring checks for a zero-frame slice, source/annotation
  duration mismatch, or excessive concurrent decoder pressure;
- `ffmpeg_slice_failed`: ffmpeg returned non-zero; its captured diagnostic
  output is retained;
- `missing_source_video`, `missing_nxt_agent`, `unexpected_roi_shape`, and
  `confidence_length_mismatch`: explicit source, metadata, or output-contract
  failures;
- native child-process failure such as `signal_11`: an environment/library
  crash. The initial universal failure was dlib 20.0.1 crashing while loading
  the CNN model; production preprocessing must use and record the verified
  AV-HuBERT-compatible environment.

Every newly processed meeting now writes:

```text
data/<build-id>/
  logs/visual/last_invocation.json              # Python/dlib/OpenCV/ffmpeg/workers
  logs/visual/<split>/<meeting>.log             # complete child stdout/stderr
  logs/visual/<split>/<meeting>.result.json     # exit/signal, elapsed time, counts
  <split>/visual/<meeting>/failures.jsonl       # one structured record per failed turn
```

Each failure record includes the turn ID, speaker/agent, camera, timestamps,
source and clip paths, exception class/stage, ffmpeg output when applicable,
and source/clip decoder probes. The meeting manifest stores the failure-log
path, reason counts, attempts, successes and visual-turn coverage. Resume output
prints these counts for pre-existing manifests instead of calling file
existence alone a clean completion.

The strict audit now fails on missing meeting manifests, base/visual eligible
turn mismatch, processing failures, absent or inconsistent failure ledgers,
missing/unreadable ROI files, and ROI frame coverage outside 90%-110% of
`duration * 25`. For diagnosis only, processing failures can be reported
without making that condition alone fatal:

```bash
python scripts/audit_ami_manifests.py \
  --run-dir data/ami_full_v3 \
  --allow-processing-failures
```

Do not mix already completed `ami_full_v2` manifests, which lack structured
failure ledgers, with newly instrumented outputs and claim one homogeneous
build. Preserve `ami_full_v2` as diagnostic evidence and create a new immutable
build ID for the instrumented rebuild:

```bash
python scripts/rebuild_ami_base_manifests.py \
  --run-dir data/ami_full_v3 \
  --splits train dev test

python -u scripts/build_ami_visual_manifests.py \
  --run-dir data/ami_full_v3 \
  --splits train dev test \
  --jobs 6

python scripts/audit_ami_manifests.py \
  --run-dir data/ami_full_v3
```

Start with `--jobs 6`. Increase concurrency only after comparing the structured
failure-reason distribution and unreadable-clip count on the same meetings.
Do not silently delete failed turns or accept a passing training run as proof
that preprocessing coverage is complete. The all-landmarks-missing policy
(reference full-frame resize fallback versus audio-only/zero-confidence
retention) must be selected and reported before the final Stage-1 dataset is
frozen.

Generate a read-only aggregate report at any point after one or more meetings
finish:

```bash
python scripts/analyze_ami_visual_failures.py \
  --run-dir data/ami_full_v3 \
  --out data/ami_full_v3/visual_failure_report.json
```

The report groups failures by reason, stage and camera; lists the worst
meetings; identifies unreadable/zero-byte clips and annotation timestamps past
the probed source duration; and links native process failures to their complete
meeting logs.

## Convert to training JSONL

```bash
python scripts/ami_visual_to_jsonl.py \
  --manifest-dir data/ami_full_v2/train/visual \
  --out data/ami_full_v2/train/train.jsonl

python scripts/ami_visual_to_jsonl.py \
  --manifest-dir data/ami_full_v2/dev/visual \
  --out data/ami_full_v2/dev/dev.jsonl

python scripts/ami_visual_to_jsonl.py \
  --manifest-dir data/ami_full_v2/test/visual \
  --out data/ami_full_v2/test/test.jsonl
```

Conversion fails closed if `lip_conf_v` is missing or its length differs from
the mouth-ROI frame count. `--allow-missing-lip-confidence` exists only for
non-C1 diagnostics and must not be used for dual-gate identity training.

## Required audit gate

```bash
python scripts/audit_ami_manifests.py \
  --run-dir data/ami_full_v2
```

The audit fails on turn caps recorded in metadata, unverified camera mappings,
missing global identities, missing/fabricated confidence tracks, ROI/confidence
length mismatch, unreadable ROI files, or meeting overlap across splits.
It also verifies that every manifest carries the same immutable
`dataset_build_id` as the run directory.

## Comparing legacy and repaired AMI

Keep experiment outputs versioned in the same way:

```text
out/ami_legacy_v1/...
out/ami_full_v2/...
```

The legacy data can still be audited explicitly without changing it:

```bash
python scripts/audit_ami_manifests.py \
  --run-dir data/ami_legacy_v1 \
  --train data/ami_train_visual \
  --dev data/ami_dev_visual \
  --test data/ami_test_visual
```

That audit is expected to fail, but its JSON statistics provide the frozen
legacy reference. Train new experiments only from one version directory at a
time and record the version name in the run config.

## Experiment declaration

This pipeline currently uses AMI IHM: individual close-talking headset audio.
It is not a far-field AMI result. Turns come from reference transcript
boundaries, so the current evaluation uses oracle turns; diarization is an
input/precondition rather than a system output. Report these two conditions
with every Stage-1 result.
