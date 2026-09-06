# LRS2 and VoxCeleb2 server preprocessing

These builders create portable, versioned JSONL manifests and materialised
inputs for the current AVSD-GER trainers. Run them from the repository root so
manifest paths remain repository-relative. Copy the complete output directory
back into the same relative location on another machine.

## Prerequisites

- `ffmpeg` on `PATH`;
- the project Python environment;
- LRS2 production mouth extraction: the pinned dlib environment and the three
  model assets configured below;
- enough free space. LRS2 mouth ROI is stored as `uint8` by default to avoid
  the fourfold size cost of `float32` arrays.

The builders are resumable by default. Existing WAV, image and NPY outputs are
validated and reused. `--overwrite` deliberately regenerates them. Every run
writes `audit.json` and `failures.jsonl`; any failure causes a non-zero exit
status while retaining all successful outputs. `build_config.json` prevents a
resume from mixing outputs made with different ROI backends, dtypes, split
seeds or curation settings.

## LRS2

First run a small production-backend smoke test:

```bash
python scripts/prepare_lrs2_full.py \
  --lrs2-root datasets/lrs2 \
  --output-root data/lrs2_full_v1 \
  --manifest-root "$PWD" \
  --splits train val \
  --max-items 10 \
  --workers 1
```

Then remove `--max-items` and run all official splits:

```bash
python scripts/prepare_lrs2_full.py \
  --lrs2-root datasets/lrs2 \
  --output-root data/lrs2_full_v1 \
  --manifest-root "$PWD" \
  --splits pretrain train val test \
  --workers 4
```

`dlib` is the production default and uses:

```text
checkpoints/shape_predictor_68_face_landmarks.dat
checkpoints/mmod_human_face_detector.dat
av_hubert/avhubert/preparation/data/20words_mean_face.npy
```

Use more than one dlib worker only after checking host RAM/VRAM use: every
worker loads its own detectors. `--roi-backend haar` is useful only for a
pipeline smoke test and must not be used for the reported production result.

Recommended starting points (benchmark 100 clips before increasing them):

- CPU-only, NVMe SSD: `--workers 8 --ffmpeg-threads 1`; try 12 or 16 only if
  CPU utilisation scales and I/O wait remains low;
- CPU-only, HDD/network disk: start with 4 workers;
- CUDA-enabled dlib: start with 1 worker, then at most 2 after checking VRAM;
- VoxCeleb2 ffmpeg-only C1 extraction: normally 8-16 workers on NVMe.

Giving every ffmpeg process multiple internal threads while also using many
workers oversubscribes the host and is commonly slower. The default therefore
keeps `--ffmpeg-threads 1`.

Outputs:

```text
data/lrs2_full_v1/
  audio/{split}/...
  mouth_roi/{split}/...
  lip_conf/{split}/...
  faces/{split}/...
  manifests/{pretrain,train,val,test}.jsonl
  build_config.json
  audit.json
  failures.jsonl
```

LRS2 is already utterance-segmented. The builder follows the official list
files instead of globbing every MP4, so unlisted media cannot leak into a
split.

## VoxCeleb2 for C1

This path requires the face-tracked **video/MP4** release. An AAC-only archive
cannot produce the paired face inputs used by C1.

Run a curated smoke test:

```bash
python scripts/prepare_voxceleb2_c1.py \
  --dev-root datasets/voxceleb2/dev/mp4 \
  --test-root datasets/voxceleb2/test/mp4 \
  --output-root data/voxceleb2_c1_v1 \
  --manifest-root "$PWD" \
  --max-per-video 2 \
  --max-per-speaker 20 \
  --max-items 100 \
  --workers 4
```

For the production curated subset, omit `--max-items`. Increase or remove the
per-video/per-speaker caps only after measuring storage and identity balance:

```bash
python scripts/prepare_voxceleb2_c1.py \
  --dev-root datasets/voxceleb2/dev/mp4 \
  --test-root datasets/voxceleb2/test/mp4 \
  --output-root data/voxceleb2_c1_v1 \
  --manifest-root "$PWD" \
  --max-per-video 5 \
  --max-per-speaker 100 \
  --workers 8
```

The official VoxCeleb2 `dev` speakers are the training source. By default, a
deterministic, speaker-disjoint 2% subset is withheld as local validation.
Official `test` is written only to `test.jsonl` and is never used for training
or checkpoint selection. Each record contains a 16 kHz mono WAV, a 224x224
representative face image and the corpus-global VoxCeleb speaker identity. Its
scalar visual-quality value records that the official face track is available;
it is not presented as detector or landmark confidence. VoxCeleb2 is used for
C1 only, not as a source of C2/GER transcripts.

## Copying results back

Copy these directories intact, including manifests and audit files:

```text
data/lrs2_full_v1
data/voxceleb2_c1_v1
```

Do not use `--absolute-paths` if results will move between machines. Before
training, require `failures.jsonl` to be empty (or document and freeze an
explicit exclusion ledger) and preserve each `audit.json` with the run.
