# AMI Full v4 Training Repair Specification

## Status and scope

`data/ami_full_v4` is a completed dataset build, not an unfinished visual
preprocessing run. Its strict audit passed with 102/12/12 train/dev/test
manifests, 32,435 successful visual turns, zero processing failures, exact
turn/exclusion accounting, no split overlap, and no short or unreadable ROI
artifacts. The remaining work is in the training code that consumes the
exported JSONL records.

The first real-backbone smoke test proved that Stage-1, frozen-feature cache,
and Stage-2 checkpoint paths can execute. It did **not** prove that Stage-2
learned correctly: most `align_ctc` debug steps reported `ctc=0`, while the
epoch mean was only 0.2729. The smoke outputs were discarded and must not be
used as initialization for a production run.

The primary GER model decision is now 8B-first: use the existing local
`checkpoints/Meta-Llama-3-8B-Instruct` checkpoint for the main English system,
then run a 3B model later as a controlled scale/efficiency baseline. This does
not change the order of the non-GER stages: Stage-1 identity training and the
align/CTC-only warm-up remain LLM-free. It changes the first GER-enabled smoke
and production run from the current Qwen2.5-3B configuration to Llama-3-8B.

This document freezes what is wrong, what will be repaired, the expected good
and adverse outcomes, and how the repaired experiment differs from previous
runs.

## Implementation status (2026-08-30)

The code repairs in this specification are now implemented on the
`mix_datasets` branch. They do not modify `data/ami_full_v4`.

- CTC uses feasibility-checked, learnable 8x-16x temporal expansion with
  `zero_infinity=False` and explicit length/zero/non-finite diagnostics.
- `lip_conf`, acoustic SNR quality and the oracle-turn speaker mask survive
  JSONL loading and frozen-feature caching. An all-zero visual gate produces
  an exactly zero visual residual.
- AMI Stage-1 uses participant labels, speaker-balanced batches and a
  multi-positive bidirectional contrastive loss. Full-corpus agglomerative
  clustering is no longer the supervised AMI path.
- Stage-1 selects `best.pt` by dev A-to-V/V-to-A speaker retrieval accuracy.
  Cached Stage-2 selects `align_ctc` by dev CTC loss, text-only `ger_lora` by
  dev WER, and AV/joint GER runs by dev SA-WER. `last.pt` is separately saved.
- Resume restores model, optimizer and RNG state and rejects a changed config,
  train/dev manifest, cache signature, model family or dataset provenance.
- Frozen-feature cache schema v3 hashes train/dev manifests, feature-affecting
  source files and configuration, the VSR checkpoint, and quality schema.
  GER family is intentionally excluded because Llama and Qwen consume the
  same frozen ASR/VSR/identity features.
- `llama-3-8b-instruct` (4096 hidden) and `qwen2.5-7b-instruct` (3584 hidden)
  are separate validated profiles. Llama-3-8B is the primary AMI model;
  Qwen2.5-7B is a later model-family comparison and does not overwrite it.
- The complete CPU unit suite and end-to-end stub Stage-1/Stage-2 lifecycle
  tests pass. The real local Llama-3-8B BF16 load/backward preflight also
  passes on the RTX PRO 6000 Blackwell: 8,076,425,216 total parameters,
  46,155,776 trainable parameters, finite loss 3.015625, all 457 observed
  gradient tensors finite, 233 non-zero, 16.17 GiB peak allocated and
  17.27 GiB peak reserved memory for the short four-target-token step.

### AMI full v4 commands after the repair

Export train and dev JSONL if they are not already present. Test may be
exported for the final sealed evaluation, but it is never passed to a trainer.

```bash
python scripts/ami_visual_to_jsonl.py \
  --manifest-dir data/ami_full_v4/train/visual \
  --out data/ami_full_v4/train/train.jsonl

python scripts/ami_visual_to_jsonl.py \
  --manifest-dir data/ami_full_v4/dev/visual \
  --out data/ami_full_v4/dev/dev.jsonl
```

Train C1 identity with dev-only checkpoint selection:

```bash
python -u scripts/train_identity.py \
  --config one_go/runs/config_real_en_llama3_8b.yaml \
  --manifest data/ami_full_v4/train/train.jsonl \
  --dev-manifest data/ami_full_v4/dev/dev.jsonl \
  --out checkpoints/ami_full_v4_stage1 \
  --no-wandb
```

Build schema-v3 frozen features and run the LLM-free align/CTC stage. The same
train/dev cache is reusable by Llama and Qwen because GER is not involved in
feature extraction.

```bash
python -u scripts/train_stage2_pro6000.py \
  --config one_go/runs/config_real_en_llama3_8b.yaml \
  --manifest data/ami_full_v4/train/train.jsonl \
  --dev-manifest data/ami_full_v4/dev/dev.jsonl \
  --cache-dir cache/ami_full_v4_train_quality_v3 \
  --dev-cache-dir cache/ami_full_v4_dev_quality_v3 \
  --out checkpoints/ami_full_v4_align_ctc \
  --warmup align_ctc \
  --stage1-pool checkpoints/ami_full_v4_stage1/identity_pool_stage1.pt \
  --debug-loss-every 100 \
  --no-wandb
```

An interrupted job resumes only into the same output directory:

```bash
python -u scripts/train_stage2_pro6000.py [same arguments as above] \
  --resume checkpoints/ami_full_v4_align_ctc/last.pt
```

Do not reuse the discarded `ami_full_v4_smoke_features`: it predates cache
schema v3 and the repaired CTC/quality path. The reusable preflight command is:

```bash
python scripts/preflight_ger_model.py \
  --config one_go/runs/config_real_en_llama3_8b.yaml \
  --backward
```

This gate has passed for a short synthetic AV context. It does not predict the
peak memory of a full-length batch, so the next action remains a new
model-specific 256-record real-backbone smoke before production training.

## Current problems and planned repairs

### P0: the selected Llama-3-8B GER model is not registered

**Current evidence**

- The local checkpoint exists at
  `checkpoints/Meta-Llama-3-8B-Instruct` and its `config.json` reports
  `model_type=llama`, `hidden_size=4096`, 32 layers, BF16 weights, and an
  8,192-token maximum context.
- `MODEL_PROFILES` in `avsd_ger/c2_alignment/model_backend.py` currently
  supports only `qwen2.5-3b-instruct` and `llama-3.2-3b-instruct`.
- `one_go/runs/config_real_en.yaml` still selects Qwen2.5-3B. Merely changing
  its model path would fail the family and hidden-size validation.
- The dense backend supports `auto/fp32/fp16/bf16`; it does not currently
  provide a production QLoRA/4-bit path. LoRA reduces trainable parameters but
  does not remove the memory needed for the dense 8B base weights.

**Repair**

1. Register a distinct `llama-3-8b-instruct` model profile with
   `hidden_size=4096`, Hugging Face model type `llama`, and validated Llama-3
   LoRA target modules.
2. Add a new English configuration such as
   `one_go/runs/config_real_en_llama3_8b.yaml`. Do not overwrite
   `config_real_en.yaml`; preserving the 3B configuration is required for the
   later controlled comparison.
3. Point the new configuration to the local checkpoint and set
   `allow_download: false` for production reproducibility. Select BF16 only
   after the target GPU passes a capability check; otherwise fail with an
   explicit message rather than silently changing numerical precision.
4. Validate the tokenizer chat template, speaker special-token insertion,
   embedding resize, EOS/PAD behavior, LoRA target coverage, and the 4096-wide
   AV-context projection on a real model load.
5. Add unit tests for profile selection, family/hidden-size mismatch rejection,
   target-module validation, and the new configuration. Add a real-backbone
   load/generate smoke that is opt-in so ordinary CPU tests remain lightweight.
6. Give 8B and 3B runs distinct output, adapter, log, and cache names. Never
   resume an 8B run from a 3B checkpoint or reuse a cache whose signature does
   not match the active data/config/model provenance.
7. Run a load-only GPU memory preflight, then a tiny GER forward/backward
   smoke, then the repaired 256-record end-to-end smoke. Record peak allocated
   and reserved memory before selecting production sequence length and
   gradient accumulation.

**Expected good outcome**

- The 8B model loads from the intended local directory without a network
  fallback, passes one forward/backward LoRA step, and produces transcript-only
  output under the project prompt.
- The AV-context bridge uses the model-reported 4096 hidden dimension and
  receives finite gradients.
- The later 3B run remains independently reproducible from its preserved
  configuration and output namespace.

**Possible adverse outcome**

- Dense BF16 8B training can exceed the available memory once activations,
  AV-context projection, optimizer state, and generation buffers are included.
- The 8B model can over-correct AMI names, acronyms, repetitions, or
  disfluencies even when its text is more fluent. Dev WER and explicit
  over-correction analysis, not language fluency, determine acceptance.
- If memory is insufficient, gradient checkpointing, shorter bounded context,
  and gradient accumulation may be added as documented 8B execution settings.
  Introducing 4-bit QLoRA is a separate implementation change and must not be
  silently mixed into the dense 3B comparison.

### P0: invalid CTC examples are silently converted to zero loss

**Current evidence**

- `CTCHead` receives word-pooled aligned features with sequence length
  `N_words`, repeats every feature four times, and trains against a character
  target.
- Many English targets need more than four CTC positions per word after
  letters, spaces, apostrophes, and repeated-character constraints are
  counted.
- `torch.nn.functional.ctc_loss(..., zero_infinity=True)` converts an
  impossible alignment's infinite loss to zero.
- The smoke run consequently printed zero CTC/total loss for most inspected
  steps even though checkpoint files were produced.

**Repair**

1. Add an explicit feasibility calculation for every target:
   `minimum_ctc_steps = target_length + adjacent_repeat_count`.
2. Fail closed when the aligned sequence is empty or cannot represent the
   target. Never treat an infeasible alignment as a valid zero-loss sample.
3. Replace identical fixed repetition with a learnable temporal expansion
   from token-level aligned features to character-resolution subframes. The
   expansion must provide at least `minimum_ctc_steps`, and different
   subframes must be able to emit different character logits.
4. Report input length, target length, minimum required length, feasibility,
   zero-loss count, and non-finite count in training logs and epoch summaries.
5. Keep `zero_infinity=False` in production so a regression stops the run.

Merely increasing identical repetition from four to eight is not considered a
complete repair: identical repeated logits do not add temporal information and
can lower the numerical loss without making the sequence more transcribable.

**Expected good outcome**

- Every accepted text-bearing sample has a finite, non-silenced CTC loss.
- The aligner and CTC head receive non-zero gradients on the great majority of
  training turns.
- Dev CTC loss and decoding metrics become interpretable rather than being
  diluted by hidden zero-loss samples.

**Possible adverse outcome**

- Learnable temporal expansion increases parameters, memory, and training
  time.
- CTC loss will initially look larger because invalid examples are no longer
  reported as zero.
- A numerically valid CTC objective may still fail to improve dev WER; CTC must
  therefore be validated and ablated rather than assumed beneficial.

### P0: Stage-2 drops the visual confidence signal

**Current evidence**

- The v4 JSONL export contains per-frame `lip_conf`.
- `train_stage2._load_record()` does not return it.
- `train_stage2_pro6000` does not store it in feature shards.
- Both trainers call `IDConditionedAligner` without `lip_conf_v` or
  `snr_per_tok`.
- When quality inputs are absent, the aligner constructs all-zero gates. The
  current additive constant is identical for every visual key, so softmax
  cancels it and visual attention is not actually suppressed.
- v4 has 7,367 successful turns (22.7%) whose landmarks were missing for the
  whole clip. They contain real resized frames, but their confidence is
  intentionally all zero.

**Repair**

1. Carry `lip_conf`, its source, and speaker mask through JSONL loading and
   frozen-feature cache shards.
2. Resample `lip_conf` deterministically to the AV-HuBERT feature clock and
   validate its length.
3. Compute or cache acoustic quality on the ASR token clock.
4. Pass both signals to the aligner.
5. Change low-confidence handling so the cross-modal residual itself is
   attenuated. If visual confidence is all zero, the visual cross-attention
   contribution must be zero and the audio residual path must remain usable.
6. Include quality schema and parameters in the cache signature; old caches
   become incompatible by design.

**Expected good outcome**

- C2/C3 claims match the implementation: high-quality visual evidence helps,
  interpolated evidence is down-weighted, and all-zero fallback turns behave
  as auditable audio-dominant examples.
- Controlled `with confidence` versus `without confidence` ablations become
  meaningful.

**Possible adverse outcome**

- Effective AV training data decreases because 22.7% of turns provide no
  trusted visual evidence and another 21.3% contain partial confidence.
- AV gains may shrink compared with the old trainer that consumed every
  visual feature as equally reliable. Such a drop would indicate removal of
  optimistic visual leakage, not necessarily a worse method.
- All Stage-2 feature caches must be rebuilt, adding disk and preprocessing
  time.

### P0: Stage-1 is not speaker-aware at full AMI scale

**Current evidence**

- Stage-1 runs `AgglomerativeClustering` once over every surviving train
  utterance. Agglomerative clustering has approximately quadratic scaling and
  is inappropriate for roughly twenty thousand or more embeddings.
- AMI v4 already contains corpus-global participant/speaker labels.
- The current diagonal InfoNCE objective treats only the same row as positive.
  Other turns from the same participant can appear in the denominator as
  negatives. Because many turns share one enrollment face, these are false
  negatives with contradictory supervision.

**Repair**

1. For the supervised AMI baseline, use corpus-global participant labels and
   do not run global cold-start clustering.
2. Use a speaker-balanced sampler and a supervised multi-positive,
   bidirectional audio-to-face/face-to-audio contrastive objective. All turns
   with the same participant label are positives or are masked from the
   negative set.
3. Keep cold-start as a separate, clearly labelled open-set/deployment
   experiment. If evaluated on AMI, cluster within a meeting/session rather
   than over the complete train split.
4. Report speakers per batch, positives per anchor, rejected-by-gate turns,
   and per-speaker sample counts.

**Expected good outcome**

- Stage-1 scales linearly with the number of records apart from encoder work.
- C1 receives consistent identity supervision and should improve dev AV
  speaker-ID/retrieval accuracy.
- The trained pool has an explicit supervised-AMI interpretation.

**Possible adverse outcome**

- This baseline is supervised identity training and must not be described as
  self-supervised cold start.
- Speaker balancing changes the empirical turn distribution and may reduce
  the influence of frequent speakers.
- Better train retrieval does not guarantee cross-session generalization;
  participant-disjoint dev evaluation remains mandatory.

### P1: no dev model selection, best checkpoint, or safe resume

**Current evidence**

- Current Stage-1 and Stage-2 entry points consume one training manifest.
- The configured `stop_on: av_sid_acc_plateau` is not implemented.
- Production checkpoints are written at the end rather than selected by a
  frozen dev criterion.

**Repair**

1. Add an explicit dev manifest to both stages.
2. Stage-1 selection metric: participant-disjoint AV identity retrieval/SID
   accuracy, with loss as a secondary diagnostic.
3. Stage-2 selection metrics: dev CTC loss plus WER/SA-WER appropriate to the
   fixed AMI-AV-valid subset. Never select on test.
4. Save `last` and `best` checkpoints with optimizer, scheduler, epoch, random
   state, data build ID, config, and code commit.
5. Add safe resume that rejects mismatched data/config/cache signatures.

**Expected good outcome**

- Epoch count and checkpoint choice become reproducible and independent of
  test results.
- Long runs can recover after interruption.

**Possible adverse outcome**

- Dev evaluation increases runtime and checkpoint storage.
- Different selection metrics may choose different epochs; the primary metric
  must be frozen before test evaluation.

### P1: cache provenance is incomplete

**Repair**

The production cache signature must include the manifest digest, dataset build
ID, code commit, ASR/VSR/identity checkpoint digests, quality schema, and all
feature-affecting configuration. Cache directories must encode their profile,
for example `ami_full_v4_train_beam10_nbest5_quality_v2`.

**Expected trade-off**

This prevents silent stale-cache reuse but deliberately invalidates caches
after relevant code or model changes.

## Difference from previous data and training

### Dataset difference

| Aspect | Earlier legacy/v3 state | AMI full v4 |
| --- | --- | --- |
| Turn coverage | Legacy versions included a 50-turn cap; v3 removed the cap but retained 7,903 processing failures | No turn cap; 32,435 successful visual turns and zero processing failures |
| Successful hours | v3 approximately 26.09 h | v4 approximately 33.78 h |
| Missing landmarks | Mostly became `landmark_all_frames` failures | Real-frame AV-HuBERT-style resize fallback with `lip_conf=0` |
| Source defects | Missing/short media mixed with processing failures | 550 fixed, audited source exclusions: 358 official missing Closeups and 192 verified duration overruns |
| Test definition | Failure-dependent surviving subset | Frozen AMI-AV-valid subset with deterministic exclusions |
| Enrollment | Two speakers used first-seconds fallback | Long speech segments are windowed and quality-ranked; no fallback remains |

### Training difference after repair

| Aspect | Current/previous trainer | Repaired v4 trainer |
| --- | --- | --- |
| Stage-1 labels | Global agglomerative pseudo-labels even when AMI labels exist | Corpus-global participant labels for the supervised AMI baseline |
| Contrastive positives | One diagonal row; same-speaker rows can be false negatives | Multi-positive speaker-aware bidirectional contrastive learning |
| Cold start | Whole training corpus | Separate session-level open-set experiment |
| CTC | Fixed identical 4x repetition; impossible cases silently become zero | Feasibility-checked learnable temporal expansion; fail closed on invalid cases |
| Visual confidence | Present in data but discarded by Stage-2 | Propagated through loader/cache and applied to the visual residual |
| All-zero fallback | Treated as ordinary visual evidence | Explicit audio-dominant fallback with zero trusted visual contribution |
| Cache | Manifest stat and partial config signature | Data/code/model/quality provenance signature |
| Model selection | Last epoch, no implemented dev plateau | Frozen dev metric, best+last checkpoints, resumable state |
| Test usage | Easy to evaluate repeatedly during development | Test remains sealed until all choices are frozen on dev |

## Expected result envelope

The repair is successful only if all of the following hold on a new smoke and
then on the frozen dev split:

- zero infeasible or silently zeroed CTC examples;
- finite non-zero CTC gradients on accepted text-bearing turns;
- Stage-2 cache contains and validates quality tracks;
- all-zero-confidence turns produce zero visual residual contribution;
- no same-speaker negatives in Stage-1;
- no global full-corpus agglomerative clustering in supervised AMI training;
- best checkpoint is selected using dev only;
- test is not read during training or model selection.

The expected positive result is a slower but scientifically interpretable
training run with trustworthy C1/C2/C3 ablations. A possible negative result is
that AV gains become smaller, training losses increase, or audio-only performs
similarly after unreliable visual evidence is removed. Those outcomes are
valid findings. The repair is intended to remove silent invalid supervision,
not to guarantee a larger headline score.

## Implementation order

1. CTC feasibility, learnable expansion, diagnostics, and tests.
2. Quality propagation and genuinely suppressive visual residual gating.
3. Speaker-aware Stage-1 objective and sampler.
4. Dev evaluation, best/last checkpoints, and resume.
5. Provenance-complete feature cache.
6. Register Llama-3-8B, add its non-overwriting config and tests, and complete
   the load-only plus one-step GPU memory preflight.
7. New 256-record Llama-3-8B real-backbone smoke; discard it unless every gate
   above passes.
8. Full v4 Stage-1 and cached align/CTC warm-up without loading an LLM.
9. Llama-3-8B GER/joint training, dev selection and ablation, followed by one
   final frozen test evaluation.
10. Freeze the 8B recipe, then run the preserved 3B configuration as the
    model-scale/efficiency comparison. A 27B best-recipe-only run remains
    optional.

## Operator checklist: what to do next

Do not launch another full AMI training job yet. Work through these gates in
order and retain the named evidence for each one.

| Order | Action | Files or artifacts affected | Pass evidence |
| --- | --- | --- | --- |
| 1 | Implement and test feasible learnable CTC expansion | CTC head, Stage-2 trainers, unit tests | No infeasible accepted targets, no hidden zero loss, finite non-zero gradients |
| 2 | Carry and apply visual/acoustic quality | JSONL loader, feature shards, aligner/gating, cache schema, tests | `lip_conf` survives caching; all-zero confidence produces zero visual residual |
| 3 | Make Stage-1 speaker-aware | Stage-1 dataset/sampler/objective and tests | No global AMI clustering and no same-speaker false negatives |
| 4 | Add dev selection and safe resume | Stage-1/2 CLIs, checkpoint schema, evaluation code | `best`/`last` checkpoints, dev-only selection, mismatch-safe resume |
| 5 | Complete cache provenance | cache signature/index and tests | Manifest/data/code/model/quality mismatches are rejected |
| 6 | Add the Llama-3-8B profile and new config | model backend, `config_real_en_llama3_8b.yaml`, config/backend tests | Local config validates as Llama, hidden size 4096, all LoRA targets found |
| 7 | Run GPU preflights | Disposable load/one-step outputs | BF16 load and backward fit; peak memory and throughput recorded |
| 8 | Regenerate the 256-record smoke cache under a new name | New smoke cache only | Cache signature matches repaired code and v4 data |
| 9 | Run the complete 8B smoke | New smoke checkpoints/logs only | Every P0 gate passes and dev metrics execute |
| 10 | Run full AMI stages | Versioned production cache/checkpoints/logs | Stage-1 and align/CTC warm-up complete; 8B joint run selected on dev |
| 11 | Evaluate once on frozen test | Versioned result report | WER, SA-WER, GER WERR, SCR, fallback and over-correction metrics recorded |
| 12 | Run 3B comparison | Separate 3B config/cache/checkpoints/results | Same frozen protocol and seeds; memory/time/quality comparison reported |

The implementation should happen in the same `mix_datasets` worktree and must
not modify or overwrite `data/ami_full_v4`, the existing 3B configuration, or
any completed v4 manifests. Smoke caches and checkpoints are disposable and
must use new model-specific names. Production runs start only after the smoke
report proves every P0 invariant.
