# AVSD-GER Long-Term Dataset Roadmap

This document is the project-level memory for the agreed dataset and training
roadmap. The six stages are controlled experiments, not a requirement to mix
all datasets in one dataloader. In the notation below, `A -> B` means
pre-train/warm up on A and then fine-tune and evaluate on B; `A + B` means that
the named modules are combined in the same system.

## Shared experimental rules

- Keep the official train/dev/test or train/dev/eval boundaries for every
  dataset. Never use evaluation meetings, utterances, or speakers for training.
- Freeze the AMI test protocol after Stage 1 so every later AMI result remains
  directly comparable.
- Report results separately by dataset. AMI and LRS2 use English word-based
  metrics, while MISP uses Chinese character- and diarization-oriented metrics.
- Do not claim direct numerical superiority over LRS3-based AVGER, Llama-AVSR,
  or MMS-LLaMA unless they are reproduced under the same data and protocol.
- Treat C1 identity learning, C2 audio-visual alignment, GER, and C3 confidence
  gating as separately ablatable contributions.
- A pretrained backbone using VoxCeleb2 is not the same experiment as adding
  raw VoxCeleb2 samples to task-specific training. State this distinction in
  all experiment descriptions.

## Stage 1: Complete AMI baseline

**Goal:** Establish the trustworthy English meeting baseline and the fixed
comparison point for all later stages.

**Data and training:** AMI only.

**Implementation recipe:** See `docs/AMI_DATA_PIPELINE.md`. Rebuilt manifests
must pass `scripts/audit_ami_manifests.py` before training starts.

### Official AV-HuBERT preprocessing versus the AMI adaptation

The Stage-1 visual pipeline should be described as **AV-HuBERT-style visual
preprocessing adapted to AMI oracle-turn clips**, not as a byte-for-byte copy
of the complete AV-HuBERT dataset preparation workflow.

| Aspect | Official AV-HuBERT preparation | This project's AMI preparation |
| --- | --- | --- |
| Input unit | Processes each supplied video as one sequence; LRS2/LRS3 inputs are normally utterance-level clips | AMI Closeup files are meeting-length streams, so each stream is first segmented using the manifest's oracle turn boundaries |
| Face detector | dlib frontal HOG detector, with the dlib CNN detector as fallback | Same HOG-first, CNN-fallback detector path; do not replace it with Haar or HOG-only for the production baseline |
| Landmarks | dlib 68-point shape predictor | Same 68-point predictor |
| Missing landmarks | Interpolation/extrapolation within the input video sequence | Same procedure, but its temporal scope is one AMI turn rather than an entire meeting stream |
| All frames missing landmarks | The reference `align_mouth.py` resizes the complete input frames as a fallback | The production baseline uses the same full-frame 96x96 resize; its added confidence track is all zero so the fallback remains auditable |
| Alignment and crop | Mean-face similarity transform using stable points 33, 36, 39, 42 and 45; mouth points 48-67; 96x96 ROI; 12-frame smoothing window | Same alignment landmarks, mouth range, ROI size and smoothing rule |
| Intermediate/output form | Landmark PKL files followed by aligned mouth-ROI videos in the reference scripts | Detection and alignment are orchestrated per turn and stored as model-ready NumPy ROI arrays |
| Dataset-specific metadata | LRS/Vox-style file identities and manifests | Official AMI meeting-specific Closeup mapping, corpus-global participant IDs, IHM/oracle-turn declarations and enrollment metadata |
| Missing source media | The reference file check writes missing audio/video IDs to `missing.list`; it does not fabricate paired samples | Official AMI corpus defects are a fixed exclusion ledger; unknown missing/corrupt media remain fatal failures |
| Confidence | The reference preparation does not provide the project's C3 confidence track | Adds an auditable per-frame detection/interpolation confidence track; AV-HuBERT resize fallback frames receive zero |

The original legacy AMI visual pipeline already used the order `slice turn ->
detect landmarks -> align/crop mouth ROI`. The repaired `ami_full_v2` pipeline
retains that order. It repairs the legacy turn caps, camera assumptions,
meeting-local identities and fabricated all-ones confidence rather than
silently changing the visual preprocessing unit.

For reproducibility, production preprocessing must use the AV-HuBERT-pinned
`dlib==19.22.1`, the official CNN detector, 68-point predictor and mean-face
assets. The current `dlib==20.0.1` native crash is an environment compatibility
failure, not justification for changing the detector. First verify that
`cnn_face_detection_model_v1` loads successfully in the isolated preprocessing
environment, then run a one-meeting smoke test before launching all splits.

All Stage-1 through Stage-4 AMI comparisons must keep this preprocessing and
the fixed AMI-AV-valid test protocol unchanged. AMI officially documents that
`TS3003d` has no Closeup1, Closeup2, or Closeup3; turns requiring those streams
are deterministically excluded rather than synthesized, while Closeup4 remains.
Every ablation, including audio-only rows compared with the AV system, uses the
same fixed AV-valid subset. Full official-split audio-only results may be shown
separately and must not be compared numerically as if they used the same turns.
A future experiment that computes
landmarks over complete meeting streams before turn slicing is a separate
preprocessing ablation and must use a new dataset build ID (for example,
`ami_full_v5`) rather than overwrite the production baseline.

**Required data repairs and checks:**

- Remove the current 50-turn-per-meeting truncation and regenerate complete
  train/dev/test manifests.
- Replace the constant lip-confidence fallback with a real, auditable visual
  quality/confidence signal, or explicitly mark confidence as unavailable.
- Verify stable participant identities rather than accidentally treating every
  meeting-local label as a globally distinct person.
- Record the microphone condition explicitly. The current preparation path uses
  AMI IHM (individual headset microphone); it must not be described as a
  far-field meeting experiment.
- Validate audio, face, and lip paths, durations, speaker mapping, split
  isolation, and dataset statistics after regeneration.
- State clearly when evaluation uses oracle speech turns. In that setting,
  diarization is a precondition rather than a predicted system output.

**Evaluation focus:** WER, SA-WER, speaker confusion rate (SCR), AV speaker-ID
accuracy, UNKNOWN/rejection behavior, and C3 fallback usage. Include clean and
controlled degraded audio/video conditions where practical.

## Stage 2: LRS2-C2/GER -> AMI

**Goal:** Test whether sentence-level audio-visual and GER warm-up transfers to
English meetings.

**Training flow:** Use LRS2 for C2 audio-visual alignment/Q-Former and English
GER warm-up, then fine-tune the complete system on AMI. Do not initially treat
LRS2 and AMI as an undifferentiated sample pool.

**Important boundary:** LRS2 is useful for target-speaker AV alignment and GER,
but it is not the main source for persistent cross-session C1 identity labels.

**Evidence required:** Compare against the fixed Stage 1 AMI test set and report
the change in WER, SA-WER, GER relative WER reduction, and fallback behavior.

## Stage 3: VoxCeleb2-C1 -> AMI

**Goal:** Isolate the value of large-scale voice-face identity pretraining.

**Training flow:** Pretrain C1/IdentityFuser with a curated, quality-controlled
VoxCeleb2 subset if raw VoxCeleb2 training is used, then fine-tune and evaluate
on AMI. Existing VoxCeleb2-pretrained ECAPA-TDNN or AV-HuBERT backbones must be
reported separately from this task-specific C1 experiment.

**Scope:** VoxCeleb2 has no suitable manually verified meeting transcripts for
the full AVSD-GER task, so it is primarily a C1 resource. Raw VoxCeleb2 is
optional until Stage 1/2 evidence shows C1 identity performance is a bottleneck.

**Evidence required:** Compare against Stage 1, focusing on AV speaker-ID
accuracy, SCR, SA-WER, open-set rejection, and robustness under face/voice
degradation.

## Stage 4: VoxCeleb2-C1 + LRS2-C2/GER -> AMI

**Goal:** Build the main English AVSD-GER system by combining complementary
module-specific pretraining.

**Training flow:** VoxCeleb2 pretrains C1; LRS2 warms up C2 and English GER; the
combined model is then fine-tuned and evaluated on AMI.

**Role in the thesis:** This is the intended main English system. Compare it
with Stages 1-3 to distinguish the individual and combined contributions of C1
identity pretraining and C2/GER warm-up.

**Evaluation focus:** The complete AMI metric set from Stage 1, plus clean and
degraded modality ablations and confidence-gated fallback analysis.

## Stage 5: VoxCeleb2-C1 + MISP-C2/GER -> MISP

**Goal:** Extend AVSD-GER to Mandarin, high-overlap, real meeting conditions.

**Training flow:** Reuse or adapt the VoxCeleb2-trained C1 identity module;
train/adapt C2 and a Chinese GER component on the official MISP training split;
tune on development data and evaluate only on the official evaluation split.

**Language boundary:** Do not reuse an English-only GER head unchanged. Use a
Chinese GER adapter/head or an explicitly multilingual tokenizer and decoder.
C1 is largely language-independent; much of C2 can be shared or initialized
cross-lingually; GER remains language-sensitive.

**Evaluation focus:** CER, cpCER and DER as required by the MISP protocol, plus
speaker-attribution, rejection, and modality-degradation analyses when labels
permit. MISP results are reported separately from AMI rather than averaged into
one score.

## Stage 6 (optional): Shared C1/C2 with bilingual GER heads

**Goal:** Explore multilingual AVSD-GER only if time and compute remain after
the English main system and Chinese extension are complete.

**Recommended architecture:**

```text
VoxCeleb2 --------> shared C1 identity module -----------+
                                                         |
LRS2 ------------> shared/English-initialized C2 --------+--> English GER head --> AMI
                                                         |
MISP ------------> cross-lingual C2 adaptation ----------+--> Chinese GER head --> MISP
```

The preferred initial design shares C1 and most of C2 while keeping separate
English and Chinese GER heads/adapters. A single language-conditioned GER head
is a later ablation, not a prerequisite.

**Evaluation:** Evaluate AMI and MISP separately with their native metrics and
compare each result with its corresponding monolingual system. The multilingual
system is successful only if cross-lingual sharing preserves or improves the
two monolingual baselines; a combined English/Chinese score is not sufficient.

## Priority and stopping rule

Stages 1-4 form the core English thesis path. Stage 5 is the planned Chinese
high-overlap meeting extension. Stage 6 is explicitly optional and must not
delay completion of the validated English system. Raw VoxCeleb2 collection and
processing should likewise be deferred unless C1 ablations justify its cost.
