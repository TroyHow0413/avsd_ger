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
