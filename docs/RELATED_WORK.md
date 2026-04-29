# Related Work

## At a glance

![Comparison: Our Framework vs DualHyp vs AVSD vs DiarizationLM](figures/framework_comparison.png)

> Drop the comparison figure at `docs/figures/framework_comparison.png`. It captures the four-way comparison summarised below.

---

## Lineage and delta

This framework **adopts the dual-hypothesis prompting paradigm popularised by DualHyp (NeurIPS 2025)** — i.e. feeding both an ASR N-best and a VSR hypothesis as text into an LLM for generative error correction — and extends it along three orthogonal axes:

1. **Identity conditioning (C1)** — a fused ECAPA + ArcFace embedding `z_id`, registered as a `[Speaker: ID_i]` special token, and an EMA-refreshed pool that gives the model *persistent memory of who is speaking*.
2. **Continuous soft prefix (C2)** — instead of relying on text alone, an ID-conditioned, token-level cross-attention over Whisper × AV-HuBERT features produces `f_align`, which is projected into the LLM as a `<AV_CTX>` soft prefix. The aligner is also **per-speaker key-masked** so it cannot hallucinate lip evidence from other speakers.
3. **Closed-loop confidence gate (C3)** — composite confidence + acoustic-rescore gate decides whether to accept, re-align, re-identify, or refresh the pool. Removing the gate (the `c3_wo_conf_gate` ablation) must *degrade* SA-WER relative to disabling C3 entirely; this is the structural-safety claim of the framework.

No DualHyp code or checkpoint is used. The lineage is **conceptual** (prompt structure inherited), not **implementational**.

---

## Method comparison (text version of the figure)

|  | **Ours** | **DualHyp** (NeurIPS 2025) | **AVSD** | **DiarizationLM** |
|---|---|---|---|---|
| **Main focus** | Who said what, and is it correct? | What is said (better transcript)? | Who speaks when? | Who speaks when (with LLM)? |
| **Input** | Raw audio + video | Raw audio + video | Raw audio + video | Diarization segments / ASR text chunks |
| **Identity** | Explicit, persistent, cross-modal ID embeddings (pool with online update) | None | Implicit (per-segment cluster output) | Speaker labels in text only |
| **LLM role** | Speaker-aware reasoning + error correction with identity constraints and aligned features | Fuse modalities + correct errors at text level | Not used | Model speaker consistency in text for diarization |
| **Output** | Diarization-aware transcript | Single-stream transcript | Speaker timeline | Speaker timeline |
| **Primary metrics** | SA-WER ↓, SCR ↓, DER ↓, WDER ↓, JER ↓, AV-SID Acc ↑ | WER ↓, CER ↓ | DER ↓, JER ↓, AV-SID Acc ↑ | WDER ↓, JER ↓ |
| **Strengths** | Solves speaker confusion (SCR); long sessions via closed-loop; ID propagated across all modules; backbone-agnostic | Strong single-speaker performance; simple and efficient; LLM correction | Specialised for diarization; cross-modal cues; strong on who-speaks-when | LLM captures long-range context; strong on overlapping speech; better diarization consistency |
| **Limitations** | More complex training; needs stable ID propagation; higher compute | Not speaker-aware; one-shot, no memory; no explicit identity modeling | No transcription; cannot correct recognition errors; no text-level reasoning | Needs external diarization/ASR; no cross-modal info; cannot fix transcription errors |

---

## Why DualHyp cannot be retrofitted into this role

Two structural reasons:

1. **No identity slot in the prompt.** DualHyp's prompt is `(audio_hyp, visual_hyp) → corrected_text`. There is nowhere for a per-speaker embedding or speaker token to live. Adding one is not a parameter change — it changes what the LLM is being asked to do.
2. **No state across utterances.** DualHyp is one-shot. There is no pool, no EMA, no `tau_update` gate. The closed loop in C3 is a *property of the system*, not a hyperparameter.

This is what we mean by "**backbone-agnostic**": you can swap Whisper / AV-HuBERT / Llama for other comparable models, but you cannot drop the C1/C3 scaffolding without giving up the contribution.

---

## Datasets we target

* **AMI Meeting Corpus** — primary evaluation target for this codebase: multi-speaker real meetings with stable speaker annotations. SA-WER, SCR, DER, JER, and AV-SID Acc are the headline metrics.
* **MISP-Meeting / MISP-2025** — secondary real-conversation validation, especially for Mandarin multi-speaker meetings. This is the stronger out-of-domain test once the MISP manifest/annotation adapter is in place.
* **LRS2** — optional AVSR sanity/smoke setting using the available complete subset and local preprocessing into audio + mouth ROI. It checks that the audio-visual path works, but it is not the main contribution because LRS2 is mostly single-speaker utterance-level data.

LRS3 and MuAViC are not current experimental commitments unless complete data access and preprocessing are available. They can be mentioned as future extensions, but not as target datasets for the present paper/evaluation plan.

---

## Citation notes

When citing DualHyp / AVSD / DiarizationLM in the paper, the comparison figure and the table above are the canonical positioning. The README points here so contributors land on this file before writing their own related-work prose.
