# Architecture

> Full chain: **identity (early) → feature (mid) → alignment (mid) → LLM (late) → feedback (closed-loop)**.

```
          ┌────────────────────┐                   ┌──────────────────────┐
  audio ─►│  Whisper-large-v3  │── enc_feats ─┐    │   C1: Identity Pool  │
          │  + word timestamps │              │    │   (voice + face,     │
          └────────────────────┘              │    │    fused 256-d z_id) │
                                              │    └──────────┬───────────┘
          ┌────────────────────┐              │               │ z_id
  video ─►│  AV-HuBERT Large   │── vsr_feats ─┤               ▼
          │  + speaker mask    │              │    ┌──────────────────────┐
          └────────────────────┘              ├───►│ C2: ID-Cond Aligner  │
                                              │    │ token-level cross-attn│
                                              │    │ Concat+Linear inject │
                                              │    │ soft-gated, masked   │
                                              │    └──────────┬───────────┘
                                              │               │ f_align
                                              ▼               ▼
                                    ┌─────────────────────────────────────┐
                                    │ GER Head: Llama-3-8B + LoRA         │
                                    │ prompt = [Speaker token]            │
                                    │          + Audio N-best (text)      │
                                    │          + Visual hyp (text)        │
                                    │          + <AV_CTX> soft prefix     │
                                    └──────────────────┬──────────────────┘
                                                       │ ŷ
                                                       ▼
                                    ┌─────────────────────────────────────┐
                                    │ C3: Composite confidence            │
                                    │   = 0.6 acoustic_rescore            │
                                    │   + 0.25 av_consistency             │
                                    │   + 0.10 nbest_variance             │
                                    │   + 0.05 llm_entropy                │
                                    │ Decision tree:                      │
                                    │   high  → ACCEPT_AND_UPDATE (EMA)   │
                                    │   mid   → ACCEPT_NO_UPDATE          │
                                    │   low   → REIDENTIFY / REALIGN      │
                                    └─────────────────────────────────────┘
```

---

## C1 — Cross-Modal Identity Pool

**Purpose**: maintain a stable, persistent speaker representation that survives across utterances and across modalities.

* **Voice encoder**: `speechbrain/spkrec-ecapa-voxceleb` → 192-d embedding.
* **Face encoder**: InsightFace `buffalo_l` (ArcFace ResNet-100) → 512-d embedding.
* **Fusion**: a small linear fuser projects `[voice‖face] → 256-d z_id`.
* **Dual-gate** (spec §8 step 2): per-frame keep iff `SNR ≥ τ_a` **and** `lip_conf ≥ τ_v`. Default `τ_a=8 dB`, `τ_v=0.7`.
* **Cold-start** (spec §8 step 3): agglomerative clustering with `linkage=average`, `distance_threshold=0.55`. Speakers whose nearest cluster distance exceeds `0.65` are tagged `__unknown__`.
* **EMA refinement**: `e_new = (1-α) e_old + α e_obs` with `α=0.1`, **gated** on `s_acoustic ≥ τ_update=0.55`.

**Files**: `avsd_ger/c1_identity/{voice_encoder.py, face_encoder.py, identity_pool.py, gate.py, cold_start.py}`.

---

## C2 — ID-Conditioned Alignment + GER

### Aligner (`c2_alignment/id_conditioned_aligner.py`)

Spec §2 C2 mandates four design choices:

1. **ID injection = `Concat + Linear`** (FiLM is relegated to the §10 ablation table).
   ```
   Q = Linear( Concat(f_a_token, e_id) )    # [N_tok, D]
   K = V = f_v_segment                       # [T_v, D]
   ```
2. **Token-level granularity**: Whisper encoder frames are pooled into per-word vectors using Whisper's word timestamps before entering cross-attention.
3. **Per-speaker key-padding mask**: VSR frames not assigned to the current speaker are masked out — this is the structural reason the aligner cannot hallucinate lip evidence from another speaker.
4. **Soft gating**: `attn_weight *= min(SNR_per_token, lip_conf_per_frame)`, implemented as an additive log-bias on the logits so gradients flow.

### GER head (`c2_alignment/ger_head.py`)

* **LLM**: `meta-llama/Meta-Llama-3-8B-Instruct`, LoRA `r=16, α=32, dropout=0.05`, target modules `{q,k,v,o,gate,up,down}_proj`.
* **Soft prefix**: a Q-Former–style projector turns `f_align` into a fixed-length sequence of pseudo-token embeddings (`<AV_CTX>`).
* **Speaker token**: `[Speaker: ID_i]` is **registered** as a special token before training so the LLM cannot fragment it.
* **Prompt (spec §2 C2, verbatim)**:
  ```
  {speaker_tag}
  Audio hypothesis: {asr_nbest}
  Visual hypothesis: {lip_hyp}
  Aligned feature context: <AV_CTX>
  Correct the transcript. Preserve the speaker label.
  Output:
  ```

---

## C3 — Closed-Loop Feedback

**Composite confidence** (`c3_feedback/confidence_scorer.py`):

| Component | Weight | What it measures |
|---|---|---|
| acoustic_rescore | 0.60 | mean token log-prob of the corrected text under Whisper |
| av_consistency  | 0.25 | cosine between observed `(voice,face)` and pool's `z_id` |
| nbest_variance  | 0.10 | hypothesis disagreement across the N-best |
| llm_entropy     | 0.05 | per-token entropy from the GER decoder |

**Decisions** (`c3_feedback/closed_loop.py` → `LoopAction`):

* `total ≥ 0.55` and `s_acoustic ≥ τ_update` → **ACCEPT_AND_UPDATE** (EMA-refresh the pool entry).
* `total ≥ 0.55` but `s_acoustic < τ_update` → **ACCEPT_NO_UPDATE** (output is good enough but identity may be wrong; don't pollute the pool).
* `0.35 ≤ total < 0.55` → **REALIGN** (re-run C2 with same `id_q`).
* `total < 0.35` → **REIDENTIFY** (skip current top-1 ID and re-query the pool).
* Hard cap `max_iters=3`.

The acoustic-rescore-gated EMA is the structural safety mechanism that prevents the loop from drifting on confident-but-wrong outputs. Removing it (`disable_conf_gate: true`) must perform **worse** than disabling C3 entirely — see `docs/EVALUATION.md`.

---

## Ablation flags

Set in `configs/default.yaml` under `ablation:`. One flag at a time reproduces one row of spec §10 Table 2.

| Flag | Effect on `pipeline.run()` |
|---|---|
| `disable_c1` | Replace `id_q.z_id` with zeros → GER is no longer ID-aware. |
| `disable_c2` | Skip the GER head; return ASR 1-best as the hypothesis. |
| `disable_c3` | Cap `max_iters=1`; never EMA-update the pool. |
| `disable_conf_gate` | Promote every `ACCEPT_NO_UPDATE` → `ACCEPT_AND_UPDATE` (unconditional pool update). |
