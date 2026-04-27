# Training

> **Practical, end-to-end real-data rollout (Phase D → G):** [`PHASE_D_REAL_MODELS.md`](PHASE_D_REAL_MODELS.md). That doc covers GPU memory budgets, real manifest construction, mouth-ROI preprocessing, the post-Stage-2 re-enrolment step, and common failure modes. This file is the reference for *what* the two stages are; that file is the reference for *how* to run them.

Two stages, **in this order**, per spec §7.

| Stage | Frozen | Optimised | Loss |
|---|---|---|---|
| Stage 1 | ASR encoder, VSR encoder | identity fuser, aligner, projection heads | bidirectional InfoNCE (`L_{A→V} + L_{V→A}`) + CTC |
| Stage 2 | none | everything (incl. LoRA on Llama-3-8B) | CTC + GER cross-entropy + bidirectional InfoNCE |

> **Spec §7 invariant (non-negotiable):** `lr_stage2 == lr_stage1 × 0.1`.
> `scripts/train_stage2.py` enforces this at runtime and raises `ValueError` if `configs/default.yaml` is edited to violate it. The default values (`stage1.lr = 1e-3`, `stage2.lr = 1e-4`, `stage2.lr_ratio_to_stage1 = 0.1`) already satisfy it.

---

## Stage 1

**Goal**: learn the ID-conditioned cross-modal alignment so the C2 aligner produces useful `f_align` before the LLM is in the loop.

```bash
python scripts/train_identity.py \
    --config configs/default.yaml \
    --manifest data/train_manifest.json
```

* **Frozen**: Whisper encoder, AV-HuBERT encoder.
* **Trained**: identity fuser, ID-conditioned aligner (`Concat+Linear` injector + cross-attn blocks), CTC head over expanded token-level features.
* **Objective**: `L_total = L_{A→V} + L_{V→A} + L_CTC` (config: `objective: [infonce_av, infonce_va, ctc]`).
* **Stop rule**: `av_sid_acc_plateau` on a held-out set.
* **InfoNCE temperature**: `0.07`. Bidirectional is non-optional — single-direction collapses the alignment.

---

## Stage 2

**Goal**: end-to-end fine-tune with the GER head in the loop.

```bash
python scripts/train_stage2.py \
    --config configs/default.yaml \
    --manifest data/train_manifest.json
```

* **Trained**: identity fuser, aligner, CTC head, GER LoRA (Llama-3-8B), Q-Former projector, id_proj.
* **Loss**: `L_total = w_ctc · L_CTC + w_ger · L_GER_CE + w_info · L_InfoNCE`, defaults `w_ctc = 1.0, w_ger = 1.0, w_info = 0.5`.
* **Teacher forcing**: GER cross-entropy is computed with `labels = [-100 over prompt span, target_ids over answer span]`, off-by-one shifted. The same `_render_text` / `_inputs_embeds` paths used at inference are reused so the prompt is identical between train and eval (no train-test prompt skew).
* **LR**: `1e-4`, exactly `0.1 ×` Stage 1. Runtime guard:
  ```python
  expected = stage1_lr * ratio
  if abs(stage2_lr_cfg - expected) > 1e-9:
      raise ValueError("Stage 2 LR must equal Stage 1 LR * 0.1 ... Spec §7 forbids deviation.")
  ```

**Checkpoints land at:**
* `out/identity_pool_stage2.pt`
* `out/aligner_stage2.pt`
* `out/ctc_head_stage2.pt`
* GER LoRA adapters via `peft`'s `save_pretrained` (path configured in the script).

---

## Switching off stub mode

`configs/default.yaml` has `stub_backbones: true` by default so wiring tests pass without weights. To actually train:

1. Set `stub_backbones: false`.
2. Make sure each backbone can resolve weights:
   * Whisper: HF cache populated by `faster-whisper` and `transformers` on first call.
   * AV-HuBERT: `checkpoints/avhubert_large_lrs3_iter5.pt` exists and matches `vsr.config`.
   * ECAPA: `speechbrain/spkrec-ecapa-voxceleb` reachable.
   * InsightFace: `buffalo_l` model pack downloads on first call.
   * Llama-3-8B: `hf auth login` with a token that has access to `meta-llama/Meta-Llama-3-8B-Instruct`. (Legacy `huggingface-cli login` is deprecated; the new unified CLI ships as `hf`.)
3. Confirm `device: cuda` is set (or `mps` for Apple Silicon dev).

---

## Notes / TODOs visible in the code

* `avsd_ger/backbones/asr_whisper.py` — the N-best is currently the 1-best repeated `n_best` times (search for `# TODO: replace duplicated 1-best padding with real CT2 beam outputs`). Wiring real beam outputs from CT2 is required before Stage 2 training, otherwise `L_GER_CE` sees a degenerate prompt and `nbest_variance` in the C3 confidence is identically zero.
* CTC head expansion factor is `4×` by default (`avsd_ger/training/ctc_loss.py`). If your corpus has very long words you may need to bump this; the loss will throw an inputs-shorter-than-targets error otherwise.
