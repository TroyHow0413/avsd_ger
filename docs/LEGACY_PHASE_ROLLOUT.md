# Legacy Phase Rollout Notes

This document preserves the old Phase 0/A/B/C/D/E/F/G rollout notes from the README. These labels are historical shorthand only; the current training workflow uses concrete scripts such as `scripts/train_identity.py`, `scripts/train_stage2.py`, and the optional `one_go/train.py` wrapper.

## Legacy Rollout Notes: old Phase 0/A/B/C/D/E/F/G

Follow the phases in order. Each phase says what you **get** from finishing it (the payoff), what it **needs** (prereqs), and exactly what to run. W&B flags are uniform across all training/eval scripts — see [W&B flags](../README.md#wb-flags) in the current README.

| Phase | Payoff (what you unlock) | Needs | Approx. time |
|---|---|---|---|
| **0** | Confirm C1→C2→C3 pipeline is wired correctly on synthetic tensors | conda env | 1 min |
| **A** | First numbers on the spec §10 ablation table + the structural-safety check | Phase 0 | 5 min |
| **B** | Llama-3-8B access approved (run **in parallel** with A — it waits on Meta's review) | Phase 0 | 5 min submit + hours-to-day approval |
| **C** | All 5 backbone weights on disk | Phase 0 | 30–60 min download |
| **D** | First real-model end-to-end smoke test (no training yet) | A done, B approved, C done | 10 min |
| **E** | Stage-1 trained: identity-aware alignment | D | hours–day, 1×A100 |
| **F** | Stage-2 trained: full multi-task with GER LoRA on Llama-3-8B | E | 1–3 days, 1–2×A100 |
| **G** | Final ablation eval on a real test set + headline metrics in W&B summary | F | tens of minutes |

> **Recommended ordering:** Phase 0 → run A and B **at the same time** (B is async, just submit and wait) → C → D → E → F → G.

---

### Phase 0 — Stub-mode smoke test (already done if you've followed install)

**Payoff:** confirms C1 enrollment + retrieval + C2 alignment + GER + C3 closed-loop are wired correctly. Pure synthetic tensors, no weights needed.


```bash
python scripts/enroll_identity.py --manifest data/sample_manifest.json
python scripts/run_sample.py     --manifest data/sample_manifest.json --utt utt_0001
```

**Expected output** (deterministic, seed=1337):
```
text       : the quick brown fox jumps over the lazy dog
speaker_id : spk_02
confidence : 0.837
decision   : accept_and_update
```

If `confidence` ≈ 0.837 and `decision == accept_and_update`, your environment is healthy.

---

### Phase A — Eval infrastructure on stub data (no downloads)

**Payoff:** the **5-row ablation table** (full / w/o C1 / w/o C2 / w/o C3 / C3 w/o gate) with all five primary metrics (SA-WER, SCR, AV-SID Acc, DER, JER), plus the spec-mandated structural-safety check (`c3_wo_conf_gate ≥ wo_c3` PASS). All on stub tensors — exercises the eval code without any external dependency.

```bash
python scripts/eval_ablations.py \
    --config   configs/default.yaml \
    --manifest data/sample_session_manifest.json \
    --pool     checkpoints/identity_pool.pt \
    --out      out/ablation_report_stub.json \
    --no-power \
    --wandb-project  avsd-ger \
    --wandb-run-name stub-ablation-smoke \
    --wandb-tags     stub eval ablation
```

**Sample session manifest:** `data/sample_session_manifest.json` (3 turns, 2 speakers, already in the repo).

**Look for in stdout:**
```
=== running ablation: full_model    flags={} ===
{ ... five metrics ... }
... (4 more rows) ...
[spec check] C3-w/o-gate SA-WER (...) >= w/o-C3 SA-WER (...): PASS
[wrote] out/ablation_report_stub.json
```

**Look for in W&B:** the `ablation/<row>/<metric>` charts populate; the `summary/spec_check_c3_gate_pass` summary key is `true`.

---

### Phase B — Apply for Llama-3-8B access (run in parallel with A)

**Payoff:** unlocks the GER head — without this you can't load Llama-3-8B-Instruct, so Stage-2 training and any non-stub `run_sample.py` will fail at the GER step. **Submit early; approval is async.**

```bash
python -m pip install --upgrade "huggingface_hub>=0.34,<1.0"  # only needed if `hf` is not found
hf auth login                  # paste a Read-scope token from https://huggingface.co/settings/tokens
hf auth whoami                 # confirms the token
# Fallback for old pinned envs: huggingface-cli login
```

Then in a browser, visit [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) → click **Request access** → fill the form. Approval is typically a few hours to a day.

You can finish Phase A and start Phase C while waiting.

---

### Optional — pre-download Whisper into `checkpoints/`

If the training server has a slow Hugging Face/Xet connection, pre-download the
Whisper weights on a faster machine, then upload the whole `checkpoints/`
directory. The ASR wrapper first checks these local directories and only
downloads when they are missing:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-large-v3", local_dir="checkpoints/whisper/Systran-faster-whisper-large-v3")
snapshot_download("openai/whisper-large-v3", local_dir="checkpoints/whisper/openai-whisper-large-v3")
PY
```

Expected local layout:

```text
checkpoints/whisper/Systran-faster-whisper-large-v3/
checkpoints/whisper/openai-whisper-large-v3/
```

On the server, no extra flag is needed as long as `configs/default.yaml` keeps
`asr.checkpoint_dir: checkpoints/whisper`.

---

### Phase C — Download backbone weights

**Payoff:** all 5 backbones can load real weights when you flip `stub_backbones: false`.

| Backbone | How to get the weights | Notes |
|---|---|---|
| Whisper-large-v3 | Auto-pulled by `faster-whisper` + `transformers` on first call | Cached under `checkpoints/whisper/` so it can be packed and uploaded with the project |
| AV-HuBERT Large | Manual: download `large_lrs3_iter5.pt` from the AV-HuBERT repo's Model Zoo, drop at `checkpoints/avhubert_large_lrs3_iter5.pt` | Path comes from `configs/default.yaml → vsr.checkpoint` |
| ECAPA-TDNN | Auto from `speechbrain/spkrec-ecapa-voxceleb` on first call | ~80 MB |
| InsightFace `buffalo_l` | Auto on first call to `face_encoder.embed()` | ~280 MB |
| Llama-3-8B-Instruct | Auto-pulled once Phase B is approved | ~16 GB; `hf auth login` already done (`huggingface_hub>=0.34` required for the `hf` command) |

Verify after:
```bash
ls -lh checkpoints/avhubert_large_lrs3_iter5.pt
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('meta-llama/Meta-Llama-3-8B-Instruct').gated)"
```

---

### Phase D — Flip `stub_backbones: false` and smoke-test

**Payoff:** confirms all 5 real backbones load and the pipeline produces a sensible transcript on real audio/video.

> **Current operational walkthrough:** [`REAL_MODEL_WORKFLOW.md`](REAL_MODEL_WORKFLOW.md). The old linked `PHASE_D_REAL_MODELS.md` now redirects to the current workflow.

Quick version (Phase D.1 — easy mode, real audio + stubbed video):

1. In `configs/default.yaml` set `stub_backbones: false`.
2. Drop a 3-10 s mono 16 kHz WAV at `data/utts/utt_0001.wav`, a frontal-face image at `data/spk_01/enroll.jpg`, set `mouth_roi: null` in the manifest.
3. Re-run:
   ```bash
   python scripts/enroll_identity.py --manifest data/sample_manifest.json
   python scripts/run_sample.py     --manifest data/sample_manifest.json --utt utt_0001
   ```

**First run** triggers ~25 GB of downloads (Whisper-large-v3 ×2 paths ≈ 6 GB, ECAPA ≈ 80 MB, InsightFace ≈ 280 MB, **Llama-3-8B ≈ 16 GB**). Subsequent runs are fast.

**Look for:** `text` is now your actual transcript, `s_acoustic` is a real Whisper rescore (typically -0.1 to -0.6 for clean speech), `top_ids` has the speaker you enrolled.

If the GER step OOMs (24 GB GPU + Llama-3 fp16 won't fit), enable 4-bit loading — see [`REAL_MODEL_WORKFLOW.md#gpu-notes`](REAL_MODEL_WORKFLOW.md#gpu-notes).

---

### Phase E — Stage-1 training (identity-aware alignment)

**Payoff:** trained C1 fuser + identity-conditioned C2 aligner. Loss is bidirectional InfoNCE + CTC; backbones stay frozen (per spec §7).

**Stub-mode rehearsal** (no real data — verifies wandb + training loop on synthetic batches):
```bash
python scripts/train_identity.py \
    --config   configs/default.yaml \
    --manifest data/sample_train_manifest.jsonl \
    --out      checkpoints/stage1/ \
    --wandb-project  avsd-ger \
    --wandb-run-name stage1-stub-rehearsal \
    --wandb-tags     stage1 stub
```

> The script auto-falls back to 8 synthetic records if the manifest path is missing, so this also works if you point `--manifest` at any non-existent path.

**Real training** (swap to your real JSONL when you have it — one record per line, fields `wav_path`, `face_path`, `lip_conf`):
```bash
python scripts/train_identity.py \
    --config   configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --out      checkpoints/stage1/ \
    --wandb-project  avsd-ger \
    --wandb-run-name stage1-real-v1 \
    --wandb-tags     stage1 real
```

**W&B charts to watch:**
- `stage1/loss/total` — should drop steadily; A→V and V→A losses should converge to similar values (bidirectional balance).
- `stage1/acc/A->V` and `stage1/acc/V->A` — should rise above chance (1 / batch_size = 1/64 ≈ 0.016) within ~500 steps.
- `stage1/cold_start/K` — number of pseudo-speakers found by agglomerative clustering (sanity check vs. your dataset's true speaker count).

Stop criterion (per spec §7): **AV-SID accuracy plateau on a held-out set**. The current script uses `epochs` cap as a placeholder; plug in your dev-set evaluator for real plateau detection.

Output: `checkpoints/stage1/identity_pool_stage1.pt`.

---

### Phase F — Stage-2 training (multi-task, full unfreeze, GER LoRA)

**Payoff:** end-to-end fine-tuned model with the LLM in the loop. Loss is `L_CTC + L_GER_CE + 0.5 * L_InfoNCE`; everything is unfrozen.

**Stub-mode rehearsal** (manifest path is intentionally non-existent — the script falls back to 8 synthetic records):
```bash
python scripts/train_stage2.py \
    --config   configs/default.yaml \
    --manifest data/sample_train_manifest.jsonl \
    --out      checkpoints/stage2/ \
    --wandb-project  avsd-ger \
    --wandb-run-name stage2-stub-rehearsal \
    --wandb-tags     stage2 stub
```

> In stub mode you'll see `ctc=0.0000 ger=0.9300 info=...` — the CTC and GER losses are deterministic placeholders (the heads return fixed values when `stub_backbones: true`). Only `info` (InfoNCE on random embeddings) varies. This proves the training loop + autograd + optimizer + W&B are wired correctly; real loss curves require Phase D first.

**Real training:**
```bash
python scripts/train_stage2.py \
    --config   configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --out      checkpoints/stage2/ \
    --wandb-project  avsd-ger \
    --wandb-run-name stage2-real-v1 \
    --wandb-tags     stage2 real lora
```

**Spec §7 invariant** is enforced at startup: if `stage2.lr ≠ stage1.lr × 0.1` the script raises `ValueError` and refuses to run. Defaults already satisfy this (`1e-3 × 0.1 == 1e-4`).

**W&B charts to watch:**
- `stage2/loss/total`, `stage2/loss/ctc`, `stage2/loss/ger`, `stage2/loss/info` — all should decrease; GER loss is the slowest to move.
- `stage2/lr` — should stay flat at `1e-4` after warmup.

Output: `checkpoints/stage2/identity_pool_stage2.pt`, `aligner_stage2.pt`, `ctc_head_stage2.pt`, plus the LoRA adapter under `out/peft/` (saved by the GER head).

---

### Phase G — Final ablation eval on a real test set

**Payoff:** the headline numbers for the paper. SA-WER, SCR, AV-SID Acc, DER, JER for all 5 ablation rows + energy per row + the spec safety PASS/FAIL.

> Phase G uses a **session manifest** (turns + ref_text + ref_speaker), not the JSONL training manifest. Format: see [`EVALUATION.md#session-manifest`](EVALUATION.md#session-manifest).

**Stub-mode rehearsal** (uses the included 3-turn / 2-speaker session — same manifest as Phase A):
```bash
python scripts/eval_ablations.py \
    --config   configs/default.yaml \
    --manifest data/sample_session_manifest.json \
    --pool     checkpoints/identity_pool.pt \
    --out      out/ablation_report_stub.json \
    --no-power \
    --wandb-project  avsd-ger \
    --wandb-run-name ablation-stub-rehearsal \
    --wandb-tags     stub eval ablation
```

> **Important — Phase G needs an *enrolled* pool, not a freshly-trained one.** `train_stage2.py` only updates the fuser weights inside the pool; it never calls `pool.enroll()`, so `checkpoints/stage2/identity_pool_stage2.pt` is **empty of speakers**. For stub rehearsal, reuse the Phase 0 pool (`checkpoints/identity_pool.pt`). For real eval, you must re-enroll your test speakers using the trained fuser before evaluating — see "Real eval" below.

**Real eval** (after Phase F — re-enroll speakers using the trained fuser, then point Phase G at that pool — current walkthrough in [`REAL_MODEL_WORKFLOW.md#re-enroll-and-evaluate`](REAL_MODEL_WORKFLOW.md#re-enroll-and-evaluate)):
```bash
# 1. Re-enroll test speakers using the Stage-2 fuser. enroll_identity.py loads
#    fuser weights from --in-pool if it exists, then runs the enrollment loop.
python scripts/enroll_identity.py \
    --manifest data/your_real_test_speakers.json \
    --in-pool  checkpoints/stage2/identity_pool_stage2.pt \
    --out-pool checkpoints/stage2/identity_pool_stage2_enrolled.pt

# 2. Eval against the enrolled pool.
python scripts/eval_ablations.py \
    --config   configs/default.yaml \
    --manifest data/your_real_test_session_manifest.json \
    --pool     checkpoints/stage2/identity_pool_stage2_enrolled.pt \
    --out      out/ablation_report_real.json \
    --idle-calibrate-s 2.0 \
    --wandb-project  avsd-ger \
    --wandb-run-name ablation-final-v1 \
    --wandb-tags     final eval ablation
```

After the run, the **W&B run summary** has one entry per `(ablation_row, metric)` pair plus `summary/spec_check_c3_gate_pass`. Sort runs by `summary/full_model/sa_wer` to compare experiments.

> **Sanity check that the pool is enrolled**: when `eval_ablations.py` starts you'll see a line like `[pool] loaded from ... — N speakers`. If `N == 0`, the eval will collapse to `wo_c1`-like numbers (every turn returns `is_unknown=True`, AV-SID Acc = 0, DER = JER = 1.0). That's the symptom you saw in stub Phase G when pointing `--pool` at the bare Stage-2 file.

---

