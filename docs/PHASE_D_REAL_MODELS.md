# Real-Model Rollout — Phases D → G

This is the operational guide for switching off `stub_backbones` and putting real Whisper-large-v3, AV-HuBERT, ECAPA, ArcFace and Llama-3-8B in the loop. By the time you read this, your stub rehearsal should already be green (Phase 0 / A / E / F / G all passing on synthetic tensors).

> **What this doc replaces:** the optimistic one-liner "set `stub_backbones: false` and re-run" in the main README. Real models bring real preprocessing, real downloads, and real GPU memory budgets.

---

## Pre-flight checklist

Run these checks **before** flipping `stub_backbones: false`. Any failure here will surface as a much uglier error inside the pipeline.

```bash
# 1. Llama-3-8B access (you already passed this)
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='meta-llama/Meta-Llama-3-8B-Instruct', filename='config.json')
print('LLAMA-3 ACCESS OK ->', p)
"

# 2. AV-HuBERT checkpoint
ls -lh checkpoints/avhubert_large_lrs3_iter5.pt
# expect ~3.7 GB

# 3. fairseq + AV-HuBERT importable
#    NOTE: AV-HuBERT uses non-relative absolute imports for its sibling
#    modules (hubert_pretraining, hubert_dataset, noise, ...), so BOTH
#    av_hubert/ and av_hubert/avhubert/ must be on PYTHONPATH.
python -c "
import sys, fairseq
sys.path.insert(0, 'av_hubert')
sys.path.insert(0, 'av_hubert/avhubert')
import avhubert
print('fairseq', fairseq.__version__, '/ avhubert OK')
"

# 4. GPU + driver
python -c "
import torch
print('cuda:', torch.cuda.is_available(),
      '/ device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
      '/ free:', torch.cuda.mem_get_info()[0] // 2**20, 'MiB' if torch.cuda.is_available() else '')
"

# 5. ffmpeg + libsndfile (audio decode path)
ffmpeg -version | head -1
python -c "import soundfile; print('soundfile OK')"
```

If any of these fail, fix it before going further. The pipeline doesn't degrade gracefully when a backbone is missing — it crashes on first call.

---

## GPU memory budget

| Backbone | fp16 footprint | Notes |
|---|---|---|
| Whisper-large-v3 | ~3.0 GB | encoder + decoder, runs once per utterance |
| AV-HuBERT Large | ~1.0 GB | feature extractor only, no decoder |
| ECAPA-TDNN | ~80 MB | one shot per enrolment |
| InsightFace `buffalo_l` | ~280 MB | one shot per enrolment |
| Llama-3-8B-Instruct (fp16) | ~16 GB | ~24 GB with KV cache during generation |
| Llama-3-8B-Instruct (4-bit nf4) | ~5–6 GB | ~9 GB with KV cache |

**Practical recipes:**

| GPU | Recommended config |
|---|---|
| RTX 3090 / 4090 (24 GB) | Llama-3 in **4-bit** (`bitsandbytes` nf4) is required. Stage-2 training also needs gradient checkpointing on the LLM. |
| A100 40 GB | Llama-3 fp16 fits. Stage-2 with batch=1 is comfortable. |
| A100 80 GB / H100 | Everything in fp16 with batch=2-4 and gradient accumulation. |

To enable 4-bit, no code change is needed — `bitsandbytes` is already in `environment.yml`. The GER head's loader will pick the smallest-footprint dtype that fits when LoRA is on. If it doesn't, set `device_map="auto"` and `load_in_4bit=True` explicitly in `avsd_ger/c2_alignment/ger_head.py`.

---

## Phase D — First real-model smoke test

**Payoff:** you see the real Whisper transcription and the real Llama-3 GER output for one short clip. This is the moment the pipeline stops being a wiring diagram and starts producing meaningful text.

There are two flavours of Phase D, depending on how much preprocessing you want to do upfront.

### D.1 — Easy mode: real audio + stubbed video (5 minutes of work)

Useful for: confirming Whisper, ECAPA, ArcFace, and Llama-3 all load and produce sensible output. AV-HuBERT path is exercised on random video — expect garbage VSR hypothesis but the rest is real.

1. **Get a short audio clip.** Any 3–10 second mono 16 kHz WAV. If you don't have one handy, record yourself:
   ```bash
   # macOS:
   sox -d -c 1 -r 16000 data/utts/utt_0001.wav trim 0 5
   # Linux (arecord):
   arecord -c 1 -r 16000 -f S16_LE -d 5 data/utts/utt_0001.wav
   ```
   Or grab a public sample (e.g. a LibriSpeech dev clip, or any English podcast snippet trimmed to 5 s).

2. **Drop a placeholder face image** at `data/spk_01/enroll.jpg` — any frontal-face photo. ArcFace will produce a real embedding from it.

3. **Edit `configs/default.yaml`:**
   ```yaml
   seed: 1337
   device: cuda
   stub_backbones: false      # ← was true, now false
   ```

4. **Edit `data/sample_manifest.json`** so the paths actually exist:
   ```json
   "utterances": [
     {
       "utt_id": "utt_0001",
       "speaker_id": "spk_01",
       "audio": "data/utts/utt_0001.wav",
       "mouth_roi": null,            ← null → run_sample falls back to random video
       "transcript_gold": "<your reference text>"
     }
   ]
   ```

5. **First run** — expect ~25 GB of downloads on the first call:
   ```bash
   python scripts/enroll_identity.py --manifest data/sample_manifest.json
   python scripts/run_sample.py     --manifest data/sample_manifest.json --utt utt_0001
   ```

   **What you'll see during the first run:**
   - faster-whisper downloads its CTranslate2 Whisper-large-v3 (~3 GB) → cached in `~/.cache/huggingface/`.
   - HF transformers downloads the same model (~3 GB) for encoder hidden states + rescoring.
   - speechbrain downloads ECAPA (~80 MB) into `pretrained_models/`.
   - InsightFace downloads `buffalo_l` (~280 MB) into `~/.insightface/models/`.
   - HF downloads `meta-llama/Meta-Llama-3-8B-Instruct` (~16 GB) — this is the slowest step. With a fast pipe this is ~10 minutes; on a poor connection an hour or two.

   Subsequent runs are fast because everything is cached.

6. **Expected output:**
   ```
   === AVSD-GER output ===
   text        : <YOUR ACTUAL TRANSCRIPT, e.g. "hello this is a test recording">
   speaker_id  : spk_01
   confidence  : 0.7-0.95   ← real composite confidence, not the stub 0.837
   s_acoustic  : -0.05 to -0.4  ← real Whisper rescore log-prob
   iterations  : 1            ← typically; clean clips accept on iter 0
   trace       : [{...}]
   ```

   If `text` is anywhere close to what you actually said, **Phase D.1 is green**.

### D.2 — Full mode: real audio + real lip ROI

This unlocks the AV-HuBERT path, which is necessary before Stage-2 training has any meaning.

AV-HuBERT expects mouth ROIs as `[T, 1, 96, 96]` greyscale tensors saved as `.npy`. The official preprocessing pipeline lives in the AV-HuBERT repo:

```bash
# Inside the av_hubert clone you already have on PYTHONPATH:
cd av_hubert/preparation
# Follow the README in this folder. The headline step is:
python align_mouth.py \
    --video-direc /path/to/your_videos \
    --output-direc /path/to/aligned_mouths \
    --landmark-direc /path/to/landmarks \
    --filename-path /path/to/filename_list.csv
```

This script runs face detection + 68-point landmark detection + lip crop + greyscale + resize. Output is one `.npy` per video at 25 fps `[T, 1, 96, 96]`.

> **Quick path for first smoke test** — if you only need 1–2 utterances, grab a public sample mouth-ROI tensor from the AV-HuBERT `lrs3` data preparation README, drop it at `data/utts/utt_0001_mouth.npy`, and skip the alignment step. You only need to run the full preprocessor when you scale to real training/eval.

Once you have a real mouth ROI npy, point `data/sample_manifest.json` at it (`"mouth_roi": "data/utts/utt_0001_mouth.npy"`) and re-run `run_sample.py`. The `lip_hyp` field in the trace is now meaningful.

### D.3 — Common Phase D failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: ... is a gated repo` | HF token missing or doesn't have Llama-3 access | `hf auth login` + apply at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct |
| `RuntimeError: CUDA out of memory` during GER step | 24 GB GPU + Llama-3 fp16 | Switch GER head to 4-bit (`bitsandbytes`) — see GPU memory recipe above |
| `ModuleNotFoundError: avhubert` | `av_hubert` clone not on PYTHONPATH | `export PYTHONPATH="$PWD/av_hubert:$PWD/av_hubert/avhubert:$PYTHONPATH"` — see next row, both paths are needed |
| `ModuleNotFoundError: hubert_pretraining` (or `hubert_dataset`, `noise`, ...) | Only outer `av_hubert/` on PYTHONPATH, not inner `av_hubert/avhubert/` | AV-HuBERT's `hubert.py` uses non-relative absolute imports for its siblings (`from hubert_pretraining import ...` without a leading dot), so the inner dir must also be on PYTHONPATH. Persist with a conda activate hook:<br>`mkdir -p $CONDA_PREFIX/etc/conda/activate.d && echo 'export PYTHONPATH="$PWD/av_hubert:$PWD/av_hubert/avhubert:$PYTHONPATH"' > $CONDA_PREFIX/etc/conda/activate.d/avhubert_path.sh` |
| `FileNotFoundError: ...avhubert_large_lrs3_iter5.pt` | Checkpoint path mismatch | Verify `vsr.checkpoint` in `configs/default.yaml` matches your file |
| Hangs on first call | First-time HF / SpeechBrain download | Wait it out; `htop` will show network / disk I/O. Llama-3 is the slowest. |
| Whisper transcription is empty / very short | Wav too short or wrong sample rate | Verify `python -c "import soundfile; print(soundfile.info('data/utts/utt_0001.wav'))"` shows 16000 Hz mono |
| `confidence` very low (<0.4) and `text` is gibberish | Mismatched mouth ROI alignment with audio | Check that the npy is from the same video as the wav and same speaker |

---

## Phase E — Real Stage-1 training (identity-aware alignment)

**Payoff:** trained C1 fuser + identity-conditioned C2 aligner. Backbones still frozen (per spec §7), so the GPU pressure is moderate.

### E.1 — Build a real training manifest

Format: one record per line, JSONL:

```jsonl
{"utt_id": "lrs3_pretrain_00000001", "wav_path": "/data/lrs3/pretrain/00000001.wav", "face_path": "/data/lrs3/pretrain/00000001.jpg", "lip_conf": [0.93, 0.91, 0.95, ...]}
{"utt_id": "lrs3_pretrain_00000002", ...}
```

`lip_conf` should be the per-frame lip-detector confidence at the same fps as your video (or close — the `DualGate` will nearest-neighbour upsample to align).

For LRS3:
- One utterance ≈ one `.mp4`. Preprocess once with `av_hubert/preparation/align_mouth.py` (gives you the lip crops + landmark conf series).
- Voice path can point at the audio extracted via `ffmpeg -i in.mp4 -ac 1 -ar 16000 out.wav`.
- Face path can be a single frame snapshot from the middle of the video.

A 5–10k utterance subset is plenty for stage-1 to give meaningful InfoNCE accuracy curves.

### E.2 — Run training

```bash
python scripts/train_identity.py \
    --config   configs/default.yaml \
    --manifest data/lrs3_pretrain_manifest.jsonl \
    --out      checkpoints/stage1/ \
    --wandb-project  avsd-ger \
    --wandb-run-name stage1-lrs3-v1 \
    --wandb-tags     stage1 real lrs3
```

### E.3 — What healthy training looks like

- `stage1/loss/total` drops from ~4 (random init) to ~0.5–1.5 over the first 1–2 epochs, then plateaus.
- `stage1/acc/A->V` and `stage1/acc/V->A` should both rise above `1 / batch_size`. With `batch_size=64`, chance = 0.016; healthy is >0.5 by epoch 2-3.
- `stage1/cold_start/K` should be close to your dataset's true speaker count (within ±20%). Wildly off → the dual-gate is filtering too aggressively or the corpus is more diverse than `distance_threshold=0.55` expects.

Output: `checkpoints/stage1/identity_pool_stage1.pt` (fuser + initial enrollments).

### E.4 — Stop criterion (spec §7)

The spec says "AV-SID accuracy plateau on a held-out set." The current script uses an `epochs` cap as a placeholder. To do this properly, plug in a dev-set evaluator that:
1. Holds out ~10 % of utterances.
2. Each epoch end: compute AV-SID Acc on the held-out set.
3. Stop when 3 consecutive epochs show no improvement >0.5 pp.

You can hack this in `scripts/train_identity.py` after the inner epoch loop without touching any other module.

---

## Phase F — Real Stage-2 training (multi-task, full unfreeze, GER LoRA)

**Payoff:** end-to-end fine-tuned model with Llama-3 LoRA in the loop. This is the most demanding phase memory-wise.

### F.1 — Memory tuning (24 GB GPU users read this first)

Stage-2 forward involves all 5 backbones simultaneously. Per-step memory at fp16:

```
Whisper encoder + decoder    ~3.5 GB
AV-HuBERT features           ~1.5 GB
ECAPA + ArcFace              ~0.5 GB
Llama-3-8B fp16              ~16 GB
LoRA adapters + Q-Former     ~0.3 GB
Activations + gradients      ~3-5 GB
                            ────────
Total                        ~25 GB
```

On a 24 GB card you cannot fit this in fp16. Two knobs to pull:

1. **4-bit Llama-3** drops the LLM to ~5 GB (LoRA still trains). Edit `avsd_ger/c2_alignment/ger_head.py` LLM loader to add `load_in_4bit=True` and `bnb_4bit_compute_dtype=torch.bfloat16`.
2. **Gradient checkpointing** on the LLM — `self._llm.gradient_checkpointing_enable()` after instantiation. Saves ~30 % activation memory at the cost of ~25 % step time.

With both: 24 GB is workable.

### F.2 — Run training

```bash
python scripts/train_stage2.py \
    --config   configs/default.yaml \
    --manifest data/lrs3_train_manifest.jsonl \
    --out      checkpoints/stage2/ \
    --wandb-project  avsd-ger \
    --wandb-run-name stage2-lrs3-v1 \
    --wandb-tags     stage2 real lrs3 lora
```

The script enforces `stage2.lr == stage1.lr × 0.1` at startup (spec §7). If you change `stage1.lr` in the config, `stage2.lr` must move with it or the script raises `ValueError`.

### F.3 — What healthy training looks like

- `stage2/loss/ctc` starts at ~5–8 (high because targets are character-level), drops to ~1–2 over 2–3 epochs.
- `stage2/loss/ger` starts at ~3 (Llama-3 has no idea about your domain yet), drops to ~1–1.5 by epoch 5.
- `stage2/loss/info` should stay roughly flat (the InfoNCE was already trained in Stage-1; this is just regularisation).
- `stage2/lr` flat at `1e-4` after warmup.

If `loss/ger` doesn't move, your LoRA is either not being optimised or the `target_modules` in `cfg.ger.lora` don't match the Llama-3 attention layer names. Verify with:
```python
from peft import PeftModel
print([n for n, p in pipe.ger._llm.named_parameters() if p.requires_grad][:5])
```

Output: `checkpoints/stage2/{identity_pool_stage2.pt, aligner_stage2.pt, ctc_head_stage2.pt}` + LoRA adapter dir.

> **Important (this bit me in stub rehearsal):** `train_stage2.py` only updates the **fuser weights** in the saved pool. It does **not** call `pool.enroll()`. The saved pool has 0 speakers. You **must re-enroll** test speakers with the trained fuser before Phase G — see Phase G below.

---

## Phase G — Real ablation eval

**Payoff:** the headline numbers for the paper. SA-WER, SCR, AV-SID Acc, DER, JER × 5 ablation rows + per-row energy + the spec safety check.

### G.1 — Re-enrol your test speakers using the trained Stage-2 fuser

This is the step that makes Phase G meaningful (and not the wo_c1-collapse you saw in stub rehearsal).

```bash
# Build a separate JSON listing only your test speakers + their enrolment audio:
#   data/test_speakers.json   (same shape as data/sample_manifest.json's "speakers" block)
python scripts/enroll_identity.py \
    --in-pool  checkpoints/stage2/identity_pool_stage2.pt \
    --manifest data/test_speakers.json \
    --out-pool checkpoints/stage2/identity_pool_stage2_enrolled.pt
```

The `--in-pool` flag loads the trained Stage-2 fuser **before** running enrolments, so the new `z_id` vectors are computed with the trained fuser, not a fresh init.

Sanity check: when this runs you'll see
```
[in-pool] loaded checkpoints/stage2/identity_pool_stage2.pt -- start size = 0
[enroll] alice  pool size = 1
[enroll] bob    pool size = 2
[enroll] carol  pool size = 3
[save]  identity pool -> checkpoints/stage2/identity_pool_stage2_enrolled.pt
```

### G.2 — Run the 5-row ablation table

```bash
python scripts/eval_ablations.py \
    --config   configs/default.yaml \
    --manifest data/lrs3_test_session_manifest.json \
    --pool     checkpoints/stage2/identity_pool_stage2_enrolled.pt \
    --out      out/ablation_report_lrs3_test.json \
    --idle-calibrate-s 2.0 \
    --wandb-project  avsd-ger \
    --wandb-run-name ablation-lrs3-test-v1 \
    --wandb-tags     final eval ablation lrs3
```

Sanity check the first stdout line: `[pool] loaded ... -- N speakers` where N matches your test-speaker count. If `N == 0`, you're back in the wo_c1-collapse and Phase G.1 didn't take.

### G.3 — Inspecting results

The `--out` JSON has one record per row:
```json
{"results": [
  {"ablation": "full_model", "metrics": {...}, "power": {...}, "transcript": "..."},
  {"ablation": "wo_c1",      "metrics": {...}, ...},
  ...
]}
```

The W&B run summary has:
- `summary/<row>/<metric>` — sortable in the UI to compare experiments.
- `summary/spec_check_c3_gate_pass` — boolean. **Must be `true`** for a valid run; if `false`, the gate-removed variant didn't degrade and that's a structural-safety failure.

### G.4 — What "good" looks like for the paper

There's no fixed bar — these depend on the dataset — but the **shape** of the table should be:

| Row | SA-WER | AV-SID Acc | Notes |
|---|---|---|---|
| `full_model` | best | highest | the headline number |
| `wo_c1` | clearly worse | drops dramatically | C1's contribution; if AV-SID is still high, C1 isn't doing what we think |
| `wo_c2` | worse than full, better than wo_c1 | similar to full | C2 fixes text; doesn't directly affect ID |
| `wo_c3` | slightly worse than full | similar to full | C3 helps on ambiguous cases |
| `c3_wo_conf_gate` | **worst row** | **worst** | spec §10 structural safety: removing the gate should be worse than removing C3 entirely |

If `c3_wo_conf_gate` is **not** strictly worse than `wo_c3`, the safety property of the gate is suspect — investigate before publishing.

---

## Quick reference card

```
Phase 0  → enroll + run_sample on stub data        (verify wiring)
Phase A  → eval_ablations on stub data             (verify metrics + safety check)
Phase B  → hf auth login + Llama-3 access request  (async)
Phase C  → download AV-HuBERT (manual)             (auto: Whisper, ECAPA, ArcFace, Llama-3)
Phase D  → flip stub_backbones=false; smoke test   (one real clip)
Phase E  → train_identity.py on real manifest      (Stage-1: InfoNCE + CTC)
Phase F  → train_stage2.py on real manifest        (Stage-2: CTC + GER + InfoNCE)
Phase G  → re-enroll w/ Stage-2 fuser, eval_ablations on test set
```

Per-phase artifacts:

```
checkpoints/identity_pool.pt            ← Phase 0 output (initial fuser, dev speakers)
checkpoints/stage1/identity_pool_stage1.pt  ← Phase E output (trained fuser, dev speakers)
checkpoints/stage2/identity_pool_stage2.pt  ← Phase F output (trained fuser, NO speakers)
checkpoints/stage2/identity_pool_stage2_enrolled.pt  ← Phase G.1 output (trained fuser + test speakers)
checkpoints/stage2/aligner_stage2.pt        ← Phase F output
checkpoints/stage2/ctc_head_stage2.pt       ← Phase F output
out/ablation_report_lrs3_test.json          ← Phase G output
```

If you don't see all of these at the end, work backwards from whichever is missing to find which phase didn't finish.
