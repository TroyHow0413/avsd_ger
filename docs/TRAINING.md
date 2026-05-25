# Training

This project currently trains through concrete scripts, not Phase A-G labels.

| Job | Entry point | Output |
|---|---|---|
| Stage-1 identity training | `scripts/train_identity.py` | `identity_pool_stage1.pt` |
| Stage-2 multi-task training | `scripts/train_stage2.py` | `identity_pool_stage2.pt`, `aligner_stage2.pt`, `ctc_head_stage2.pt`, `ger/` |
| Optional wrapper | `one_go/train.py` | Calls Stage-1 and/or Stage-2 with a generated config under `one_go/runs/` |

`configs/default.yaml` defaults to `stub_backbones: true`, so both trainers can run a wiring rehearsal without downloading the real backbones. For real training, set `stub_backbones: false`, use `device: cuda`, and make sure the backbone weights in the README are available.

---

## Stage-1: `train_identity.py`

Stage-1 trains the C1 identity fuser with bidirectional InfoNCE over voice and face embeddings. Backbones are constructed through the wrappers and stay frozen.

Basic run:

```bash
python scripts/train_identity.py \
    --config configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --out checkpoints/stage1/ \
    --epochs 5 \
    --wandb-project avsd-ger \
    --wandb-run-name stage1-real-v1
```

AMI visual manifest directory run:

```bash
python scripts/train_identity.py \
    --config configs/default.yaml \
    --manifest-dir data/ami_train_visual \
    --out checkpoints/stage1/
```

Important arguments:

| Argument | Meaning |
|---|---|
| `--config` | YAML config path. Defaults to `configs/default.yaml`. |
| `--manifest` | JSONL training manifest, one utterance per line. |
| `--manifest-dir` | Directory of AMI visual per-meeting manifests. The script resolves the sibling `<dir>.jsonl`. |
| `--out` | Output directory. Defaults to `checkpoints/stage1/`. |
| `--epochs` | Overrides `training.stage1.epochs`. |
| `--lr` | Overrides `training.stage1.lr`. |
| `--warmup-steps` | Overrides `training.stage1.warmup_steps`. |

Expected JSONL fields:

| Field | Required | Notes |
|---|---|---|
| `wav_path` | Real mode: yes | Audio path. Relative paths are resolved from the repo root. |
| `face_path` | Real mode: yes | Face crop or enrollment image path. |
| `lip_conf` | Recommended | Per-frame lip confidence for the dual gate. |
| `speaker_id` / `ref_speaker` | Optional | Used for bookkeeping; cold-start pseudo-labels still drive Stage-1 training. |

If `--manifest` points to a missing file, the script falls back to 8 synthetic records. That is useful for stub checks, but it is not real training.

Output:

```text
checkpoints/stage1/identity_pool_stage1.pt
```

---

## Stage-2: `train_stage2.py`

Stage-2 composes CTC, GER cross-entropy, and InfoNCE depending on `--warmup`.

Basic full run:

```bash
python scripts/train_stage2.py \
    --config configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --stage1-pool checkpoints/stage1/identity_pool_stage1.pt \
    --out checkpoints/stage2/ \
    --warmup joint \
    --wandb-project avsd-ger \
    --wandb-run-name stage2-real-v1
```

The script enforces the learning-rate invariant at startup:

```text
training.stage2.lr == training.stage1.lr * training.stage2.lr_ratio_to_stage1
```

Defaults satisfy this with `1e-4 == 1e-3 * 0.1`. If you pass `--lr`, the script updates the ratio in memory for that run.

### Warmup Modes

| Mode | Loaded modules | Trainable pieces | Loss weights |
|---|---|---|---|
| `joint` | ASR, VSR, identity, aligner, CTC, GER | fuser, aligner, CTC, GER LoRA/projectors | `ctc=1`, `ger=1`, `info=0.5` |
| `align_ctc` | ASR, VSR, identity, aligner, CTC | aligner, CTC | `ctc=1`, `ger=0`, `info=0` |
| `ger_lora` | ASR, GER; with `--no-encoder-context`, also identity and optionally VSR for text/lip n-best context | GER LoRA only; QFormer/id projector frozen | `ctc=0`, `ger=1`, `info=0` |
| `ger_qformer` | ASR, VSR, identity, GER | GER LoRA, QFormer, id projector | `ctc=0`, `ger=1`, `info=0` |

`--no-encoder-context` disables Whisper/AV-HuBERT encoder features for GER and is valid only with `--warmup ger_lora`.

Low-memory GER LoRA run:

```bash
python scripts/train_stage2.py \
    --config configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --stage1-pool checkpoints/stage1/identity_pool_stage1.pt \
    --out checkpoints/stage2_ger_lora/ \
    --warmup ger_lora \
    --no-encoder-context \
    --llm-quant 4bit
```

Useful initialization arguments:

| Argument | Meaning |
|---|---|
| `--stage1-pool` | Loads a Stage-1 identity pool before Stage-2. This initializes the fuser. |
| `--aligner-checkpoint` | Optional aligner `state_dict` to initialize a warmup or joint run. |
| `--ctc-checkpoint` | Optional CTC head `state_dict` to initialize a warmup or joint run. |
| `--ger-projectors-checkpoint` | Optional `ger_projectors.pt` containing `qformer` and `id_proj` state dicts. |

Other practical overrides:

| Argument | Meaning |
|---|---|
| `--manifest-dir` | Same AMI visual directory convenience as Stage-1. |
| `--epochs` | Overrides `training.stage2.epochs`. |
| `--lr` | Overrides `training.stage2.lr` and adjusts the in-memory ratio guard. |
| `--ger-mode` | Overrides `cfg.ger.mode`: `audio_only`, `av`, or `visual_only`. |
| `--asr-backend` | Overrides ASR backend: `faster-whisper` or `openai-whisper`. |
| `--asr-beam-size` | Overrides ASR beam size. |
| `--asr-n-best` | Overrides ASR n-best count. |
| `--llm-quant` | Overrides Llama precision: `auto`, `fp16`, `bf16`, `int8`, or `4bit`. |
| `--debug-loss-every` | Prints detailed loss/debug info every N steps. |
| `--no-fail-on-nonfinite` | Logs non-finite losses instead of failing immediately. |
| `--grad-clip` | Gradient clipping norm. Defaults to `1.0`. |

Expected JSONL fields for real mode:

| Field | Required | Notes |
|---|---|---|
| `wav_path` / `audio` | Yes | Audio path. |
| `video_path` / `mouth_roi` / `video` | Required unless using a text-only GER path | `.npy` mouth ROI for AV-HuBERT is the common path. |
| `face_path` / `enrollment_face` | Required when identity modules are loaded | Face crop or enrollment image. |
| `target` / `ref_text` | Yes | Text target for CTC/GER. |
| `neg_wav_path`, `neg_face_path` | Optional | Used for negative identity pairs when present. |

If the manifest path is missing, Stage-2 falls back to 8 synthetic records. That is only a stub rehearsal.

Outputs:

```text
checkpoints/stage2/identity_pool_stage2.pt
checkpoints/stage2/aligner_stage2.pt
checkpoints/stage2/ctc_head_stage2.pt
checkpoints/stage2/ger/ger_projectors.pt
checkpoints/stage2/ger/lora_adapter/   # when a PEFT LoRA adapter exists
checkpoints/stage2/ger/tokenizer/       # when a tokenizer is available
```

Important: `identity_pool_stage2.pt` stores the trained fuser state, but it does not enroll evaluation speakers. Before final evaluation, load it with `scripts/enroll_identity.py --in-pool` and save a separate enrolled pool.

---

## Optional Wrapper: `one_go/train.py`

`one_go/train.py` writes a runtime config and delegates to the same scripts above.

```bash
python one_go/train.py \
    --stage all \
    --manifest data/your_real_train_manifest.jsonl \
    --real \
    --device cuda \
    --stage2-warmup joint
```

Use it when you want a quick single command. Use the direct scripts when debugging, changing warmup/checkpoint inputs, or running on a cluster scheduler.
