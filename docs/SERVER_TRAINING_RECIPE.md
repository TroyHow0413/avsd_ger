# Server Training Recipe

This is the long-form server/W&B recipe used for the AMI visual runs. The three training stages are:

1. identity / InfoNCE
2. `align_ctc` warmup
3. `joint` Stage-2 training

The final `eval_ablations.py` command is an evaluation step, not a training stage.

Replace the W&B placeholders before launching:

| Placeholder | Meaning | Example |
|---|---|---|
| `<project name>` | W&B project name | `avsd-ger-4090-stage2` |
| `<identity run name>` | Stage-1 W&B run name | `server-identity-infonce-ami-train-visual` |
| `<align ctc run name>` | align/CTC warmup W&B run name | `server-fw-align-ctc` |
| `<joint run name>` | joint Stage-2 W&B run name | `server-fw-joint` |
| `<eval run name>` | eval W&B run name | `eval-fw-joint-dev-full-model-audio-only` |
| `<tags...>` | Space-separated W&B tags | `server faster_whisper joint ami_train_visual` |

## Shared Settings

These commands assume the server repo root is:

```bash
/home/ai-faculty/avsd_ger
```

and the training manifest is:

```bash
data/ami_train_visual.jsonl
```

Run from the repo root:

```bash
cd /home/ai-faculty/avsd_ger
```

## Stage 1: Identity InfoNCE

Trains the C1 identity fuser and writes `identity_pool_stage1.pt`.

```bash
python /home/ai-faculty/avsd_ger/scripts/train_identity.py \
  --config configs/default.yaml \
  --manifest data/ami_train_visual.jsonl \
  --out checkpoints/server_identity \
  --epochs 2 \
  --lr 0.001 \
  --warmup-steps 100 \
  --wandb-project <project name> \
  --wandb-run-name <identity run name> \
  --wandb-tags <tags...>
```

Example tags:

```bash
--wandb-tags server identity infonce ami_train_visual
```

Output used by later stages:

```text
checkpoints/server_identity/identity_pool_stage1.pt
```

## Stage 2: align_ctc Warmup

Trains the ID-conditioned aligner and CTC head without loading the full GER objective.

```bash
python /home/ai-faculty/avsd_ger/scripts/train_stage2.py \
  --config configs/default.yaml \
  --manifest data/ami_train_visual.jsonl \
  --out checkpoints/server_fw_align_ctc \
  --warmup align_ctc \
  --stage1-pool checkpoints/server_identity/identity_pool_stage1.pt \
  --epochs 3 \
  --lr 0.0001 \
  --asr-backend faster-whisper \
  --ger-mode av \
  --debug-loss-every 100 \
  --grad-clip 1.0 \
  --wandb-project <project name> \
  --wandb-run-name <align ctc run name> \
  --wandb-tags <tags...>
```

Example tags:

```bash
--wandb-tags server faster_whisper align_ctc
```

Outputs used by the joint stage:

```text
checkpoints/server_fw_align_ctc/aligner_stage2.pt
checkpoints/server_fw_align_ctc/ctc_head_stage2.pt
```

## Stage 3: joint Stage-2 Training

Runs the full Stage-2 objective after seeding from the `align_ctc` warmup checkpoints.

```bash
python /home/ai-faculty/avsd_ger/scripts/train_stage2.py \
  --config configs/default.yaml \
  --manifest data/ami_train_visual.jsonl \
  --out checkpoints/server_fw_joint \
  --warmup joint \
  --stage1-pool checkpoints/server_identity/identity_pool_stage1.pt \
  --aligner-checkpoint checkpoints/server_fw_align_ctc/aligner_stage2.pt \
  --ctc-checkpoint checkpoints/server_fw_align_ctc/ctc_head_stage2.pt \
  --epochs 3 \
  --lr 0.0001 \
  --asr-backend faster-whisper \
  --ger-mode av \
  --llm-quant auto \
  --debug-loss-every 100 \
  --grad-clip 1.0 \
  --wandb-project <project name> \
  --wandb-run-name <joint run name> \
  --wandb-tags <tags...>
```

Example tags:

```bash
--wandb-tags server faster_whisper joint ami_train_visual
```

Outputs used by evaluation:

```text
checkpoints/server_fw_joint/aligner_stage2.pt
checkpoints/server_fw_joint/ctc_head_stage2.pt
checkpoints/server_fw_joint/ger/
```

## Final Eval: Dev full_model audio_only

This step evaluates the trained joint artifacts. It uses `--fresh-pool` so the eval runner loads the Stage-1 identity fuser and enrolls speakers from the dev manifests for the run.

```bash
python /home/ai-faculty/avsd_ger/scripts/eval_ablations.py \
  --config configs/default.yaml \
  --manifest data/ami_dev_visual \
  --pool checkpoints/server_identity/identity_pool_stage1.pt \
  --fresh-pool \
  --aligner-ckpt checkpoints/server_fw_joint/aligner_stage2.pt \
  --ger-ckpt checkpoints/server_fw_joint/ger \
  --only full_model \
  --ger-mode audio_only \
  --llm-quant auto \
  --out out/eval_fw_joint_dev_full_model_audio_only.json \
  --wandb-project <project name> \
  --wandb-run-name <eval run name> \
  --wandb-tags <tags...>
```

Example tags:

```bash
--wandb-tags eval faster_whisper joint dev full_model audio_only auto
```

## Quick Order Checklist

Run these in order:

1. Stage 1 identity InfoNCE
2. Stage 2 `align_ctc` warmup
3. Stage 3 `joint` training
4. Final eval

The eval command should point at `checkpoints/server_fw_joint/aligner_stage2.pt` and `checkpoints/server_fw_joint/ger`.
