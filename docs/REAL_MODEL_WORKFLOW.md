# Real Model Workflow

This is the current real-data workflow for switching off `stub_backbones` and training/evaluating the system. It replaces the old Phase D-G wording with the scripts and arguments that exist now.

---

## Pre-Flight

Before setting `stub_backbones: false`, verify the model dependencies:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import soundfile, cv2, transformers, faster_whisper, peft, speechbrain, insightface; print('deps ok')"
python -c "from pathlib import Path; print(Path('checkpoints/avhubert_large_lrs3_iter5.pt').exists())"
hf auth whoami
```

Backbone expectations:

| Backbone | Setup |
|---|---|
| Whisper-large-v3 | Auto-cached on first ASR use. |
| AV-HuBERT Large | `checkpoints/avhubert_large_lrs3_iter5.pt`. |
| ECAPA-TDNN | Auto-cached from SpeechBrain. |
| InsightFace `buffalo_l` | Auto-cached on first face embedding call. |
| GER causal LM | Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct is reused from `ger.model_path`; when missing, configs may allow a one-time Hugging Face download from `ger.model_id`. |

For Windows native shells, make sure AV-HuBERT is on `PYTHONPATH` as described in the README install section.

---

## Data Manifests

Stage-1 and Stage-2 use JSONL training manifests. One line equals one utterance.

Minimal Stage-1 record:

```json
{"utt_id":"utt_0001","wav_path":"data/utts/utt_0001.wav","face_path":"data/faces/spk01.jpg","lip_conf":[0.93,0.91,0.95],"speaker_id":"spk01"}
```

Minimal Stage-2 record:

```json
{"utt_id":"utt_0001","wav_path":"data/utts/utt_0001.wav","mouth_roi":"data/mouth/utt_0001.npy","face_path":"data/faces/spk01.jpg","target":"hello this is a test","speaker_id":"spk01"}
```

`mouth_roi` is usually an AV-HuBERT mouth crop tensor saved as `.npy`, shaped like `[T, 1, 96, 96]`. For AMI visual data, the training scripts also support `--manifest-dir`, which resolves a converted sibling `<dir>.jsonl`.

Evaluation uses a session manifest, not the training JSONL. See [`EVALUATION.md`](EVALUATION.md).

---

## Real Smoke Test

After setting `stub_backbones: false`, run one known utterance before starting training:

```bash
python scripts/enroll_identity.py --manifest data/sample_manifest.json
python scripts/run_sample.py --manifest data/sample_manifest.json --utt utt_0001
```

Check that the transcript is plausible, the speaker ID is enrolled, and the run does not fall back to stub-looking fixed values.

---

## Stage-1 Training

```bash
python scripts/train_identity.py \
    --config configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --out checkpoints/stage1/ \
    --epochs 5 \
    --wandb-project avsd-ger \
    --wandb-run-name stage1-real-v1
```

Output:

```text
checkpoints/stage1/identity_pool_stage1.pt
```

Watch:

| W&B key | Healthy sign |
|---|---|
| `stage1/loss/total` | Trends down after the first batches. |
| `stage1/acc/A->V`, `stage1/acc/V->A` | Rises above chance. |
| `stage1/cold_start/K` | Roughly plausible for the corpus speaker count. |

---

## Stage-2 Training

Full joint run:

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

Lower-memory GER-only run:

```bash
python scripts/train_stage2.py \
    --config configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --stage1-pool checkpoints/stage1/identity_pool_stage1.pt \
    --out checkpoints/stage2_ger_lora/ \
    --warmup ger_lora \
    --no-encoder-context \
    --ger-dtype bf16
```

Current warmup modes:

| Mode | Use |
|---|---|
| `joint` | Full CTC + GER + InfoNCE run. |
| `align_ctc` | Alignment and CTC warmup without GER loss. |
| `ger_lora` | Text n-best GER LoRA training; can use `--no-encoder-context`. |
| `ger_qformer` | GER LoRA plus QFormer/id projector training. |

Outputs:

```text
checkpoints/stage2/identity_pool_stage2.pt
checkpoints/stage2/aligner_stage2.pt
checkpoints/stage2/ctc_head_stage2.pt
checkpoints/stage2/ger/
```

`identity_pool_stage2.pt` is not an enrolled evaluation pool. It stores trained fuser state only.

---

## Re-Enroll And Evaluate

For per-meeting AMI evaluation, prefer `--fresh-pool`: `eval_ablations.py` loads the trained fuser from `--pool`, then enrolls the `speakers` block from the current manifest before each ablation row.

```bash
python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/your_real_test_session_manifest.json \
    --pool checkpoints/stage2/identity_pool_stage2.pt \
    --fresh-pool \
    --out out/ablation_report_real.json \
    --idle-calibrate-s 2.0 \
    --wandb-project avsd-ger \
    --wandb-run-name ablation-final-v1
```

For debugging or deployment-style fixed enrollment, you can pre-enroll evaluation speakers with the trained Stage-2 fuser:

```bash
python scripts/enroll_identity.py \
    --manifest data/your_real_test_speakers.json \
    --in-pool checkpoints/stage2/identity_pool_stage2.pt \
    --out-pool checkpoints/stage2/identity_pool_stage2_enrolled.pt
```

Then run ablations without `--fresh-pool`:

```bash
python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/your_real_test_session_manifest.json \
    --pool checkpoints/stage2/identity_pool_stage2_enrolled.pt \
    --out out/ablation_report_real.json \
    --idle-calibrate-s 2.0 \
    --wandb-project avsd-ger \
    --wandb-run-name ablation-final-v1
```

The first eval log should show a non-zero speaker count or fresh enrollment messages. If it shows `0`, the run is using the unenrolled Stage-2 pool and speaker metrics will collapse.

---

## GPU Notes

| GPU class | Practical setting |
|---|---|
| 24 GB cards | Prefer the Qwen2.5-3B config with `--warmup ger_lora --no-encoder-context --ger-dtype bf16` before attempting joint runs. |
| A100 40 GB | Joint dense fp16/bf16 runs are more realistic; still keep batch sizes conservative. |
| A100 80 GB / H100 | Joint runs have more room for full AV context and larger accumulation. |

Use `--ger-dtype auto`, `bf16`, or `fp16`. Quantized GER loading is intentionally outside the phase-1 backend scope.
