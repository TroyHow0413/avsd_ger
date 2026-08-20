# AVSD-GER

Identity-conditioned, closed-loop generative error correction for **multi-speaker** audio-visual speech recognition.

> Three contributions on top of the dual-hypothesis GER paradigm: a cross-modal **identity pool (C1)**, **ID-conditioned token-level alignment + speaker-aware GER (C2)**, and a **closed-loop confidence-gated feedback (C3)** that refines the pool online.
> See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) for the comparison with DualHyp / AVSD / DiarizationLM.
> Raw meeting-video diarization frontends are tracked separately in [`docs/AVSD_FRONTENDS.md`](docs/AVSD_FRONTENDS.md): oracle turns, common pyannote+ASD, strong Sortformer/Precision-2 style frontends, and degraded robustness profiles.

---

## Install (conda)

The project is tested with Python 3.10. Keep `pip==24.0` because
`fairseq==0.12.2` depends on older metadata that newer pip versions reject.

### Fast path: `conda env create`

```bash
git clone https://github.com/TroyHow0413/avsd_ger.git
cd avsd_ger

conda env create -f environment.yml
conda activate avsdger
```

Then force-install the correct PyTorch CUDA wheel for your GPU. This step is
intentional: several pip packages in the environment depend on torch, so a plain
`conda env create` may let pip choose a generic torch wheel before you pin the
right CUDA build.

```bash
# RTX 50-series / Blackwell, e.g. RTX 5080 or 5090, sm_120:
pip install --upgrade --force-reinstall \
    "torch>=2.7" "torchaudio>=2.7" "torchvision>=0.22" \
    --index-url https://download.pytorch.org/whl/cu128

# Older GPUs, e.g. A100/H100/RTX 30/40-series:
# pip install --upgrade --force-reinstall \
#     "torch==2.6.*" "torchaudio==2.6.*" "torchvision==0.21.*" \
#     --index-url https://download.pytorch.org/whl/cu124
```

Do not add `--no-deps` to the torch command. The torch wheel must pull its exact
CUDA runtime packages, including cublas, cudnn, cufft, cusparse, nccl, nvtx,
triton, and sympy.

### Clean deterministic path

This avoids the temporary "wrong torch first, correct torch second" behavior of
the fast path:

```bash
conda create -n avsdger python=3.10 pip=24.0 -y
conda activate avsdger

conda install -c conda-forge -y \
    "numpy>=1.24,<2.0" "scipy>=1.11" "pyyaml>=6.0" "tqdm>=4.66" \
    libsndfile ffmpeg openh264

# Choose ONE torch line:
pip install \
    "torch>=2.7" "torchaudio>=2.7" "torchvision>=0.22" \
    --index-url https://download.pytorch.org/whl/cu128

# pip install \
#     "torch==2.6.*" "torchaudio==2.6.*" "torchvision==0.21.*" \
#     --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Dependency notes:

| Dependency group | Packages | Why |
|---|---|---|
| Interpreter | `python=3.10`, `pip=24.0` | Python 3.11+ is fragile with AV-HuBERT/fairseq; pip 24.1+ rejects fairseq's old metadata. |
| Numeric / IO | `numpy>=1.24,<2.0`, `scipy`, `pyyaml`, `tqdm` | `numpy<2` keeps InsightFace/ONNX/OpenCV ABI compatibility. The pip block repeats the NumPy pin because transitive pip deps can otherwise upgrade it after conda solves the env. |
| System media libs | `libsndfile`, `ffmpeg`, `openh264` | Required by `soundfile`, `librosa`, and video/audio decode paths. |
| PyTorch | `torch`, `torchaudio`, `torchvision` from the PyTorch CUDA wheel index | RTX 50-series needs cu128 wheels with `sm_120`; older GPUs can use cu124. |
| ASR / text backbone | `faster-whisper`, `transformers`, `tokenizers`, `huggingface_hub>=0.34,<1.0`, `accelerate` | Whisper and transformer model loading. `huggingface_hub>=0.34` provides the `hf` CLI used below. |
| GER head | `transformers`, `peft`, `sentencepiece` | Local dense Qwen2.5/Llama-3.2 causal LM with LoRA. |
| Identity encoders | `speechbrain`, `insightface`, `onnxruntime` | ECAPA-TDNN voice embeddings and ArcFace face embeddings. |
| Audio / video Python IO | `librosa`, `opencv-python`, `opencv-python-headless`, `soundfile`, `python_speech_features` | Feature extraction and AV-HuBERT logfbank support. Both OpenCV wheels are pinned `<4.10` so `albumentations`/`insightface` cannot pull an OpenCV build that forces NumPy 2.x. |
| Monitoring / logging | `nvidia-ml-py`, `psutil`, `wandb` | Power logging and experiment tracking. |

### Verify the environment

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
python -c "import numpy, scipy, cv2, soundfile, librosa; print('core io ok')"
python -c "import transformers, faster_whisper, peft, speechbrain, insightface, onnxruntime; print('model deps ok')"
```

For RTX 5080/5090, `torch.cuda.get_arch_list()` must contain `sm_120`.

### Conda troubleshooting

On some Windows Anaconda installs, `conda env create` can fail before dependency
solving because the `conda-anaconda-tos` plugin cannot read its cache. Use:

```bash
conda --no-plugins env create -f environment.yml --solver classic
```

If this hangs during solving, use the clean deterministic path above.

> **Windows 11:** keep `openh264` installed. Without it, `ffmpeg` may fail at runtime with `libopenh264.so.5: cannot open shared object file`.

If an already-created server env prints an error like `A module that was compiled
using NumPy 1.x cannot be run in NumPy 2.2.6`, pip upgraded NumPy after conda
created the env. This usually comes from a transitive `opencv-python-headless`
install. Repair it in-place with:

```bash
conda activate avsdger
python -m pip uninstall -y numpy opencv-python opencv-python-headless
python -m pip install "numpy>=1.24,<2.0" "opencv-python>=4.9,<4.10" "opencv-python-headless>=4.9,<4.10"
python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

Expected: NumPy `1.26.x` and OpenCV `4.9.x`.

### Manual steps after `conda activate avsdger`

```bash
# fairseq from source — required by AV-HuBERT, PyPI fairseq is too stale
pip install "git+https://github.com/facebookresearch/fairseq.git@v0.12.2"

# AV-HuBERT repo on PYTHONPATH, plus a checkpoint-config compatibility patch.
git clone https://github.com/facebookresearch/av_hubert.git
source scripts/setup_avhubert_env.sh

# Equivalent one-off command if you only want to patch an existing server clone:
# python - <<'PY'
# from pathlib import Path
# p = Path("av_hubert/avhubert/hubert_pretraining.py")
# s = p.read_text()
# n = '    fine_tuning: bool = field(default=False, metadata={"help": "set to true if fine-tuning AV-Hubert"})\n'
# if "input_modality:" not in s:
#     p.write_text(s.replace(n, n + '    input_modality: Optional[str] = field(default="audiovisual", metadata={"help": "input modality: audio | video | audiovisual"})\n', 1))
# PY

# ⚠️  Windows 11 (native, not WSL): bash activate hooks don't run.
# Use one of these alternatives instead:
#
# Option A — conda env var (recommended, persists across activations):
#     conda env config vars set PYTHONPATH="D:\GitHub\avsd_ger_claude\av_hubert;D:\GitHub\avsd_ger_claude\av_hubert\avhubert"
#     conda deactivate && conda activate avsdger   # apply immediately
#
# Option B — batch activate hook (PowerShell users: create avhubert_path.bat):
#     New-Item -Force "$env:CONDA_PREFIX\etc\conda\activate.d\avhubert_path.bat"
#     Add-Content "$env:CONDA_PREFIX\etc\conda\activate.d\avhubert_path.bat" `
#         "@set PYTHONPATH=D:\GitHub\avsd_ger_claude\av_hubert;D:\GitHub\avsd_ger_claude\av_hubert\avhubert;%PYTHONPATH%"

# GER reuses a complete local Hugging Face model directory when present. If it
# is missing and ger.allow_download=true, it downloads ger.model_id there once:
#   checkpoints/Qwen2.5-3B-Instruct
#   checkpoints/Llama-3.2-3B-Instruct
```

### Windows 11 — known quirks after `pip install -r requirements.txt`

| Package | Issue | Fix |
|---|---|---|
| DataLoader `num_workers` | Windows uses `spawn` (not `fork`) for multiprocessing — any script that sets `num_workers > 0` without a `__main__` guard will deadlock. | Wrap the entry point of any custom script in `if __name__ == '__main__':`, or pass `--num-workers 0` / set `num_workers=0` in the DataLoader calls. All scripts in `scripts/` already have the guard. |

---

### Backbone weights (only when running real models)

* **Whisper-large-v3** — auto-downloaded into `checkpoints/whisper/` on first use (`faster-whisper` CT2 weights plus the Hugging Face encoder/rescore weights). This directory is portable; upload it with `checkpoints/` to avoid slow server downloads.
* **AV-HuBERT Large** — drop the `.pt` at `checkpoints/avhubert_large_lrs3_iter5.pt` (path in `configs/default.yaml`).
* **ECAPA-TDNN** — auto from SpeechBrain.
* **InsightFace `buffalo_l`** — auto on first use.
* **GER causal LM** — a dense HF model materialized in a configured local directory. Use `configs/qwen25_3b.yaml` for Qwen2.5-3B-Instruct or `configs/llama32_3b.yaml` for Llama-3.2-3B-Instruct. Training and evaluation default to `ger.allow_download: false` and fail fast when `ger.model_path` is incomplete. Model acquisition is a separate explicit preparation operation: temporarily opt in with `ger.allow_download: true`, materialize the model once, then restore the safe default before training or evaluation.

Set `stub_backbones: true` in an inherited smoke config to verify wiring without any of the above.

---

## Current Workflow

The old **Phase 0 / A / B / C / D / E / F / G** names were rollout labels from the first implementation pass. They are not runtime modes, and the training scripts do not read a `--phase` argument. The old notes were moved to [`docs/LEGACY_PHASE_ROLLOUT.md`](docs/LEGACY_PHASE_ROLLOUT.md).

For current day-to-day work, use the concrete scripts:

| Task | Script | Notes |
|---|---|---|
| Enroll speakers | `scripts/enroll_identity.py` | Builds or updates an identity pool. |
| Single utterance smoke test | `scripts/run_sample.py` | Runs C1 -> C2 -> C3 on one utterance. |
| Stage-1 training | `scripts/train_identity.py` | Trains the C1 identity fuser with bidirectional InfoNCE. |
| Stage-2 training | `scripts/train_stage2.py` | Trains alignment, CTC, GER LoRA/QFormer pieces depending on `--warmup`. |
| Ablation eval | `scripts/eval_ablations.py` | Runs the 5 ablation rows and writes metrics. |
| Convenience launcher | `one_go/train.py` | Optional wrapper around Stage-1 and Stage-2; useful for quick local runs, not required. |

### Minimal Stub Check

`configs/default.yaml` defaults to `stub_backbones: true`, so this checks wiring without downloading real model weights.

```bash
python scripts/enroll_identity.py --manifest data/sample_manifest.json
python scripts/run_sample.py --manifest data/sample_manifest.json --utt utt_0001
```

Optional stub ablation check:

```bash
python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/sample_session_manifest.json \
    --pool checkpoints/identity_pool.pt \
    --out out/ablation_report_stub.json \
    --no-power
```

### Real Training

Prepare the real backbones first: set `stub_backbones: false`, put AV-HuBERT at `checkpoints/avhubert_large_lrs3_iter5.pt`, place the selected GER model in its configured local directory, and let Whisper/ECAPA/InsightFace use their existing cache paths.

Stage-1:

```bash
python scripts/train_identity.py \
    --config configs/default.yaml \
    --manifest data/your_real_train_manifest.jsonl \
    --out checkpoints/stage1/ \
    --epochs 5 \
    --wandb-project avsd-ger \
    --wandb-run-name stage1-real-v1
```

Stage-2:

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

Useful Stage-2 `--warmup` modes:

| Mode | Trains | When to use |
|---|---|---|
| `joint` | aligner + CTC + identity fuser + GER projectors/LoRA | Full Stage-2 run. |
| `align_ctc` | aligner + CTC only | Debug alignment/CTC before loading the LLM path. |
| `ger_lora` | GER LoRA path only | Lower-memory GER text n-best training. Can pair with `--no-encoder-context`. |
| `ger_qformer` | GER LoRA + QFormer/id projector | Train GER soft-prefix pieces without the full joint objective. |

Dense bf16 GER example:

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

### Final Eval

`train_stage2.py` saves trained fuser weights, but it does not enroll test speakers into `identity_pool_stage2.pt`. Use one of these two valid eval patterns.

Recommended for per-meeting AMI eval: let `eval_ablations.py` load the trained fuser and fresh-enroll speakers from each session manifest:

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

Alternative for debugging or deployment-style fixed enrollment: pre-enroll once, then evaluate without `--fresh-pool`:

```bash
python scripts/enroll_identity.py \
    --manifest data/your_real_test_speakers.json \
    --in-pool checkpoints/stage2/identity_pool_stage2.pt \
    --out-pool checkpoints/stage2/identity_pool_stage2_enrolled.pt

python scripts/eval_ablations.py \
    --config configs/default.yaml \
    --manifest data/your_real_test_session_manifest.json \
    --pool checkpoints/stage2/identity_pool_stage2_enrolled.pt \
    --out out/ablation_report_real.json \
    --idle-calibrate-s 2.0 \
    --wandb-project avsd-ger \
    --wandb-run-name ablation-final-v1
```

When `eval_ablations.py` starts, check that the pool/enrollment log shows a non-zero speaker count.

### Optional One-Go Launcher

`one_go/train.py` is only a convenience wrapper. It writes a runtime config under `one_go/runs/` and then calls the same Stage-1/Stage-2 scripts above.

```bash
python one_go/train.py --stage all --manifest data/your_real_train_manifest.jsonl --real --device cuda
```

---

## W&B flags

All three scripts (`train_identity.py`, `train_stage2.py`, `eval_ablations.py`) share the same set of W&B CLI flags via `avsd_ger.wandb_logger.add_wandb_args`:

| Flag | What it sets | Default |
|---|---|---|
| `--wandb-project NAME` | `wandb.init(project=...)` | `avsd-ger` |
| `--wandb-run-name NAME` | `wandb.init(name=...)` | auto-derived from script + manifest stem |
| `--wandb-entity TEAM` | `wandb.init(entity=...)` | `$WANDB_ENTITY` env, else your default |
| `--wandb-tags T1 T2 ...` | `wandb.init(tags=[...])` | none |
| `--no-wandb` | Disables logging entirely | (logging on) |

If `wandb` is **not installed** or `--no-wandb` is set, the logger silently no-ops — the rest of the script runs unaffected. To enable live logging:

```bash
pip install wandb           # already in environment.yml / requirements.txt
wandb login                 # paste your W&B API key from https://wandb.ai/authorize
```

The metric namespaces written by each script:

| Script | W&B keys |
|---|---|
| `train_identity.py` | `stage1/loss/{total,A->V,V->A}`, `stage1/acc/{A->V,V->A}`, `stage1/lr`, `stage1/cold_start/{K,n_unknown}` |
| `train_stage2.py`   | `stage2/loss/{total,ctc,ger,info}`, `stage2/lr`, `stage2/epoch_end/{ctc,ger,info}` |
| `eval_ablations.py` | `ablation/<row>/{sa_wer,wer,scr,av_sid_acc,der,jer,energy_wh,avg_power_w}`, `summary/<row>/<metric>`, `summary/spec_check_c3_gate_pass` |

---

## Scripts

| Script | What it does | Detailed docs |
|---|---|---|
| `scripts/enroll_identity.py` | Enrol speakers into the cross-modal identity pool. Supports `--in-pool` to load a trained-fuser pool before evaluation enrollment. | [`docs/ARCHITECTURE.md#c1`](docs/ARCHITECTURE.md#c1--cross-modal-identity-pool) |
| `scripts/run_sample.py` | Run one utterance end-to-end (single-speaker path). | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| `scripts/train_identity.py` | Stage-1: identity fuser training with bidirectional InfoNCE. | [`docs/TRAINING.md`](docs/TRAINING.md) |
| `scripts/train_stage2.py` | Stage-2 multi-task training with selectable `--warmup` modes. **Enforces `lr_stage2 == lr_stage1 * ratio`** at runtime. | [`docs/TRAINING.md`](docs/TRAINING.md) |
| `scripts/eval_ablations.py` | Run the five spec ablation rows on a session manifest, write metrics + energy. | [`docs/EVALUATION.md#ablation-runner`](docs/EVALUATION.md#ablation-runner) |
| `one_go/train.py` | Optional convenience wrapper that calls Stage-1 and/or Stage-2 training. Not required for normal training. | See [Current Workflow](#current-workflow). |

---

## Layout

```
avsd_ger/
├── backbones/        # Whisper + AV-HuBERT wrappers (frozen)
├── c1_identity/      # ECAPA + ArcFace + IdentityPool + dual-gate + cold-start
├── c2_alignment/     # aligner + local HF backend/prompt/bridge/generation/checkpoint policies
├── c3_feedback/      # Composite confidence + closed-loop controller
├── training/         # CTC head, GER cross-entropy, identity (InfoNCE) loss
├── eval/             # SessionRunner, metrics (SA-WER/SCR/AV-SID/DER/JER), PowerMonitor
├── wandb_logger.py   # uniform W&B wrapper (no-op if wandb missing)
└── pipeline.py       # C1 -> C2 -> C3 orchestrator
configs/default.yaml  # all hyperparameters; ablation flags live under `ablation:`
data/                 # sample manifests for stub rehearsal
docs/                 # design + rollout docs (see table below)
scripts/              # CLI entry points
```

---

## Documentation index

| File | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | C1/C2/C3 module-level design, data shapes, key implementation choices. |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Side-by-side comparison vs. DualHyp, AVSD, DiarizationLM. |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Stage-1 / Stage-2 recipes, loss weights, the spec §7 LR invariant. |
| [`docs/SERVER_TRAINING_RECIPE.md`](docs/SERVER_TRAINING_RECIPE.md) | Server AMI visual recipe: identity -> align_ctc -> joint training plus W&B/eval commands. |
| [`docs/REAL_MODEL_WORKFLOW.md`](docs/REAL_MODEL_WORKFLOW.md) | Current real-model setup, manifest expectations, Stage-1/Stage-2 commands, re-enrollment, and eval workflow. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Manifest format, the five primary metrics, power monitor, ablation runner. |
| [`docs/LEGACY_PHASE_ROLLOUT.md`](docs/LEGACY_PHASE_ROLLOUT.md) | Archived Phase 0/A-G rollout notes from the old README. Not the current training workflow. |

---

## Status

Skeleton + spec-aligned wiring complete. AST + cross-import verified. Stub-mode enrollment, sample run, Stage-1, Stage-2, and ablation eval have working script paths. Current training uses `scripts/train_identity.py`, `scripts/train_stage2.py`, or the optional `one_go/train.py` wrapper.

