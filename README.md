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
| GER head | `peft`, `sentencepiece`, `bitsandbytes` | Llama-3 LoRA and optional quantized loading. |
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

# Gated Llama-3-8B-Instruct access (only when stub_backbones=false).
# The `hf` CLI requires huggingface_hub>=0.34. If `hf` is missing, upgrade
# the package inside the active env first:
python -m pip install --upgrade "huggingface_hub>=0.34,<1.0"
hf auth login                         # paste a Read-scope token from https://huggingface.co/settings/tokens
hf auth whoami                        # sanity check
# On older environments that you cannot upgrade yet, use: huggingface-cli login
# then request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

### Windows 11 — known quirks after `pip install -r requirements.txt`

| Package | Issue | Fix |
|---|---|---|
| `bitsandbytes` | The PyPI wheel is Linux-only; the Windows build is at a different index. | `pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui` — or skip it entirely if you're not doing 4-/8-bit Llama loading. |
| DataLoader `num_workers` | Windows uses `spawn` (not `fork`) for multiprocessing — any script that sets `num_workers > 0` without a `__main__` guard will deadlock. | Wrap the entry point of any custom script in `if __name__ == '__main__':`, or pass `--num-workers 0` / set `num_workers=0` in the DataLoader calls. All scripts in `scripts/` already have the guard. |

---

### Backbone weights (only when running real models)

* **Whisper-large-v3** — auto-downloaded by `faster-whisper` and `transformers` on first use.
* **AV-HuBERT Large** — drop the `.pt` at `checkpoints/avhubert_large_lrs3_iter5.pt` (path in `configs/default.yaml`).
* **ECAPA-TDNN** — auto from SpeechBrain.
* **InsightFace `buffalo_l`** — auto on first use.
* **Llama-3-8B-Instruct** — pulled by the GER head once `hf auth login` is done. The `hf` CLI ships with `huggingface_hub>=0.34`; older envs may only have the legacy `huggingface-cli login`.

The repo defaults to `stub_backbones: true` in `configs/default.yaml`, so you can verify wiring without any of the above.

---

## Roadmap: from stub mode to real models

Follow the phases in order. Each phase says what you **get** from finishing it (the payoff), what it **needs** (prereqs), and exactly what to run. W&B flags are uniform across all training/eval scripts — see [W&B flags](#wb-flags) below.

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

### Phase C — Download backbone weights

**Payoff:** all 5 backbones can load real weights when you flip `stub_backbones: false`.

| Backbone | How to get the weights | Notes |
|---|---|---|
| Whisper-large-v3 | Auto-pulled by `faster-whisper` + `transformers` on first call | ~3 GB, cached under `~/.cache/huggingface/` |
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

> **Full operational walkthrough:** [`docs/PHASE_D_REAL_MODELS.md`](docs/PHASE_D_REAL_MODELS.md) — pre-flight checklist (Llama-3 access verification, AV-HuBERT path check, fairseq import, GPU sanity), GPU memory budget per GPU class, mouth-ROI preprocessing pointer, common failure modes, and the full **D → G** real-data rollout.

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

If the GER step OOMs (24 GB GPU + Llama-3 fp16 won't fit), enable 4-bit loading — see [`docs/PHASE_D_REAL_MODELS.md#gpu-memory-budget`](docs/PHASE_D_REAL_MODELS.md#gpu-memory-budget).

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

> Phase G uses a **session manifest** (turns + ref_text + ref_speaker), not the JSONL training manifest. Format: see [`docs/EVALUATION.md#session-manifest`](docs/EVALUATION.md#session-manifest).

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

**Real eval** (after Phase F — re-enroll speakers using the trained fuser, then point Phase G at that pool — full walkthrough in [`docs/PHASE_D_REAL_MODELS.md`](docs/PHASE_D_REAL_MODELS.md#phase-g--real-ablation-eval)):
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
| `scripts/enroll_identity.py` | Enrol speakers into the cross-modal identity pool. Supports `--in-pool` to load a trained-fuser pool before enrolling (used by Phase G). | [`docs/ARCHITECTURE.md#c1`](docs/ARCHITECTURE.md#c1--cross-modal-identity-pool) |
| `scripts/run_sample.py` | Run one utterance end-to-end (single-speaker path). | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| `scripts/train_identity.py` | Stage-1: identity-aware alignment with InfoNCE + CTC. | [`docs/TRAINING.md#stage-1`](docs/TRAINING.md#stage-1) |
| `scripts/train_stage2.py` | Stage-2 multi-task: CTC + GER cross-entropy + bidirectional InfoNCE. **Enforces `lr_stage2 == lr_stage1 * 0.1`** at runtime. | [`docs/TRAINING.md#stage-2`](docs/TRAINING.md#stage-2) |
| `scripts/eval_ablations.py` | Run the five spec ablation rows on a session manifest, write metrics + energy. | [`docs/EVALUATION.md#ablation-runner`](docs/EVALUATION.md#ablation-runner) |

---

## Layout

```
avsd_ger/
├── backbones/        # Whisper + AV-HuBERT wrappers (frozen)
├── c1_identity/      # ECAPA + ArcFace + IdentityPool + dual-gate + cold-start
├── c2_alignment/     # ID-conditioned aligner + GER head (Llama-3-8B + LoRA)
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
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Manifest format, the five primary metrics, power monitor, ablation runner. |
| [`docs/PHASE_D_REAL_MODELS.md`](docs/PHASE_D_REAL_MODELS.md) | **Real-model rollout (Phase D → G)** — pre-flight checks, GPU memory budget, mouth-ROI preprocessing, stage-1/2 training on real data, re-enrolment for eval, common failure modes. |

---

## Status

Skeleton + spec-aligned wiring complete. AST + cross-import verified. Stub-mode Phases 0/A/E/F/G all green; real-model rollout starts at Phase D — see [`docs/PHASE_D_REAL_MODELS.md`](docs/PHASE_D_REAL_MODELS.md).
