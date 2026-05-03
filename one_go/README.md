# one_go 可行性入口

这个文件夹是对现有框架的薄封装，不重写核心逻辑。核心代码仍然走：

- `avsd_ger/pipeline.py`
- `scripts/enroll_identity.py`
- `scripts/run_sample.py`
- `scripts/eval_ablations.py`
- `scripts/train_identity.py`
- `scripts/train_stage2.py`

## 逻辑

整体链路是：

```text
C1 身份池注册/查询
  -> C2 Whisper + AV-HuBERT 特征对齐 + Llama GER
  -> C3 置信度判断，决定接受、重对齐、重识别、是否更新身份池
```

`main.py` 用来测试推理可行性：

1. 生成临时配置到 `one_go/runs/config_stub.yaml` 或 `one_go/runs/config_real.yaml`
2. 调 `one_go/c1_enroll.py` 注册 speaker，输出 `one_go/runs/identity_pool.pt`
3. 调 `scripts/run_sample.py` 跑一个 utterance
4. 可选调 `scripts/eval_ablations.py` 跑 5 行消融评估

默认是 stub mode：不会下载大模型，不要求真实音频/视频文件存在，适合先测代码框架能不能跑通。

## 先跑最小可行性测试

在项目根目录运行：

```bash
python one_go/main.py --mode smoke
```

默认会生成 `ger.mode: audio_only`，所以如果 manifest 里的 `mouth_roi` 是 `null`，不会再用随机 video fallback，也不会把 `Visual hypothesis` / `<AV_CTX>` 交给 GER。

如果以后已经有真实 mouth ROI，可以显式打开 AV-GER：

```bash
python one_go/main.py --mode smoke --ger-mode av
```

如果当前 shell 的 Python 不是项目环境，可以指定解释器：

```bash
python one_go/main.py --mode smoke --python C:\ProgramData\anaconda3\envs\avsdger\python.exe
```

这会做：

- enroll identity pool
- run `utt_0001`
- 输出文本、speaker id、confidence、trace

跑完整 stub 可行性，包括消融：

```bash
python one_go/main.py --mode all
```

输出主要文件：

- `one_go/runs/identity_pool.pt`
- `one_go/runs/ablation_report.json`
- `one_go/runs/config_stub.yaml`

AVSD frontend profile 只记录“turn manifest 是由哪种前端条件产生的”，不会启动 pyannote/Sortformer/ASD 本身。可选值：

- `oracle_turns`
- `common_pyannote_lightasd`
- `strong_sortformer_talknet`
- `degraded_pyannote`

查看这些 profile：

```bash
python scripts/frontend_profiles.py --format markdown
```

用 one_go 跑一个目录里的所有 session manifest：

```bash
python one_go/main.py ^
  --mode eval ^
  --session-manifest data/ami_test/manifests ^
  --pool checkpoints/identity_pool.pt ^
  --ablation-out out/ami_ablation ^
  --frontend-profile common_pyannote_lightasd
```

开启 W&B：

```bash
python one_go/main.py ^
  --mode eval ^
  --session-manifest data/ami_test/manifests ^
  --pool checkpoints/identity_pool.pt ^
  --ablation-out out/ami_ablation ^
  --frontend-profile common_pyannote_lightasd ^
  --wandb-project avsd-ger ^
  --wandb-run-name ami-common-frontend
```

说明：原始 `scripts/enroll_identity.py` 会构建完整 Pipeline，所以注册阶段也会加载 AV-HuBERT 和 Llama。`one_go/c1_enroll.py` 只加载 C1 的 ECAPA、InsightFace 和 IdentityPool，更适合作为真实模型 smoke test 的第一步。

## 真实模型 smoke test

只有在以下条件都满足后再跑：

- Hugging Face 已登录，并且有 `meta-llama/Meta-Llama-3-8B-Instruct` 权限
- AV-HuBERT checkpoint 路径和 `configs/default.yaml` 一致
- GPU 显存足够，或使用 4bit/int8
- manifest 里的 audio、mouth ROI、face/enrollment 文件真实存在

命令：

```bash
python one_go/main.py --mode smoke --real --device cuda --llm-quant 4bit
```

## AMI ES2004a 真实音频 smoke test

准备一个从 `datasets/ami` 切出来的小 manifest：

```bash
python one_go/prepare_ami_smoke.py --meeting ES2004a --speaker B
```

然后运行真实模型 smoke：

```bash
python one_go/main.py \
  --mode smoke \
  --real \
  --device cuda \
  --llm-quant 4bit \
  --ger-mode audio_only \
  --manifest one_go/runs/ami_es2004a/ES2004a_B_manifest.json \
  --utt ES2004a_B_utt \
  --pool one_go/runs/ami_es2004a/identity_pool.pt
```

这个 manifest 使用真实 AMI headset wav，但 `mouth_roi` 仍为 `null`，所以默认会走 audio-only GER。它适合验证真实音频下 Whisper + C1 + GER + C3 的可行性；要验证真正的 `lip_hyp`，还需要从 AMI video 预处理出 mouth ROI `.npy`，然后用 `--ger-mode av`。

使用自己的 manifest：

```bash
python one_go/main.py ^
  --mode smoke ^
  --real ^
  --device cuda ^
  --llm-quant 4bit ^
  --manifest data/sample_manifest.json ^
  --utt utt_0001
```

## 训练入口

Stage 1：训练 C1 fuser / C2 alignment 相关的身份对齐流程。

先用 stub 跑通训练控制流：

```bash
python one_go/train.py --stage stage1
```

如果需要指定 conda 环境里的 Python：

```bash
python one_go/train.py --stage stage1 --python C:\ProgramData\anaconda3\envs\avsdger\python.exe
```

真实 Stage 1：

```bash
python one_go/train.py ^
  --stage stage1 ^
  --real ^
  --device cuda ^
  --manifest data/lrs3_pretrain_manifest.jsonl ^
  --stage1-out checkpoints/stage1
```

Stage 2：多任务训练，包含 CTC、GER CE、InfoNCE，并训练 Llama LoRA。

stub 控制流：

```bash
python one_go/train.py --stage stage2
```

真实 Stage 2：

```bash
python one_go/train.py ^
  --stage stage2 ^
  --real ^
  --device cuda ^
  --llm-quant 4bit ^
  --manifest data/lrs3_train_manifest.jsonl ^
  --stage2-out checkpoints/stage2
```

连续跑两阶段：

```bash
python one_go/train.py --stage all --real --device cuda --llm-quant 4bit --manifest data/lrs3_train_manifest.jsonl
```

## manifest 要点

单样本 smoke manifest 走 `data/sample_manifest.json` 这种结构：

```json
{
  "speakers": [
    {
      "speaker_id": "spk_01",
      "enrollment_audio": "data/spk_01/enroll.wav",
      "enrollment_face": "data/spk_01/enroll.jpg"
    }
  ],
  "utterances": [
    {
      "utt_id": "utt_0001",
      "audio": "data/utts/utt_0001.wav",
      "mouth_roi": "data/utts/utt_0001_mouth.npy",
      "transcript_gold": "the quick brown fox jumps over the lazy dog"
    }
  ]
}
```

训练 manifest 是 JSONL，一行一个 utterance。Stage 1 当前脚本主要读取：

```json
{"utt_id": "x001", "wav_path": "path/to.wav", "face_path": "path/to.jpg", "lip_conf": [0.9, 0.8]}
```

注意：`scripts/train_stage2.py` 里的真实数据 loader 目前还是项目定制 TODO，stub mode 可以跑控制流，真实 Stage 2 需要你先补 `_load_record()` 或接入自己的 LRS3/AMI 数据适配器。

## 常用排错

- 只想验证代码框架：不要加 `--real`
- Llama 显存不够：加 `--llm-quant 4bit`
- 真实 AV-HuBERT 文本为空：可能用的是 pretraining checkpoint，没有 decoder/dict
- 报 `/checkpoint/bshi/.../dict.km.txt` 缺失：这是 AV-HuBERT checkpoint 内部保存的作者机器绝对路径。当前代码会在缺失时用 dummy HuBERT dictionary 重建 encoder；`lip_hyp` 可能为空，但 `<AV_CTX>` 特征路径仍可用
- 报 `encoder_attn.k_proj.weight` 1024/512 mismatch：这是旧 AV-HuBERT seq2seq 配置没有显式写 `encoder_embed_dim`，Fairseq 默认成 512。当前已在 `av_hubert/avhubert/hubert_asr.py` 构建 decoder 前把它补成 large checkpoint 需要的 1024
- pool 为空：先跑 `python one_go/main.py --mode smoke` 生成 `one_go/runs/identity_pool.pt`
- 想看将执行什么命令：加 `--dry-run`

## 让 lip_hyp 不为空

`lip_hyp` 是 AV-HuBERT VSR decoder 解码出来的视觉文本。它不为空需要同时满足：

1. 使用 fine-tuned VSR checkpoint，例如 `checkpoints/self_large_vox_433h.pt`，而不是 pretraining-only 的 `avhubert_large_lrs3_iter5.pt`
2. `configs/default.yaml` 里保持 `vsr.emit_text: true`
3. checkpoint 能正确加载 seq2seq decoder 和 target dictionary
4. `mouth_roi` 指向真实的 `[T, 1, 96, 96]` 或 `[T, 96, 96]` mouth ROI `.npy`；缺失时会降级为 audio-only，不会再触发随机 video fallback
5. mouth ROI 必须和 audio 同一段、同一 speaker、约 25 fps

当前 sample manifest 里的 `data/utts/utt_0001_mouth.npy` 如果不存在，代码会把该 turn 标记为 `has_visual=false` 并降级到 audio-only GER；如果要测试 `<AV_CTX>` 和 `lip_hyp`，请提供真实 ROI 并使用 `--ger-mode av`。
