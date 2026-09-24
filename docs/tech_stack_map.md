# AVSD-GER 技术栈地图

> Audio-Visual Speaker Diarization + Generative Error Correction  
> **文档状态：** 当前技术概览；运行参数以 `configs/default.yaml` 及继承的模型配置为准。详见[文档索引](../README.md#documentation-index)。
> 对齐版本：2026-09-24 | 代码库：`avsd_ger/`

---

## 目录

1. [系统总体架构](#1-系统总体架构)
2. [骨干模型 Backbone](#2-骨干模型-backbone)
3. [C1 身份识别](#3-c1-身份识别)
4. [C2 对齐与 GER](#4-c2-对齐与-ger)
5. [C3 置信度与闭环反馈](#5-c3-置信度与闭环反馈)
6. [前端预处理 Frontend](#6-前端预处理-frontend)
7. [训练目标](#7-训练目标)
8. [评估指标](#8-评估指标)
9. [依赖与环境](#9-依赖与环境)

---

## 1. 系统总体架构

```
原始音视频
    │
    ▼
┌─────────────────────────────────┐
│  Frontend (离线预处理)           │
│  pyannote → Light-ASD → dlib    │
│  → Manifest JSON + .npy 文件    │
└────────────────┬────────────────┘
                 │
    ┌────────────▼────────────┐
    │       C1 Identity       │  ← ECAPA-TDNN + ArcFace
    │   说话人身份识别 / 池化   │     IdentityFuser MLP
    └────────────┬────────────┘
                 │ z_id [256]
    ┌────────────▼────────────┐
    │       C2 Alignment      │  ← Whisper + AV-HuBERT
    │   IDConditionedAligner  │     QFormer + 已注册 Causal LM LoRA
    │        GER Head         │
    └────────────┬────────────┘
                 │ corrected transcript
    ┌────────────▼────────────┐
    │      C3 Feedback        │  ← ConfidenceScorer
    │   置信度评估 + 闭环控制  │     ClosedLoopController
    └─────────────────────────┘
```

**数据流总览：**

| 阶段 | 输入 | 输出 | 关键文件 |
|------|------|------|---------|
| Frontend | 原始视频 + 音频 | Manifest JSON, `.npy` ROI | `avsd_ger/frontend/` |
| C1 | 音频帧, 人脸帧 | `z_id [256]`, identity pool | `avsd_ger/c1_identity/` |
| C2 | Whisper hidden, AV-HuBERT feat, z_id | `f_align`, GER transcript | `avsd_ger/c2_alignment/` |
| C3 | transcript, confidence signals | 最终 transcript, 动作决策 | `avsd_ger/c3_feedback/` |

---

## 2. 骨干模型 Backbone

### 2.1 ASR — Whisper-large-v3

| 属性 | 值 |
|------|----|
| 模型 | `openai/whisper-large-v3` |
| 编码器输出 | `[T_a, 1280]` hidden states |
| 解码后端 | faster-whisper (CTranslate2, INT8) |
| 特征提取后端 | HuggingFace transformers |
| 束搜索宽度 | beam_size=10，n_best=5（配置层，faster-whisper 实际仅返回 1-best） |
| 重打分 | Teacher-forced mean token log-prob |

**技术方向：**
- `faster-whisper decode` → **模型量化推理 / CUDA 内核融合** (INT8 Quantization + CTranslate2 kernel fusion)
- `HF encoder hidden states` → **大规模弱监督语音预训练** (Weak-supervised Seq2Seq ASR Pre-training)
- `rescore()` → **N-best 声学模型重打分** (Acoustic Rescoring, Log-likelihood Scoring)
- `pool_encoder_to_tokens()` → **声学帧到词级时序聚合** (Frame-to-Token Temporal Aggregation, Mean Pooling within word timestamp windows)

> **后端限制：** faster-whisper streaming API 只提供句级 1-best，因此该后端的
> `n_best=5` 不会产生真实句级候选。需要文本 N-best 的 GER 训练时使用
> `--asr-backend openai-whisper`；该实现会跨配置温度收集去重候选。

---

### 2.2 VSR — AV-HuBERT Large

| 属性 | 值 |
|------|----|
| checkpoint | `self_large_vox_433h.pt` |
| 编码器输出维度 | 1024-dim |
| 视频帧率 | 25 fps |
| 唇语解码 | sentencepiece BPE，`_decode_lip()` |
| 音频占位 | `audio_zeros` (104-dim placeholder) |
| 框架 | fairseq (非 PyPI 版本，需 git clone) |

**技术方向：**
- `AV-HuBERT 预训练` → **多模态自监督表征学习** (Audio-Visual Self-supervised Learning, masked prediction)
- `Conv3D frontend` → **视频时空特征提取** (3D Convolutional Spatio-temporal Encoding)
- `sentencepiece BPE decoder` → **视觉语音识别 / 唇语识别** (Visual Speech Recognition, VSR)

> **解码诊断：** 预训练 checkpoint 没有 decoder/dictionary 时会自动关闭
> `lip_hyp`；运行时解码失败会写入 `last_decode_error`、输出 warning，并按配置
> 禁用本次文本通道。连续视觉特征 `<AV_CTX>` 仍可使用。

---

### 2.3 说话人声纹 — ECAPA-TDNN

| 属性 | 值 |
|------|----|
| 来源 | SpeechBrain `spkrec-ecapa-voxceleb` |
| 输出维度 | 192-dim, L2-normalized |
| 框架 | SpeechBrain ≥ 1.0.0 |

**技术方向：**
- **说话人表征学习 / 时延神经网络** (Speaker Embedding, Temporal-Dilated Neural Network with SE-blocks)

---

### 2.4 人脸身份 — ArcFace ResNet-100

| 属性 | 值 |
|------|----|
| 模型包 | InsightFace `buffalo_l` |
| 输出维度 | 512-dim, L2-normalized |
| 推理后端 | ONNX Runtime |
| 人脸选取策略 | 最大检测框 (largest bbox) |

**技术方向：**
- **人脸表征学习 / 角度间距度量学习** (Face Recognition, Angular Margin Metric Learning — ArcFace loss)

---

### 2.5 大语言模型 — 本地稠密 Hugging Face Causal LM

| 属性 | 值 |
|------|----|
| 已注册模型 | Qwen2.5-3B（默认）、Llama-3.2-3B、Llama-3-8B（AMI full-v4 主模型）、Qwen2.5-7B |
| 配置 | `configs/default.yaml`、`qwen25_3b.yaml`、`llama32_3b.yaml`、`llama3_8b.yaml`、`qwen25_7b.yaml` |
| 权重策略 | `ger.model_path` 本地稠密权重；默认 `allow_download: false` |
| 精度 | `dtype: auto`，或显式 `fp32` / `fp16` / `bf16` |
| 解码 | Greedy decode (`do_sample=False`) |
| 最大生成 | `max_new_tokens=64` |
| 接口 | `inputs_embeds`（嵌入级输入，绕过 tokenizer） |

**技术方向：**
- **大语言模型 / 指令微调** (Instruction-tuned Autoregressive LM, Qwen2/Llama architecture)

---

## 3. C1 身份识别

### 3.1 IdentityFuser MLP

```
voice_emb [192] ──voice_proj──► [256]
                                        ├── cat ──► [512] ──fuse──► [256] = z_id
face_emb  [512] ──face_proj──► [256]
```

| 层 | 结构 | 技术方向 |
|----|------|---------|
| `voice_proj` | Linear(192→256) + GELU | 投影头设计 (Projection Head) |
| `face_proj` | Linear(512→256) + GELU | 投影头设计 (Projection Head) |
| `fuse` | Linear(512→256) + GELU + Linear(256→256) | **多模态嵌入融合** (Multi-modal Feature Fusion) |
| L2-Norm | 归一化输出 | **度量学习 / 归一化嵌入空间** (Normalized Embedding Space for Cosine Similarity) |

---

### 3.2 IdentityPool

| 组件 | 技术方向 |
|------|---------|
| Cosine similarity query | **度量学习 / 最近邻检索** (Metric Learning, Nearest-neighbour Retrieval) |
| EMA Update (α=0.1) | **在线原型更新 / 动量平滑** (Online Prototype Update, Exponential Moving Average) |
| `top_k=3` 候选池 | **top-k 检索** (Top-k Candidate Retrieval) |

---

### 3.3 AgglomerativeColdStart

| 参数 | 值 | 技术方向 |
|------|----|---------| 
| linkage | average | **无监督聚类 / 层次聚类** (Agglomerative Hierarchical Clustering) |
| `distance_threshold=0.55` | 自动确定 K | **开集识别 / 阈值决策** (Open-set Recognition) |
| `delta_unknown=0.65` | 未知说话人拒绝 | **开集检测** (Open-set Detection, Threshold-based Rejection) |

---

### 3.4 DualGate

| 条件 | 阈值 | 技术方向 |
|------|------|---------|
| `tau_a_snr_db` | SNR > 8 dB | **信号质量估计** (Blind SNR Estimation, Percentile Energy Heuristic) |
| `tau_v_lip_conf` | lip_conf > 0.7 | **视觉质量过滤** |
| AND 逻辑门 | 两路同时满足才更新 | **多模态质量感知硬门控** (Multi-modal Quality-aware Hard Gating) |
| 帧率对齐 | 最近邻上采样 | **时序对齐 / 帧率重采样** (Temporal Alignment, Nearest-neighbour Resampling) |

---

### 3.5 训练损失：BidirectionalInfoNCE

```
L_total = L_{A→V} + L_{V→A}

L_{A→V} = -1/N Σ log [ exp(sim(a_i, v_i)/τ) / Σ_j exp(sim(a_i, v_j)/τ) ]
```

| 参数 | 值 | 技术方向 |
|------|----|---------| 
| τ (temperature) | 0.07 | **对比学习温度超参** (Temperature Scaling for Contrastive Loss Sharpness) |
| 双向对称 | 必选 | **双向对比自监督学习** (Bidirectional Contrastive Self-supervised Learning, Symmetric InfoNCE) |

> **规格注意**：单向 InfoNCE 会使融合向量偏向主导模态，破坏跨模态检索。规格强制要求双向。

---

## 4. C2 对齐与 GER

### 4.1 IDConditionedAligner

#### 身份注入 — ConcatLinearInject

```
Whisper hidden [T_a, 1280]
z_id [256] ──── expand ──► [T_a, 256]
     cat ──► [T_a, 1536] ──Linear──► [T_a, 512] = 条件化特征
```

**技术方向：条件特征注入 / 身份调制** (Conditional Feature Injection, Identity Conditioning via Concatenation)

---

#### 注意力 — SoftGatedCrossAttention

```
Q ← LayerNorm(audio_feat)        # Q-Norm (Pre-norm)
K,V ← LayerNorm(visual_feat)     # KV-Norm (Pre-norm)

logits = Q @ K.T / sqrt(d)
logits += log(speaker_gate + eps)   # 加性对数偏置
attn = softmax(logits) @ V
```

| 组件 | 技术方向 |
|------|---------|
| Q-Norm, KV-Norm (LayerNorm before Q/K/V) | **Pre-norm Transformer 架构 / 训练稳定性** (Pre-norm Architecture for Gradient Flow) |
| `logits += log(gate+eps)` 加性对数偏置 | **软注意力门控** (Soft Attention Gating in Log-domain) |
| `speaker_mask_v` per-speaker 掩码 | **说话人感知注意力掩码** (Speaker-aware Attention Masking) |
| 跨模态 Q(audio)→KV(visual) | **跨模态注意力** (Cross-modal Attention, Audio queries Visual keys/values) |

---

#### Transformer Stack（2层）

| 组件 | 技术方向 |
|------|---------|
| FFN: Linear→GELU→Linear | **位置级非线性变换** (Position-wise Feed-Forward Network) |
| Post-FFN LayerNorm | **训练稳定 / 梯度流** (Layer Normalization, Gradient Flow Stabilization) |
| 2层叠加 | **深层上下文建模** (Deep Contextual Representation, Multi-layer Transformer Stack) |

---

### 4.2 QFormerProjector

```
16 learned queries [16, d_model]
    │  cross-attention attends to f_align [T, 512]
    ▼
[16, d_model] ──Linear──► [16, 4096] = soft visual tokens → LLM
```

| 组件 | 技术方向 |
|------|---------|
| 16 learned query embeddings (nn.Parameter) | **软提示 / 可学习查询** (Soft Prompting, Learnable Query Tokens — BLIP-2 Q-Former架构) |
| Cross-attention query→f_align | **跨模态桥接 / 特征压缩** (Cross-modal Bridging, Attention-based Feature Compression) |
| Linear(d_model→4096) | **模态维度对齐** (Modality Dimensionality Alignment, LLM input projection) |

---

### 4.3 GER Head（已注册 Causal LM + LoRA）

#### 提示结构

```
[Speaker: ID_i]                        ← id_proj 加性偏置注入身份
Audio hypothesis: <ASR n-best>
Visual hypothesis: <lip_hyp>
Aligned feature context: <AV_CTX>     ← 替换为 QFormer 16个软 token
Correct the transcript...
Output:
<TARGET TOKENS>                        ← 训练时 teacher-forced CE
```

| 组件 | 技术方向 |
|------|---------|
| `id_proj` 加性偏置 on `[Speaker: ID_i]` token | **身份条件嵌入注入** (Identity-conditioned Token Embedding Injection) |
| `inputs_embeds` 接口 + `<AV_CTX>` 占位符分割 | **嵌入级提示工程** (Embedding-level Prompt Construction) |
| `apply_chat_template` (user-only role) | **指令微调对话格式化** (Instruction-tuned Chat Formatting) |
| `_clean_generated_text()` 后处理 | **规则式输出过滤** (Rule-based Post-processing, Artifact Removal) |

#### LoRA 配置

| 参数 | 值 | 技术方向 |
|------|----|---------| 
| r=16, α=32 | 低秩分解 | **参数高效微调** (Parameter-Efficient Fine-Tuning, Low-Rank Adaptation) |
| target modules | q/k/v/o/gate/up/down_proj (全部) | **全模块 LoRA** (Full-module LoRA coverage) |
| dropout=0.05 | LoRA dropout | **正则化** |
| 本地稠密权重 + `dtype` | `auto` / FP32 / FP16 / BF16 | **可复现本地推理与训练** |

> **训练状态说明：** 本地稠密权重由统一的 `LocalHFCausalLMBackend` 加载，
> LoRA 与 QFormer/identity projector 在新建运行中从可训练初始值开始；正式推理必须加载
> 与模型 family、hidden size、tokenizer 及 LoRA targets 匹配的 Stage-2 GER checkpoint。

#### GER 训练损失

```python
labels = [-100] * P_prompt + [token_ids of target]
loss = cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

**技术方向：条件语言模型监督学习** (Conditional LM Supervised Training, Teacher Forcing with selective loss masking)

---

## 5. C3 置信度与闭环反馈

### 5.1 ConfidenceScorer（4信号加权）

| 信号 | 权重 | 计算方式 | 技术方向 |
|------|------|---------|---------|
| `asr_rescore` | 0.60 | teacher-forced mean token log-prob → sigmoid squash | **声学模型重打分** (Acoustic Rescoring) |
| `av_consistency` | 0.25 | `z_id` 与帧嵌入余弦相似度 | **跨模态一致性度量** (Cross-modal Consistency, Cosine Similarity) |
| `nbest_variance` | 0.10 | N-best Levenshtein 相似度方差 | **N-best 不确定性估计** (Hypothesis Uncertainty Estimation) |
| `llm_entropy` | 0.05 | 生成 token 分布熵 | **语言模型不确定性** (LLM Output Entropy, Predictive Uncertainty) |

`squash_logprob(lp) = sigmoid(lp × 1.5 + 2)` — **对数概率校准** (Log-prob Calibration, Sigmoid Squashing)

---

### 5.2 Safety Gates（生成安全约束）

| 门控 | 规则 | 技术方向 |
|------|------|---------|
| Artifact blacklist | 含禁词则拒绝 | **规则式后处理过滤** |
| Length ratio | `len(ger) / len(asr) ≤ 1.8` | **长度比约束** (Length Ratio Guard) |
| Token overlap | `overlap(ger, asr) ≥ 0.5` | **词汇重叠约束** (Token Overlap Constraint) |
| Acoustic fallback | 低置信度回退到 ASR 1-best | **安全回退策略** (Acoustic Safety Fallback) |

**整体技术方向：生成安全约束 / 输出质量控制** (Generation Safety Constraints, Output Quality Filtering)

---

### 5.3 ClosedLoopController（4动作决策）

| 动作 | 触发条件 | 技术方向 |
|------|---------|---------|
| `ACCEPT_AND_UPDATE` | conf ≥ tau_update (0.55) | EMA 更新 identity pool |
| `ACCEPT_NO_UPDATE` | mid ≤ conf < tau_update | 接受但不更新 |
| `REALIGN` | conf_low ≤ conf < mid | 重新跑 C2 对齐 |
| `REIDENTIFY` | conf < conf_low (0.35) | 重新跑 C1 识别 |

**技术方向：闭环控制 / 规则式反馈决策** (Closed-loop Control, Rule-based Decision Policy)；`max_iters=3` 防止无限循环。

---

## 6. 前端预处理 Frontend

### 6.1 处理流程（离线）

```
原始视频 (.mp4)
    │
    ├─► pyannote diarization          # 说话人分割
    ├─► Light-ASD (ASD)               # 主动说话人检测
    ├─► RetinaFace / InsightFace      # 人脸检测
    │       └─► SORT 跟踪             # 多目标跟踪
    └─► dlib 68点关键点               # 唇部 ROI 提取
            └─► mean-face 仿射变换
                    └─► [T,1,96,96] .npy 文件
                            └─► Manifest JSON
```

### 6.2 各组件技术方向

| 组件 | 技术方向 |
|------|---------|
| pyannote/speaker-diarization-community-1 | **说话人分割** (Speaker Diarization, Neural Segmentation + Embedding Clustering) |
| Light-ASD | **主动说话人检测** (Audio-Visual Active Speaker Detection, AV temporal classification) |
| RetinaFace / InsightFace 人脸检测 | **人脸检测** (Face Detection, Anchor-free RetinaNet-based) |
| SORT 多目标跟踪 | **多目标跟踪** (Multi-Object Tracking, Kalman Filter + IoU matching) |
| dlib 68点关键点 + mean-face 仿射变换 | **人脸对齐 / 标准化** (Face Alignment, Affine Warping to canonical face) |
| MouthROIExtractor `[T,1,96,96]` | **唇部 ROI 提取** (Mouth ROI Extraction for VSR preprocessing) |
| haar 级联分类器 (fallback) | **传统 CV** (Classical CV, Viola-Jones Cascade Classifier) |

### 6.3 Frontend Profiles

| profile | 说话人分割 | ASD | 说明 |
|---------|-----------|-----|------|
| `oracle_turns` | 参考标注 | — | 上界基线 |
| `common_pyannote_lightasd` | pyannote | Light-ASD | 默认生产配置 |
| `strong_sortformer_talknet` | SortFormer | TalkNet | 强力配置 |
| `degraded_pyannote` | pyannote | — | 降级对照 |

---

## 7. 训练目标

### Stage 1（冻结 ASR/VSR 编码器）

| 目标 | 权重/说明 | 技术方向 |
|------|---------|---------|
| `infonce_av` + `infonce_va` | 双向对称 τ=0.07 | **双向对比自监督学习** |
| `ctc` | 序列标注辅助 | **连接时序分类** (CTC, Connectionist Temporal Classification) |
| 模型选择 | dev 双向检索准确率均值；保存并恢复 `best.pt` | **验证集模型选择** |

### Stage 2（解冻全部）

| 目标 | 技术方向 |
|------|---------|
| `ger_ce` (teacher-forced CE) | **条件 LM 监督训练** (Teacher Forcing) |
| `ctc` | CTC 辅助 |
| `infonce_av` + `infonce_va` | 持续对比对齐 |

**学习率策略：** Stage 2 lr = Stage 1 lr × 0.1 (`lr_ratio_to_stage1: 0.1`)

---

## 8. 评估指标

| 指标 | 含义 | 技术方向 |
|------|------|---------|
| **SA-WER** | 说话人归因词错率 | **说话人感知 WER** (Speaker-attributed Word Error Rate) |
| **WER** | 词错率 | 标准 ASR 评估 |
| **SCR** | 说话人混淆率 | **身份级错误度量** (Speaker Confusion Rate) |
| **AV-SID Acc** | 音视频身份识别准确率 | **多模态身份识别评估** |
| **DER** | miss + FA + confusion | **分段级说话人评估** (Diarization Error Rate) |
| **JER** | mean(1 - Jaccard) per speaker | **说话人级公平评估** (Jaccard Error Rate) |

**关键实现：**
- Hungarian assignment（scipy）：**最优二分图匹配** (Optimal Bipartite Matching)，用于假设→参考说话人标签映射
- Word-level Levenshtein：**序列对齐 / 动态规划** (Sequence Alignment, DP-based Edit Distance)

---

## 9. 依赖与环境

### PyTorch 安装（必须手动）

```bash
# RTX 50-series (sm_120, cu128)
pip install "torch>=2.7" "torchaudio>=2.7" "torchvision>=0.22" \
    --index-url https://download.pytorch.org/whl/cu128

# H100/A100/RTX 30-40 series (cu124)
pip install "torch==2.6.*" "torchaudio==2.6.*" "torchvision==0.21.*" \
    --index-url https://download.pytorch.org/whl/cu124
```

### 关键依赖版本约束

| 包 | 版本约束 | 原因 |
|----|---------|------|
| `numpy` | `>=1.24, <2.0` | 保持 InsightFace/ONNX/OpenCV ABI 兼容 |
| `transformers` | `>=4.49, <4.55` | 兼容性上限 |
| `tokenizers` | `>=0.21, <0.22` | transformers 4.49+ 要求 |
| `opencv-python` / `opencv-python-headless` | `>=4.9, <4.10` | 避免传递依赖升级到 NumPy 2.x |
| `huggingface_hub` | `>=0.34, <1.0` | 提供 `hf` CLI 并保持 transformers 兼容 |

### 非 PyPI 依赖

```bash
# fairseq（PyPI 版本陈旧）
pip install "git+https://github.com/facebookresearch/fairseq.git@v0.12.2"

# AV-HuBERT（需加入 PYTHONPATH）
git clone https://github.com/facebookresearch/av_hubert.git
```

---

## 附录：关键已知问题

| 问题 | 根本原因 | 文件 |
|------|---------|------|
| face_emb 全零（身份池仅声纹） | manifest 无 `enrollment_face` 字段 | frontend manifest |
| 无 per-speaker 视觉掩码 | manifest turns 无 `speaker_mask_v` | frontend manifest |
| 新建 GER 运行不可直接当作已训练模型 | LoRA/QFormer/projector 需要 Stage-2 训练或兼容 checkpoint | `c2_alignment/ger_head.py` |
| `faster-whisper` 仅返回句级 1-best | 流式 segment API 不暴露真实句级 N-best；需要文本 N-best 时用 `openai-whisper` | `backbones/asr_whisper.py` |
| VSR 文本可能为空 | 预训练 checkpoint 没有 decoder/dictionary；解码异常会记录 `last_decode_error` 并禁用本次文本通道 | `backbones/vsr_avhubert.py` |
