"""C2 — Identity-Aware GER head (spec-aligned).

Default LLM: **Llama-3-8B-Instruct** with LoRA (spec §5.2, §6).
Prompt template: spec §2 C2, including the **speaker-consistency constraint**
('Preserve the speaker label.') which prevents the LLM from hallucinating
words across speaker boundaries.

The special token `[Speaker: ID_i]` is registered via `add_special_tokens`
so gradients can flow into a dedicated embedding rather than being split
across BPE subwords (see spec §17 debugging tip: 'SCR not decreasing after
adding ID prefix → special token not in vocab').

`f_align` is projected as a small Q-Former soft prefix that replaces the
`<AV_CTX>` placeholder in the prompt.
"""

from __future__ import annotations

from pydoc import text
from typing import Any

import torch
import torch.nn as nn


class QFormerProjector(nn.Module):
    """Compress variable-length f_align [N,D] into a fixed n_queries soft prefix."""

    def __init__(self, d_in: int, d_llm: int, n_queries: int = 16, n_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_in) * 0.02)
        self.attn = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.out = nn.Linear(d_in, d_llm)

    def forward(self, f_align: torch.Tensor) -> torch.Tensor:
        if f_align.ndim == 2:
            f_align = f_align.unsqueeze(0)
        B = f_align.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.attn(q, f_align, f_align, need_weights=False)
        return self.out(z)  # [B, n_q, d_llm]


class GERHead(nn.Module):
    """
    Prompt layout:
        [<BOS>] [sys tokens]
                [Speaker: ID_i]           ← special token embedding
                Audio hypothesis: <asr_nbest>
                Visual hypothesis: <lip_hyp>
                Aligned feature context: <AV_CTX_SOFT_TOKENS>
                Correct the transcript. Preserve the speaker label.
                Output:
                <generated text>
    """

    DEFAULT_LLM_HIDDEN = 4096  # Llama-3-8B hidden size

    def __init__(
        self,
        cfg: dict[str, Any],
        z_dim: int,
        d_align: int,
        stub: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.stub = stub
        self.device = torch.device(device)
        self.template: str = cfg["prompt_template"]
        self.max_new_tokens = int(cfg["max_new_tokens"])
        self.speaker_special_token: str = cfg.get(
            "speaker_special_token", "[Speaker: ID_i]"
        )

        self._tok = None
        self._llm = None
        self._llm_embed_dim = self.DEFAULT_LLM_HIDDEN
        self._spk_token_id: int | None = None

        if not stub:
            self._load_llm()

        # Per-speaker ID token is formed as:   id_embed + id_projection(z_id)
        # so the special token carries both a learned prior and per-utterance info.
        # Note: .to(self.device) must be applied here. nn.Module parameters
        # default to CPU; without an explicit move, MultiheadAttention's K/V
        # linear projection mixes CPU weights with CUDA inputs at forward()
        # time and crashes with `Expected all tensors to be on the same device`.
        self.id_proj = nn.Linear(z_dim, self._llm_embed_dim).to(self.device)
        self.qformer = QFormerProjector(d_in=d_align, d_llm=self._llm_embed_dim).to(
            self.device
        )

    # --------------------------------------------------------------- load LLM
    def _load_llm_old(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        llm_id = self.cfg["llm_name"]
        self._tok = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        # Register the speaker special token so we have a dedicated embedding
        # rather than BPE-fragmented pieces like '['/'Speaker'/':'/…
        num_added = self._tok.add_special_tokens(
            {"additional_special_tokens": [self.speaker_special_token]}
        )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._spk_token_id = self._tok.convert_tokens_to_ids(self.speaker_special_token)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # ---- LLM quantization mode ------------------------------------------
        # cfg.ger.llm_quant: "auto" | "fp16" | "int8" | "4bit"
        # auto rules (based on total GPU VRAM):
        #     >= 40 GB           -> fp16  (~16 GB weights + KV cache)
        #     >= 16 GB and <40   -> int8  (~9 GB)
        #     <  16 GB           -> 4bit  (~5-6 GB nf4)
        # CPU device              -> fp32 (no quantization)
        quant_mode = str(self.cfg.get("llm_quant", "auto")).lower()
        if quant_mode == "auto":
            if self.device.type != "cuda":
                quant_mode = "fp32"
            else:
                total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
                if total_gb >= 40:
                    quant_mode = "fp16"
                elif total_gb >= 16:
                    quant_mode = "int8"
                else:
                    quant_mode = "4bit"
        print(
            f"[GERHead] Llama-3 load mode: {quant_mode} "
            f"(GPU: {torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})"
        )

        # ---- accelerate device_map: force single-GPU placement when quantizing
        # device_map='auto' estimates *fp16* size BEFORE bnb quantization, so it
        # over-budgets and tries to CPU-offload, which 4bit/int8 doesn't allow.
        # device_map={'': 0} forces everything on GPU 0 — quantized model fits.
        if quant_mode in ("4bit", "int8"):
            _device_map = {"": 0}
        else:
            _device_map = "auto"

        if quant_mode == "4bit":
            from transformers import BitsAndBytesConfig

            _bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                llm_id,
                quantization_config=_bnb_cfg,
                device_map=_device_map,
            )
        elif quant_mode == "int8":
            from transformers import BitsAndBytesConfig

            _bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,  # default; preserves outliers as fp16
            )
            base = AutoModelForCausalLM.from_pretrained(
                llm_id,
                quantization_config=_bnb_cfg,
                device_map=_device_map,
            )
        elif quant_mode == "fp16":
            base = AutoModelForCausalLM.from_pretrained(
                llm_id,
                torch_dtype=torch.float16,
                device_map=_device_map,
            )
        else:  # fp32 (CPU fallback)
            base = AutoModelForCausalLM.from_pretrained(
                llm_id,
                torch_dtype=torch.float32,
                device_map=_device_map,
            )
        if num_added:
            base.resize_token_embeddings(len(self._tok))
        self._llm_embed_dim = base.config.hidden_size

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(self.cfg["lora"]["r"]),
            lora_alpha=int(self.cfg["lora"]["alpha"]),
            lora_dropout=float(self.cfg["lora"]["dropout"]),
            target_modules=list(self.cfg["lora"]["target_modules"]),
        )
        self._llm = get_peft_model(base, lora_cfg).to(self.device)

    def _load_llm(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        llm_id = self.cfg["llm_name"]
        self._tok = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        
        # 原有speaker特殊token逻辑完全保留，无任何改动
        num_added = self._tok.add_special_tokens(
            {"additional_special_tokens": [self.speaker_special_token]}
        )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._spk_token_id = self._tok.convert_tokens_to_ids(self.speaker_special_token)

        # ===================== 无bnb 自适应量化逻辑 =====================
        if self.device.type == "cuda":
            # 自动识别RTX40/50系显卡，开启原生FP8
            major, minor = torch.cuda.get_device_capability(self.device.index)
            if major >= 8 and minor >= 9:
                # E4M3格式适配前向推理，E5M2适配反向梯度，兼顾精度与范围
                dtype = torch.float8_e4m3fn
                compute_dtype = torch.bfloat16
                load_mode = "FP8" #(Native RTX 5080 optimized)
                # 开启FP8矩阵乘法硬件加速
                torch.set_float8_matmul_enabled(True)
            else:
                dtype = torch.float16
                compute_dtype = torch.float16
                load_mode = "FP16"
            _device_map = "auto"
            attn_implementation = "sdpa"
        else:
            dtype = torch.float32
            compute_dtype = torch.float32
            load_mode = "FP32 (CPU fallback)"
            _device_map = "auto"
            attn_implementation = "eager"

        print(f"[GERHead] Llama-3 load mode: {load_mode} "
            f"(GPU: {torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})")

        # ===================== 无bnb 模型加载（核心改动） =====================
        base = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=dtype,
            device_map=_device_map,
            attn_implementation=attn_implementation,
            # 开启梯度检查点，训练显存再降50%，完全无精度损失
            use_cache=False,  # 训练时关闭KV缓存，节省显存
        )
        # 训练时开启梯度检查点
        base.gradient_checkpointing_enable()

        # 原有token embedding扩容逻辑完全保留
        if num_added:
            base.resize_token_embeddings(len(self._tok))
        self._llm_embed_dim = base.config.hidden_size

        # ===================== 原有LoRA逻辑完全保留，可升级为DoRA效果更好 =====================
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(self.cfg["lora"]["r"]),
            lora_alpha=int(self.cfg["lora"]["alpha"]),
            lora_dropout=float(self.cfg["lora"]["dropout"]),
            target_modules=list(self.cfg["lora"]["target_modules"]),
            # 可选：开启DoRA，替代基础LoRA，收敛速度+纠错效果全面提升
            use_dora=True,
        )
        # 计算dtype统一，避免多模态模块dtype不匹配报错
        self._llm = get_peft_model(base, lora_cfg).to(self.device, dtype=compute_dtype)
    # --------------------------------------------------------------- prompt
    def _render_text(
        self, speaker_id: str | None, nbest: list[str], lip_hyp: str
    ) -> str:
        """Render the GER user-message AND wrap it in the LLM's chat template.

        Without chat-template wrapping Llama-3-Instruct degenerates to base-LM
        behaviour and just continues the prompt (echoing 'Audio hypothesis:'
        / 'Visual hypothesis:' patterns instead of producing the corrected
        transcript). apply_chat_template emits the proper
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>...<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|> envelope so Llama-3
        knows it's expected to ANSWER, not continue.

        The <AV_CTX> placeholder stays inside the user message; _inputs_embeds
        splits on it AFTER chat-template wrapping so QFormer soft tokens land
        at the right position.
        """
        tag = self.speaker_special_token
        asr_block = " | ".join(h.strip() for h in nbest if h.strip())
        user_content = self.template.format(
            speaker_tag=tag,
            asr_nbest=asr_block or "<none>",
            lip_hyp=lip_hyp or "<none>",
        )

        # Apply chat template if the tokenizer supports it (Llama-3-Instruct,
        # Qwen-Chat, Mistral-Instruct, ...). Fall back to raw text for base
        # LMs that don't ship one.
        if getattr(self._tok, "chat_template", None):
            try:
                return self._tok.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                pass  # any failure -> fall through to raw user_content
        return user_content

    def _inputs_embeds(
        self, z_id: torch.Tensor, f_align: torch.Tensor, text: str
    ) -> torch.Tensor:
        """
        Build inputs_embeds = [text_part_A, AV_CTX_soft_tokens, text_part_B]
        where the split point is the `<AV_CTX>` placeholder. The
        `[Speaker: ID_i]` token embedding is additively biased with
        id_proj(z_id) to inject the fused identity vector into the LLM.
        Returns: [1, T_total, D_llm].
        """
        if self.stub or self._llm is None:
            # Deterministic stub: 32-token sequence, fixed dim
            return torch.zeros(1, 32, self._llm_embed_dim, device=self.device)

        assert "<AV_CTX>" in text, "prompt template must contain <AV_CTX> placeholder"
        pre, post = text.split("<AV_CTX>", 1)

        pre_ids = self._tok(
            pre, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(self.device)
        post_ids = self._tok(
            post, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        embed_layer = self._llm.get_input_embeddings()
        pre_emb = embed_layer(pre_ids)  # [1, T_pre, D]
        post_emb = embed_layer(post_ids)  # [1, T_post, D]

        # Cast all small projection layers to match the LLM embedding dtype
        # (float16 in 4bit/fp16 mode, bfloat16 in some configs, float32 on CPU).
        # Without this, nn.Linear weights stay float32 while inputs are float16
        # → "mat1 and mat2 must have the same dtype" in every linear call.
        target_dtype = pre_emb.dtype
        self.qformer.to(target_dtype)
        self.id_proj.to(target_dtype)

        # Q-Former projects f_align (variable-length token features) into a
        # fixed-length soft prefix replacing <AV_CTX>.
        av_emb = self.qformer(f_align.to(self.device).to(target_dtype))  # [1, n_q, D]

        # Inject identity by additively biasing the [Speaker: ID_i] token
        # embedding with id_proj(z_id). This is what makes the GER head
        # speaker-aware (spec section 5.2).
        if self._spk_token_id is not None:
            id_bias = self.id_proj(z_id.to(self.device).to(target_dtype))  # [D]
            spk_pos = (pre_ids[0] == self._spk_token_id).nonzero(as_tuple=True)[0]
            if spk_pos.numel() > 0:
                pre_emb = pre_emb.clone()
                pre_emb[0, spk_pos[0]] = pre_emb[0, spk_pos[0]] + id_bias

        return torch.cat([pre_emb, av_emb, post_emb], dim=1)

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(
        self,
        z_id: torch.Tensor,
        f_align: torch.Tensor,
        nbest: list[str],
        nbest_scores: list[float] | None = None,
        lip_hyp: str = "",
        speaker_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the GER head, return {'text', 'token_logprobs', 'prompt'}."""
        if self.stub or self._llm is None:
            top = nbest[0] if nbest else ""
            return {"text": top, "token_logprobs": torch.zeros(0), "prompt": ""}

        text = self._render_text(speaker_id, nbest, lip_hyp)
        inputs_embeds = self._inputs_embeds(z_id, f_align, text)

        out = self._llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=self._tok.pad_token_id,
        )

        # `out.sequences` for inputs_embeds-based generate is just the new tokens.
        new_ids = out.sequences[0]
        text_out = self._tok.decode(new_ids, skip_special_tokens=True).strip()

        # Per-token log-probs for C3 confidence (LLM entropy component).
        token_lp = torch.zeros(0, device=self.device)
        if getattr(out, "scores", None):
            scores = torch.stack(out.scores, dim=0)           # [T_new, 1, V]
            log_probs = torch.log_softmax(scores.squeeze(1), dim=-1)
            chosen = new_ids[: scores.size(0)]
            token_lp = log_probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)

        return {"text": text_out, "token_logprobs": token_lp, "prompt": text}
