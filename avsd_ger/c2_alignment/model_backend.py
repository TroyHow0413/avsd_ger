"""Model backends for GER.

Only dense, local Hugging Face causal language models are supported.  Keeping
path resolution here prevents an accidental Hub download from model code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelProfile:
    family: str
    hidden_size: int
    hf_model_types: tuple[str, ...]
    lora_target_modules: tuple[str, ...]


MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen2.5-3b-instruct": ModelProfile(
        family="qwen2.5-3b-instruct",
        hidden_size=2048,
        hf_model_types=("qwen2",),
        lora_target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
    "llama-3.2-3b-instruct": ModelProfile(
        family="llama-3.2-3b-instruct",
        hidden_size=3072,
        hf_model_types=("llama",),
        lora_target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
}


def get_model_profile(cfg: dict[str, Any]) -> ModelProfile:
    family = str(cfg.get("model_family", "qwen2.5-3b-instruct")).lower()
    try:
        return MODEL_PROFILES[family]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(
            f"Unsupported GER model_family={family!r}; supported: {supported}"
        ) from exc


class FakeTokenizer:
    """Small tokenizer double used by unit and pipeline wiring tests."""

    def __init__(self, speaker_token: str):
        self.speaker_token = speaker_token
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.chat_template = "fake-chat-template"
        self._speaker_id = 2

    def __len__(self) -> int:
        return 258

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._speaker_id if token == self.speaker_token else 3

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        del tokenize
        body = "\n".join(message["content"] for message in messages)
        return f"<|user|>\n{body}<|assistant|>\n" if add_generation_prompt else body

    def __call__(
        self, text: str, *, return_tensors: str, add_special_tokens: bool,
    ) -> SimpleNamespace:
        del return_tensors
        ids: list[int] = [1] if add_special_tokens else []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.speaker_token, cursor):
                ids.append(self._speaker_id)
                cursor += len(self.speaker_token)
            else:
                ids.append(4 + (ord(text[cursor]) % 250))
                cursor += 1
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return ""


class FakeCausalLM(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 258):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, model_type="fake")
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


class FakeBackend:
    kind = "fake"

    def __init__(self, cfg: dict[str, Any], device: torch.device):
        self.profile = get_model_profile(cfg)
        self.model_path = "<fake>"
        self.tokenizer = FakeTokenizer(
            str(cfg.get("speaker_special_token", "[Speaker: ID_i]"))
        )
        self.model = FakeCausalLM(self.profile.hidden_size).to(device)
        self.hidden_size = self.profile.hidden_size
        self.speaker_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.speaker_token
        )
        configured = cfg.get("lora", {}).get("target_modules")
        self.lora_target_modules = (
            self.profile.lora_target_modules
            if configured is None or configured == "auto"
            else tuple(str(item) for item in configured)
        )
        unsupported = sorted(
            set(self.lora_target_modules) - set(self.profile.lora_target_modules)
        )
        if unsupported:
            raise ValueError(
                f"LoRA targets not supported by {self.profile.family}: {unsupported}"
            )


class LocalHFCausalLMBackend:
    """Dense causal LM materialized in a configured local directory."""

    kind = "local_hf"

    def __init__(self, cfg: dict[str, Any], device: torch.device):
        self.profile = get_model_profile(cfg)
        self.device = device
        self.model_path = self._resolve_model_path(cfg)
        self.lora_target_modules = self._resolve_lora_targets(cfg)

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        model_config = AutoConfig.from_pretrained(
            self.model_path, local_files_only=True
        )
        model_type = str(getattr(model_config, "model_type", ""))
        hidden_size = int(getattr(model_config, "hidden_size", -1))
        if model_type not in self.profile.hf_model_types:
            raise ValueError(
                f"Configured {self.profile.family} requires HF model_type in "
                f"{self.profile.hf_model_types}, but local config reports {model_type!r}"
            )
        if hidden_size != self.profile.hidden_size:
            raise ValueError(
                f"Configured {self.profile.family} expects hidden_size "
                f"{self.profile.hidden_size}, but local config reports {hidden_size}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, local_files_only=True
        )
        if cfg.get("require_chat_template", True) and not getattr(
            self.tokenizer, "chat_template", None
        ):
            raise ValueError(
                f"Local tokenizer for {self.profile.family} has no chat_template"
            )
        speaker_token = str(
            cfg.get("speaker_special_token", "[Speaker: ID_i]")
        )
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [speaker_token]}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.speaker_token_id = self.tokenizer.convert_tokens_to_ids(speaker_token)

        dtype = self._resolve_dtype(str(cfg.get("dtype", "auto")), device)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if device.type == "cuda":
            load_kwargs["device_map"] = {"": device.index or 0}
            load_kwargs["attn_implementation"] = str(
                cfg.get("attn_implementation", "sdpa")
            )
        base = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        if added:
            try:
                base.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            except TypeError:
                base.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = int(base.config.hidden_size)
        if self.hidden_size != self.profile.hidden_size:
            raise ValueError(
                f"Configured {self.profile.family} expects hidden_size "
                f"{self.profile.hidden_size}, but local model reports {self.hidden_size}"
            )
        available_suffixes = {
            name.rsplit(".", 1)[-1] for name, _ in base.named_modules()
        }
        missing_targets = sorted(
            set(self.lora_target_modules) - available_suffixes
        )
        if missing_targets:
            raise ValueError(
                f"Local {self.profile.family} model is missing configured LoRA "
                f"target modules: {missing_targets}"
            )
        lora = cfg.get("lora", {})
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(self.lora_target_modules),
        )
        self.model = get_peft_model(base, lora_cfg)
        if device.type != "cuda":
            self.model = self.model.to(device)

    @staticmethod
    def _resolve_model_path(cfg: dict[str, Any]) -> str:
        raw = cfg.get("model_path", cfg.get("llm_name"))
        if not raw:
            raise ValueError("ger.model_path is required for backend=local_hf")
        path = Path(str(raw)).expanduser()
        if path.is_dir() and (path / "config.json").is_file():
            return str(path.resolve())

        if not bool(cfg.get("allow_download", False)):
            raise FileNotFoundError(
                f"GER local model is incomplete or missing: {path}. "
                "Set ger.allow_download=true and ger.model_id to materialize it."
            )

        model_id = cfg.get("model_id")
        if not model_id:
            raise ValueError(
                "ger.model_id is required when ger.allow_download=true"
            )
        from huggingface_hub import snapshot_download

        path.mkdir(parents=True, exist_ok=True)
        print(
            f"[GER backend] Local model not found; downloading {model_id!r} "
            f"to {path.resolve()}",
            flush=True,
        )
        snapshot_download(repo_id=str(model_id), local_dir=str(path))
        if not (path / "config.json").is_file():
            raise RuntimeError(
                f"Hugging Face download completed without config.json: {path}"
            )
        return str(path.resolve())

    def _resolve_lora_targets(self, cfg: dict[str, Any]) -> tuple[str, ...]:
        configured = cfg.get("lora", {}).get("target_modules")
        if configured is None or configured == "auto":
            return self.profile.lora_target_modules
        actual = tuple(str(item) for item in configured)
        unsupported = sorted(set(actual) - set(self.profile.lora_target_modules))
        if unsupported:
            raise ValueError(
                f"LoRA targets not supported by {self.profile.family}: {unsupported}"
            )
        return actual

    @staticmethod
    def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
        value = value.lower()
        if value == "auto":
            if device.type != "cuda":
                return torch.float32
            return (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        choices = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if value not in choices:
            raise ValueError(
                f"Unsupported dense GER dtype={value!r}; use auto/fp32/fp16/bf16"
            )
        selected = choices[value]
        if device.type == "cpu" and selected == torch.float16:
            raise ValueError("fp16 is not supported for the dense GER backend on CPU")
        if (
            device.type == "cuda"
            and selected == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bf16 was requested but this CUDA device does not support it")
        return selected


def create_model_backend(
    cfg: dict[str, Any], device: torch.device, *, force_fake: bool = False
) -> FakeBackend | LocalHFCausalLMBackend:
    kind = "fake" if force_fake else str(cfg.get("backend", "local_hf")).lower()
    if kind == "fake":
        return FakeBackend(cfg, device)
    if kind == "local_hf":
        return LocalHFCausalLMBackend(cfg, device)
    raise ValueError(f"Unsupported GER backend={kind!r}; use local_hf or fake")
