"""Whisper-large-v3 ASR wrapper (spec-aligned).

Exposes four things that downstream modules need:
  1. N-best hypotheses (text + score) from beam search.
  2. Encoder hidden states for C2's token-level pooling.
  3. Word-level timestamps -- spec section 2 C2 requires token-level alignment using
     Whisper word timestamps.
  4. `rescore(audio, text)` -- teacher-forces the HF Whisper decoder on a
     corrected transcript and returns mean token log-prob. This is the
     acoustic confidence s_i used by the C3 confidence gate.

Frozen backbone -- no gradient updates here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ASRHypothesis:
    text: str
    logprob: float


@dataclass
class WordTiming:
    word: str
    start: float    # seconds
    end: float      # seconds


@dataclass
class ASROutputs:
    nbest: list[str] = field(default_factory=list)
    nbest_scores: list[float] = field(default_factory=list)
    encoder_features: torch.Tensor | None = None     # [T_a, 1280]
    words: list[WordTiming] = field(default_factory=list)
    frame_rate_hz: float = 50.0                      # Whisper encoder ~50 fps


class WhisperASR(nn.Module):
    ENCODER_DIM = 1280

    def __init__(self, cfg: dict[str, Any], stub: bool = False, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.stub = stub
        self.device = torch.device(device)
        self.n_best = int(cfg.get("n_best", 5))
        self.beam = int(cfg.get("beam_size", 10))
        self.expose_encoder = bool(cfg.get("expose_encoder", True))
        self.word_timestamps = bool(cfg.get("word_timestamps", True))

        self._ct2 = None
        self._hf_model = None
        self._hf_processor = None
        if not stub:
            self._load_real()

    # --------------------------------------------------------------- loader
    def _load_real(self) -> None:
        if self.cfg.get("backend", "faster-whisper") == "faster-whisper":
            from faster_whisper import WhisperModel
            self._ct2 = WhisperModel(
                self.cfg.get("model_name", "large-v3"),
                device="cuda" if self.device.type == "cuda" else "cpu",
                compute_type=self.cfg.get("compute_type", "float16"),
            )
        # HF model is needed for (a) encoder hidden states and (b) rescoring.
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        hf_id = f"openai/whisper-{self.cfg.get('model_name', 'large-v3')}"
        self._hf_processor = WhisperProcessor.from_pretrained(hf_id)
        self._hf_model = WhisperForConditionalGeneration.from_pretrained(hf_id).to(self.device)
        self._hf_model.eval()

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def transcribe(self, wav: torch.Tensor | np.ndarray, sr: int = 16000) -> ASROutputs:
        if self.stub:
            return self._stub_transcribe()

        wav_np = wav.detach().cpu().numpy().astype(np.float32) if isinstance(wav, torch.Tensor) else np.asarray(wav, dtype=np.float32)

        # N-best + word timestamps from faster-whisper
        segments, _info = self._ct2.transcribe(
            wav_np,
            beam_size=self.beam,
            best_of=self.beam,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
            word_timestamps=self.word_timestamps,
            language=self.cfg.get("language", None),
        )
        seg_list = list(segments)
        text_1best = "".join(s.text for s in seg_list).strip()
        words: list[WordTiming] = []
        if self.word_timestamps:
            for seg in seg_list:
                for w in (seg.words or []):
                    words.append(WordTiming(word=w.word, start=float(w.start), end=float(w.end)))

        # TODO: replace duplicated 1-best padding with real CT2 beam outputs
        nbest = [ASRHypothesis(text_1best, 0.0)]
        while len(nbest) < self.n_best:
            nbest.append(ASRHypothesis(text_1best, -float(len(nbest))))

        enc_feats = None
        if self.expose_encoder:
            inputs = self._hf_processor(wav_np, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            enc_out = self._hf_model.model.encoder(**inputs)
            enc_feats = enc_out.last_hidden_state.squeeze(0).detach()  # [T_a, 1280]

        return ASROutputs(
            nbest=[h.text for h in nbest[: self.n_best]],
            nbest_scores=[h.logprob for h in nbest[: self.n_best]],
            encoder_features=enc_feats,
            words=words,
        )

    # --------------------------------------------------------------- rescore
    @torch.no_grad()
    def rescore(self, wav: torch.Tensor | np.ndarray, text: str, sr: int = 16000) -> float:
        """
        Acoustic confidence s_i = mean token log-prob of `text` under Whisper,
        given `wav`. Higher = more acoustically consistent.

        Returns a float in (-inf, 0]; callers typically squash to [0,1] before
        comparing to tau_update.
        """
        if self.stub:
            # deterministic-ish: longer/stranger text gets a worse score
            base = -0.25 - 0.002 * len(text)
            return float(base)

        wav_np = wav.detach().cpu().numpy().astype(np.float32) if isinstance(wav, torch.Tensor) else np.asarray(wav, dtype=np.float32)
        inputs = self._hf_processor(wav_np, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        tokenizer = self._hf_processor.tokenizer
        forced = self._hf_model.generation_config.forced_decoder_ids or []
        prefix_ids = [tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
        for _, tok_id in forced:
            prefix_ids.append(int(tok_id))
        text_ids = tokenizer(text, add_special_tokens=False).input_ids
        eos_id = self._hf_model.config.eos_token_id
        decoder_input_ids = torch.tensor([prefix_ids + text_ids + [eos_id]], device=self.device)

        out = self._hf_model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = out.logits[0, :-1, :]
        targets = decoder_input_ids[0, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        start = len(prefix_ids)
        tok_lp = logp[start - 1: -1].gather(-1, targets[start - 1: -1].unsqueeze(-1)).squeeze(-1)
        if tok_lp.numel() == 0:
            return 0.0
        return float(tok_lp.mean().item())

    # --------------------------------------------------------------- stub
    def _stub_transcribe(self) -> ASROutputs:
        T_a = 150  # ~3 s at 50 fps
        enc = torch.randn(T_a, self.ENCODER_DIM, device=self.device) if self.expose_encoder else None
        fake = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jump over the lazy dog",
            "the quick brown fox jumps over the lazy dock",
            "a quick brown fox jumps over the lazy dog",
            "the quick brown fox jumped over the lazy dog",
        ][: self.n_best]
        tokens = fake[0].split()
        dur = 3.0 / max(1, len(tokens))
        words = [WordTiming(word=t, start=i * dur, end=(i + 1) * dur) for i, t in enumerate(tokens)]
        return ASROutputs(
            nbest=fake,
            nbest_scores=[float(-i) for i in range(len(fake))],
            encoder_features=enc,
            words=words,
        )
