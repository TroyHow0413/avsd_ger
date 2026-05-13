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
        self._ow_model = None
        self._ow_tokenizer = None
        self._hf_model = None
        self._hf_processor = None
        if not stub:
            self._load_real()

    # --------------------------------------------------------------- loader
    def _load_real(self) -> None:
        backend = self.cfg.get("backend", "faster-whisper")
        if backend == "faster-whisper":
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
        elif backend == "transformers":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            hf_id = f"openai/whisper-{self.cfg.get('model_name', 'large-v3')}"
            self._hf_processor = WhisperProcessor.from_pretrained(hf_id)
            self._hf_model = WhisperForConditionalGeneration.from_pretrained(hf_id).to(self.device)
            self._hf_model.eval()
        elif backend == "openai-whisper":
            try:
                import whisper
                from whisper.tokenizer import get_tokenizer
            except ImportError as e:
                raise RuntimeError(
                    "openai-whisper backend requires the `openai-whisper` package. "
                    "Install it with: pip install openai-whisper"
                ) from e
            self._ow_model = whisper.load_model(
                self.cfg.get("model_name", "large-v3"),
                device=str(self.device),
            )
            self._ow_model.eval()
            language = self.cfg.get("language", None)
            self._ow_tokenizer = get_tokenizer(
                multilingual=getattr(self._ow_model, "is_multilingual", True),
                language=language,
                task="transcribe",
            )
        else:
            raise ValueError(
                f"Unsupported asr.backend={backend!r}; expected "
                "faster-whisper, transformers, or openai-whisper."
            )

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def transcribe(self, wav: torch.Tensor | np.ndarray, sr: int = 16000) -> ASROutputs:
        if self.stub:
            return self._stub_transcribe()

        wav_np = wav.detach().cpu().numpy().astype(np.float32) if isinstance(wav, torch.Tensor) else np.asarray(wav, dtype=np.float32)

        backend = self.cfg.get("backend", "faster-whisper")
        if backend == "openai-whisper":
            return self._openai_transcribe(wav_np, sr=sr)
        if backend == "transformers":
            return self._hf_transcribe(wav_np, sr=sr)
        return self._faster_transcribe(wav_np, sr=sr)

    @torch.no_grad()
    def _faster_transcribe(self, wav_np: np.ndarray, sr: int = 16000) -> ASROutputs:
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

        # faster-whisper's streaming segment API gives us the decoded 1-best
        # here, not a real sentence-level N-best list. Do not pad by repeating
        # the same string: that makes the GER prompt look like a transcript
        # that was spoken multiple times and encourages LLM repetition.
        nbest = [ASRHypothesis(text_1best, 0.0)]

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

    @torch.no_grad()
    def _hf_transcribe(self, wav_np: np.ndarray, sr: int = 16000) -> ASROutputs:
        inputs = self._hf_processor(wav_np, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen = self._hf_model.generate(
            **inputs,
            num_beams=self.beam,
            max_new_tokens=128,
        )
        text = self._hf_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        enc_feats = None
        if self.expose_encoder:
            enc_out = self._hf_model.model.encoder(input_features=inputs["input_features"])
            enc_feats = enc_out.last_hidden_state.squeeze(0).detach()
        words = self._uniform_words(text, duration_s=len(wav_np) / float(sr))
        return ASROutputs(
            nbest=[text],
            nbest_scores=[0.0],
            encoder_features=enc_feats,
            words=words,
        )

    @torch.no_grad()
    def _openai_transcribe(self, wav_np: np.ndarray, sr: int = 16000) -> ASROutputs:
        # openai-whisper expects 16 kHz audio. The training loader already
        # resamples records, but keep a defensive fallback for direct callers.
        if sr != 16000:
            try:
                import librosa
                wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception:
                pass

        temperatures = self.cfg.get("temperatures", [0.0, 0.2, 0.4, 0.6, 0.8])
        if not isinstance(temperatures, (list, tuple)):
            temperatures = [temperatures]
        temperatures = [float(t) for t in temperatures]
        if not temperatures:
            temperatures = [0.0]

        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for temp in temperatures:
            kwargs: dict[str, Any] = {
                "temperature": temp,
                "word_timestamps": self.word_timestamps and not results,
                "language": self.cfg.get("language", None),
                "fp16": self.device.type == "cuda",
                "verbose": False,
                "condition_on_previous_text": False,
            }
            if temp == 0.0:
                kwargs["beam_size"] = self.beam
            else:
                # openai-whisper's best_of chooses the best sample internally;
                # repeating over temperatures exposes a practical sentence-level
                # n-best list without touching private decoder APIs.
                kwargs["best_of"] = max(self.beam, self.n_best)
            result = self._ow_model.transcribe(wav_np, **kwargs)
            text = str(result.get("text", "")).strip()
            key = " ".join(text.lower().split())
            if text and key not in seen:
                seen.add(key)
                results.append(result)
            if len(results) >= self.n_best:
                break

        if not results:
            results = [{"text": ""}]

        text_1best = str(results[0].get("text", "")).strip()
        words: list[WordTiming] = []
        if self.word_timestamps:
            for seg in results[0].get("segments", []) or []:
                for w in seg.get("words", []) or []:
                    word = str(w.get("word", "")).strip()
                    if word:
                        words.append(
                            WordTiming(
                                word=word,
                                start=float(w.get("start", 0.0)),
                                end=float(w.get("end", 0.0)),
                            )
                        )
        if not words:
            words = self._uniform_words(text_1best, duration_s=len(wav_np) / float(sr))

        enc_feats = None
        if self.expose_encoder:
            enc_feats = self._openai_encoder_features(wav_np)

        nbest = [str(r.get("text", "")).strip() for r in results if str(r.get("text", "")).strip()]
        scores = [self._openai_result_score(r) for r in results[: len(nbest)]]

        return ASROutputs(
            nbest=nbest[: self.n_best] or [text_1best],
            nbest_scores=scores[: self.n_best] or [0.0],
            encoder_features=enc_feats,
            words=words,
        )

    @staticmethod
    def _openai_result_score(result: dict[str, Any]) -> float:
        segs = result.get("segments", []) or []
        vals = [
            float(seg.get("avg_logprob"))
            for seg in segs
            if seg.get("avg_logprob") is not None
        ]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    # --------------------------------------------------------------- rescore
    @torch.no_grad()
    def rescore(self, wav: torch.Tensor | np.ndarray, text: str, sr: int = 16000) -> float:
        """
        Acoustic confidence s_i = mean token log-prob of `text` under Whisper,
        given `wav`. Higher = more acoustically consistent.

        Returns a float in (-inf, 0]; callers typically squash to [0,1] before
        comparing to tau_update.
        """
        if not text.strip():
            return -20.0

        if self.stub:
            # deterministic-ish: longer/stranger text gets a worse score
            base = -0.25 - 0.002 * len(text)
            return float(base)

        wav_np = wav.detach().cpu().numpy().astype(np.float32) if isinstance(wav, torch.Tensor) else np.asarray(wav, dtype=np.float32)
        backend = self.cfg.get("backend", "faster-whisper")
        if backend == "openai-whisper":
            return self._openai_rescore(wav_np, text, sr=sr)

        inputs = self._hf_processor(wav_np, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        tokenizer = self._hf_processor.tokenizer
        forced = self._hf_model.generation_config.forced_decoder_ids or []
        prefix_ids = [tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
        for _, tok_id in forced:
            if tok_id is not None:
                prefix_ids.append(int(tok_id))
        text_ids = tokenizer(text, add_special_tokens=False).input_ids
        eos_id = self._hf_model.config.eos_token_id
        all_ids = prefix_ids + text_ids + [eos_id]
        vocab_size = int(self._hf_model.config.vocab_size)
        if any(tok_id is None or int(tok_id) < 0 or int(tok_id) >= vocab_size for tok_id in all_ids):
            return -20.0

        max_positions = int(getattr(self._hf_model.config, "max_target_positions", 448))
        if len(all_ids) > max_positions:
            # Whisper's decoder position embeddings are bounded (448 for large-v3).
            # Long GER prompt echoes or long ASR turns should be rejected by the
            # confidence gate, not crash CUDA with an out-of-bounds index.
            return -20.0

        decoder_input_ids = torch.tensor([all_ids], device=self.device)

        out = self._hf_model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = out.logits[0, :-1, :]
        targets = decoder_input_ids[0, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        start = len(prefix_ids)
        tok_lp = logp[start - 1: -1].gather(-1, targets[start - 1: -1].unsqueeze(-1)).squeeze(-1)
        if tok_lp.numel() == 0:
            return 0.0
        return float(tok_lp.mean().item())

    @torch.no_grad()
    def _openai_encoder_features(self, wav_np: np.ndarray) -> torch.Tensor:
        import whisper

        audio = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).to(self.device)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=getattr(self._ow_model.dims, "n_mels", 80),
        ).to(self.device)
        if self.device.type == "cuda":
            mel = mel.to(next(self._ow_model.parameters()).dtype)
        return self._ow_model.encoder(mel.unsqueeze(0)).squeeze(0).detach()

    @torch.no_grad()
    def _openai_rescore(self, wav_np: np.ndarray, text: str, sr: int = 16000) -> float:
        if self._ow_tokenizer is None:
            return -20.0
        if sr != 16000:
            try:
                import librosa
                wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
            except Exception:
                return -20.0

        enc_feats = self._openai_encoder_features(wav_np).unsqueeze(0)
        sot = list(getattr(self._ow_tokenizer, "sot_sequence", ()))
        text_ids = list(self._ow_tokenizer.encode(text))
        eot = int(self._ow_tokenizer.eot)
        all_ids = sot + text_ids + [eot]
        if len(all_ids) < 2:
            return -20.0

        max_positions = int(getattr(self._ow_model.dims, "n_text_ctx", 448))
        if len(all_ids) > max_positions:
            return -20.0

        tokens = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        logits = self._ow_model.decoder(tokens[:, :-1], enc_feats)
        targets = tokens[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        prefix_len = max(1, len(sot))
        target_start = max(0, prefix_len - 1)
        tok_lp = logp[:, target_start:, :].gather(
            -1,
            targets[:, target_start:].unsqueeze(-1),
        ).squeeze(-1)
        if tok_lp.numel() == 0:
            return 0.0
        return float(tok_lp.mean().item())

    @staticmethod
    def _uniform_words(text: str, duration_s: float) -> list[WordTiming]:
        toks = text.split()
        if not toks:
            return []
        dur = max(float(duration_s), 0.01) / max(1, len(toks))
        return [
            WordTiming(word=t, start=i * dur, end=(i + 1) * dur)
            for i, t in enumerate(toks)
        ]

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
