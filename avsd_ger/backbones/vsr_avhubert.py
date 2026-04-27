"""AV-HuBERT Large VSR wrapper.

Loads the facebookresearch/av_hubert Large checkpoint pretrained on
LRS3+VoxCeleb2 (1,759 h). We read features from a specified transformer
layer for C2's cross-modal alignment; optionally decode a lip-hypothesis
string to feed into the GER prompt.

This module stays frozen — C2 learns the alignment, not AV-HuBERT itself.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class AVHubertVSR(nn.Module):
    """
    Expected pretrained layout (see av_hubert README):
      checkpoints/avhubert_large_lrs3_iter5.pt
      av_hubert/avhubert/conf/pretrain/large_vox_iter5.yaml
    """

    FEATURE_DIM = 1024  # AV-HuBERT Large d_model

    def __init__(self, cfg: dict[str, Any], stub: bool = False, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.stub = stub
        self.device = torch.device(device)
        self.layer = int(cfg.get("layer", -1))
        self.emit_text = bool(cfg.get("emit_text", True))

        self._model = None
        self._task = None
        self._generator = None  # for lip-hypothesis decoding

        if not stub:
            self._load_real()

    # ------------------------------------------------------------------ loader
    def _load_real(self) -> None:
        # AV-HuBERT uses fairseq's checkpoint loader. We import lazily to keep
        # the skeleton importable without fairseq.
        try:
            import fairseq
            from fairseq import checkpoint_utils
        except ImportError as e:
            raise RuntimeError(
                "fairseq + av_hubert are required for real VSR. "
                "Install fairseq from source and clone facebookresearch/av_hubert."
            ) from e

        # AV-HuBERT defines its `av_hubert_pretraining` task via a
        # @register_task decorator inside avhubert/hubert_pretraining.py.
        # That decorator only fires when the module is actually imported, so we
        # MUST import hubert_pretraining (not just avhubert.hubert) here,
        # otherwise fairseq's checkpoint loader will fail with
        #   AssertionError: Could not infer task type from {'_name':'av_hubert_pretraining',...}
        # Errors are deliberately NOT swallowed — if PYTHONPATH isn't set
        # correctly (both `av_hubert/` and `av_hubert/avhubert/` must be on it),
        # we want a loud failure here, not a confusing one downstream.
        import importlib
        for mod in ("avhubert", "avhubert.hubert", "avhubert.hubert_pretraining"):
            try:
                importlib.import_module(mod)
            except ImportError as e:
                raise RuntimeError(
                    f"Cannot import {mod!r} — required for AV-HuBERT task registration. "
                    f"Make sure both `av_hubert/` AND `av_hubert/avhubert/` are on PYTHONPATH. "
                    f"Original error: {e}"
                ) from e

        ckpt = self.cfg["checkpoint"]

        # PyTorch ≥ 2.6 changed torch.load's default to weights_only=True, which
        # rejects fairseq checkpoints that pickle custom objects (e.g.
        # fairseq.data.dictionary.Dictionary).  Patch torch.load for this call only.
        _orig_load = torch.load
        def _load_legacy(f, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(f, *args, **kwargs)
        torch.load = _load_legacy
        try:
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt])
        finally:
            torch.load = _orig_load  # always restore, even on error
        self._model = models[0].to(self.device).eval()
        self._task = task
        if self.emit_text:
            # Sequence generator for VSR decoding — uses AV-HuBERT's own head.
            # build_generator requires target_dictionary, which only ships
            # with FINE-TUNED VSR checkpoints (e.g. self_large_vox_433h.pt).
            # The pretraining-only ckpt (avhubert_large_lrs3_iter5.pt) has
            # no decoder/dict, so build_generator raises here. Catch silently
            # and auto-flip self.emit_text=False so:
            #   * downstream code (extract()) skips the decode call
            #   * we don't spam a warning every run with the pretraining ckpt
            # Continuous-feature path (`<AV_CTX>` soft prefix to GER) is
            # unaffected either way.
            try:
                self._generator = task.build_generator([self._model], saved_cfg)
                import logging as _log
                _log.getLogger(__name__).info(
                    "VSR: lip-hypothesis generator built — lip_hyp will contain real text."
                )
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).info(
                    f"VSR: lip-hypothesis decoder unavailable ({type(_e).__name__}); "
                    "auto-disabling emit_text. To enable real lip_hyp text output, "
                    "use a fine-tuned VSR checkpoint (e.g. self_large_vox_433h.pt) — "
                    "see configs/default.yaml comments. Feature extraction is unaffected."
                )
                self._generator = None
                self.emit_text = False

    # ------------------------------------------------------------------ extract
    @torch.no_grad()
    def extract(self, video_frames: torch.Tensor) -> dict[str, Any]:
        """Run AV-HuBERT feature extraction on a mouth-ROI clip.

        Args:
            video_frames: [T, 1, 96, 96] grayscale mouth ROI, float in [0, 1],
                          25 fps. Caller runs the face/lip preprocessing.
        Returns:
            dict with keys: vsr_features ([T_v, 1024]), lip_hyp (str).
        """
        if self.stub:
            return self._stub_extract(video_frames)

        # video_frames: [T, 1, 96, 96]  ->  unsqueeze(0)  ->  [1, T, 1, 96, 96]
        # AV-HuBERT's Conv3d frontend expects [B, C, T, H, W], not [B, T, C, H, W],
        # so we swap dims 1 and 2 to land at [1, 1, T, 96, 96].
        # Without this swap Conv3d sees 75 channels instead of 1 and crashes.
        x = video_frames.unsqueeze(0).transpose(1, 2).to(self.device)  # [1, 1, T, 96, 96]
        T_v = x.shape[2]   # time dim is at position 2 after the swap

        # AV-HuBERT pretraining ckpt is multimodal (modality_fuse='concat') and
        # its forward unconditionally calls forward_features(src_audio); passing
        # audio=None crashes inside the audio extractor with
        #   AttributeError: 'NoneType' object has no attribute 'transpose'.
        # For VSR-only inference we pass a zero audio tensor of the expected
        # shape [B, audio_feat_dim, T] (audio_feat_dim=104 = 26 log-mel bins x
        # stack_order_audio=4, sampled at 25 Hz to match the video frame rate).
        # The audio fusion contribution becomes zero — VSR features are then
        # driven by the video stream alone. Slightly less informative than
        # full AV fusion but lets the pipeline run end-to-end.
        # TODO: extract real log-mel via python_speech_features (already in env)
        # and pass it here for true AV fusion.
        AUDIO_FEAT_DIM = 104   # AV-HuBERT pretraining: 26 mel * stack_order=4
        audio_zeros = torch.zeros(1, AUDIO_FEAT_DIM, T_v, device=self.device, dtype=x.dtype)

        feats, _ = self._model.extract_features(
            source={"video": x, "audio": audio_zeros},
            padding_mask=None,
            mask=False,
            output_layer=self.layer,
        )
        vsr_feats = feats.squeeze(0).detach()  # [T_v, 1024]

        lip_hyp = ""
        if self.emit_text and self._generator is not None:
            lip_hyp = self._decode_lip(x)

        return {"vsr_features": vsr_feats, "lip_hyp": lip_hyp}

    def _decode_lip(self, x: torch.Tensor) -> str:
        """Decode video frames into a lip-hypothesis text string.

        Requires a fine-tuned AV-HuBERT VSR checkpoint with a populated
        `task.target_dictionary` (e.g. self_large_vox_433h.pt). With the
        pretraining-only ckpt this method is never reached because
        emit_text is auto-flipped to False at load time.

        Returns "" on any decoding failure rather than crashing the
        pipeline -- lip_hyp is an auxiliary text channel for the GER prompt;
        an empty string just makes the LLM rely on audio-hyp + AV_CTX.
        """
        if self._generator is None or self._task is None:
            return ""
        try:
            T_v = x.shape[2]
            AUDIO_FEAT_DIM = 104
            audio_zeros = torch.zeros(
                1, AUDIO_FEAT_DIM, T_v, device=self.device, dtype=x.dtype
            )
            sample = {
                "net_input": {
                    "source": {"video": x, "audio": audio_zeros},
                    "padding_mask": None,
                }
            }
            hypos = self._generator.generate([self._model], sample)
            if not hypos or not hypos[0]:
                return ""
            tokens = hypos[0][0]["tokens"]
            tgt_dict = getattr(self._task, "target_dictionary", None)
            if tgt_dict is None:
                return ""
            try:
                text = tgt_dict.string(tokens.int().cpu(), bpe_symbol="subword_nmt")
            except (TypeError, AttributeError):
                text = tgt_dict.string(tokens.int().cpu())
            return text.strip()
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).debug(f"_decode_lip failed: {_e}")
            return ""

    # ------------------------------------------------------------------ stub
    def _stub_extract(self, video_frames) -> dict[str, Any]:
        T_v = 75  # ~3 s at 25 fps
        feats = torch.randn(T_v, self.FEATURE_DIM, device=self.device)
        lip_hyp = "the quick brown fox jumps over the lazy dog"  # dummy
        return {"vsr_features": feats, "lip_hyp": lip_hyp}
