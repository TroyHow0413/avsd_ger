"""AVSD-GER: identity-conditioned generative error correction for audio-visual speech.

Modules:
    backbones/   — frozen ASR (Whisper-large-v3) + VSR (AV-HuBERT Large)
    c1_identity/ — enrolled speaker pool (ECAPA + ArcFace)
    c2_alignment/— ID-conditioned cross-modal alignment + LLM GER head
    c3_feedback/ — composite confidence + closed-loop retry
    pipeline     — orchestrator gluing C1 → C2 → C3
"""

__version__ = "0.1.0"
