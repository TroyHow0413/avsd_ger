"""Training-time modules (losses, schedules, cold-start enrolment).

These are intentionally kept out of the inference pipeline so that a
production deploy only needs to import `avsd_ger.pipeline` without
pulling in sklearn / accelerate / etc.
"""
from .identity_loss import BidirectionalInfoNCE, info_nce

__all__ = ["BidirectionalInfoNCE", "info_nce"]
