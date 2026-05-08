"""Frontend profiles for raw meeting-video AVSD experiments.

These profiles document the diarization / active-speaker frontends used to
produce the turn-level manifests consumed by the main C1 -> C2 -> C3 pipeline.
They do not import heavyweight third-party models; they are lightweight
experiment metadata and reporting helpers.
"""

from .registry import (
    FrontendProfile,
    get_frontend_profile,
    list_frontend_profiles,
    render_frontend_profiles_markdown,
)
from .mouth_roi import MouthROIExtractor

__all__ = [
    "FrontendProfile",
    "get_frontend_profile",
    "list_frontend_profiles",
    "render_frontend_profiles_markdown",
    "MouthROIExtractor",
]
