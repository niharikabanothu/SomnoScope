"""Explainability: Grad-CAM, band power, and the audit that pits one against the other.

``bandpower`` is pure numpy/scipy and imports eagerly. ``gradcam`` and ``audit``
need torch, so they are loaded on first access -- that way the band-power analysis
and the data pipeline stay usable in a torch-free environment.
"""

from .bandpower import (
    BANDS,
    EXPECTED_BAND,
    band_power,
    band_power_by_stage,
    dominant_band,
    relative_band_power,
    spectrum,
)

__all__ = [
    "BANDS",
    "EXPECTED_BAND",
    "band_power",
    "relative_band_power",
    "band_power_by_stage",
    "dominant_band",
    "spectrum",
    "GradCAM1D",
    "attributed_spectrum",
    "band_attribution",
    "spectral_audit",
    "format_audit",
]

_LAZY = {
    "GradCAM1D": ".gradcam",
    "attributed_spectrum": ".gradcam",
    "band_attribution": ".gradcam",
    "spectral_audit": ".audit",
    "format_audit": ".audit",
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __package__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
