"""SomnoScope -- sleep staging from single-channel EEG, with an audit attached.

The package is organised so the parts that do not need a deep learning framework
do not import one:

    somnoscope.data        EDF reading, epoching, AASM labels, subject-disjoint
                           splits, the five wearable degradations  (numpy/scipy)
    somnoscope.metrics     Cohen's kappa against the human inter-scorer band
                           (numpy/sklearn)
    somnoscope.explain     band power eagerly; Grad-CAM lazily      (torch on use)
    somnoscope.models      the dual-stream CNN                      (torch)
    somnoscope.train       cross-validated training                 (torch)
    somnoscope.robustness  the degradation sweep + shortcut probe   (torch)

Importing ``somnoscope`` itself pulls in none of the torch-dependent modules.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .data.sleepedf import EPOCH_SECONDS, STAGE_NAMES, STAGE_TO_INDEX
from .metrics import HUMAN_KAPPA_CEILING, HUMAN_KAPPA_FLOOR

__all__ = [
    "__version__",
    "STAGE_NAMES",
    "STAGE_TO_INDEX",
    "EPOCH_SECONDS",
    "HUMAN_KAPPA_FLOOR",
    "HUMAN_KAPPA_CEILING",
]

_LAZY = {
    "DualStreamCNN": ".models.dual_stream_cnn",
    "build_model": ".models.dual_stream_cnn",
    "ModelConfig": ".models.dual_stream_cnn",
    "TrainConfig": ".train",
    "train_model": ".train",
    "cross_validate": ".train",
    "run_training": ".train",
    "robustness_sweep": ".robustness",
    "shortcut_index": ".robustness",
    "spectral_audit": ".explain.audit",
    "GradCAM1D": ".explain.gradcam",
    "load_checkpoint": ".evaluate",
}
__all__ += list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __package__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
