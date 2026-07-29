"""Data loading, epoching, splitting and degradation for Sleep-EDF."""

from .edf import Annotation, read_annotations, read_edf, read_edf_header, write_edf
from .sleepedf import (
    EPOCH_SECONDS,
    STAGE_NAMES,
    STAGE_TO_INDEX,
    Recording,
    class_distribution,
    crop_wake,
    epoch_signal,
    find_pairs,
    hypnogram_from_annotations,
    load_dataset,
    load_recording,
    make_synthetic_dataset,
    make_synthetic_recording,
    stack,
)
from .splits import subject_kfold, subject_split, verify_disjoint
from .transforms import DEGRADATIONS, apply_degradation, normalize

__all__ = [
    "Annotation",
    "read_edf",
    "read_edf_header",
    "read_annotations",
    "write_edf",
    "STAGE_NAMES",
    "STAGE_TO_INDEX",
    "EPOCH_SECONDS",
    "Recording",
    "load_recording",
    "load_dataset",
    "find_pairs",
    "crop_wake",
    "epoch_signal",
    "hypnogram_from_annotations",
    "class_distribution",
    "make_synthetic_recording",
    "make_synthetic_dataset",
    "stack",
    "subject_split",
    "subject_kfold",
    "verify_disjoint",
    "DEGRADATIONS",
    "apply_degradation",
    "normalize",
]
