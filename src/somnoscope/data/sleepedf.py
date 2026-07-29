"""Sleep-EDF Expanded -> epoch tensors, with AASM 5-stage labels and wake cropping.

Two things in here carry most of the experimental weight:

1. **AASM 5-stage mapping.** Sleep-EDF was scored under the older R&K rules, which
   split deep sleep into S3 and S4. AASM merges them into N3, so the standard
   remap is ``{W, S1->N1, S2->N2, S3+S4->N3, R->REM}``. ``Movement time`` and
   ``Sleep stage ?`` are dropped rather than folded into Wake.

2. **Wake cropping.** The Sleep Cassette recordings are ~20 hours long and the
   subjects were ambulatory, so raw class balance is roughly 60-70% Wake. A model
   that predicts Wake unconditionally scores ~0.65 accuracy and kappa ~0.0 --
   accuracy stops measuring anything. Following the convention used by
   DeepSleepNet and most Sleep-EDF papers, we keep only ``crop_margin_min``
   minutes of Wake on either side of the first/last non-Wake epoch. This is a
   *labelling* decision made before any model sees the data, applied identically
   to train and test, and it is reported alongside every number this repo
   produces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .edf import Annotation, read_annotations, read_edf, write_edf

__all__ = [
    "STAGE_NAMES",
    "STAGE_TO_INDEX",
    "EPOCH_SECONDS",
    "Recording",
    "stage_from_annotation",
    "hypnogram_from_annotations",
    "crop_wake",
    "epoch_signal",
    "load_recording",
    "find_pairs",
    "load_dataset",
    "class_distribution",
    "make_synthetic_recording",
]

# AASM five-stage target space. Index order is fixed everywhere in the repo:
# confusion matrices, per-class F1 and the class-weight vector all assume it.
STAGE_NAMES: Tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
STAGE_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(STAGE_NAMES)}

EPOCH_SECONDS = 30.0

_STAGE_PATTERN = re.compile(r"sleep\s*stage\s*(\S+)", re.IGNORECASE)
_RK_TO_AASM = {
    "w": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",     # R&K S4 folds into AASM N3
    "r": "REM",
}


@dataclass
class Recording:
    """One night of EEG, already epoched and labelled."""

    subject: str                  # e.g. "SC40" -- the subject, not the night
    night: str                    # full record id, e.g. "SC4001E0"
    epochs: np.ndarray            # (n_epochs, n_samples) float32, microvolts
    labels: np.ndarray            # (n_epochs,) int64 in [0, 5)
    sampling_rate: float
    channel: str

    def __len__(self) -> int:
        return len(self.labels)


# --------------------------------------------------------------------------- #
# Hypnogram handling
# --------------------------------------------------------------------------- #
def stage_from_annotation(text: str) -> int | None:
    """Map one EDF+ annotation string to an AASM index, or ``None`` to drop it."""
    match = _STAGE_PATTERN.search(text)
    if not match:
        return None                       # "Movement time", "Lights off", ...
    token = match.group(1).strip().lower()
    stage = _RK_TO_AASM.get(token)
    if stage is None:
        return None                       # "Sleep stage ?"
    return STAGE_TO_INDEX[stage]


def hypnogram_from_annotations(
    annotations: Iterable[Annotation],
    n_epochs: int,
    epoch_seconds: float = EPOCH_SECONDS,
) -> np.ndarray:
    """Expand variable-duration stage annotations into a per-epoch label vector.

    Epochs with no annotation, or an unscorable one, get ``-1`` and are dropped
    downstream -- never silently relabelled as Wake.
    """
    hypnogram = np.full(n_epochs, -1, dtype=np.int64)
    for ann in annotations:
        stage = stage_from_annotation(ann.text)
        if stage is None:
            continue
        start = int(round(ann.onset / epoch_seconds))
        stop = int(round((ann.onset + ann.duration) / epoch_seconds))
        start = max(start, 0)
        stop = min(max(stop, start + 1), n_epochs)
        if start < n_epochs:
            hypnogram[start:stop] = stage
    return hypnogram


def crop_wake(labels: np.ndarray, crop_margin_min: float = 30.0) -> np.ndarray:
    """Boolean keep-mask retaining ``crop_margin_min`` of Wake around the sleep period.

    Returns an all-``True`` mask when the night contains no sleep at all, so the
    caller never ends up with an empty recording by surprise.
    """
    labels = np.asarray(labels)
    sleep = np.flatnonzero((labels >= 0) & (labels != STAGE_TO_INDEX["W"]))
    keep = np.ones(len(labels), dtype=bool)
    if sleep.size == 0:
        return keep
    margin = int(round(crop_margin_min * 60.0 / EPOCH_SECONDS))
    lo = max(int(sleep[0]) - margin, 0)
    hi = min(int(sleep[-1]) + margin + 1, len(labels))
    keep[:lo] = False
    keep[hi:] = False
    return keep


# --------------------------------------------------------------------------- #
# Signal handling
# --------------------------------------------------------------------------- #
def epoch_signal(
    signal: np.ndarray,
    sampling_rate: float,
    epoch_seconds: float = EPOCH_SECONDS,
) -> np.ndarray:
    """Reshape a continuous signal into non-overlapping epochs, dropping the tail."""
    per_epoch = int(round(sampling_rate * epoch_seconds))
    if per_epoch <= 0:
        raise ValueError("sampling_rate * epoch_seconds must be positive")
    n_epochs = len(signal) // per_epoch
    if n_epochs == 0:
        raise ValueError("signal shorter than a single epoch")
    return np.asarray(signal[: n_epochs * per_epoch], dtype=np.float32).reshape(
        n_epochs, per_epoch
    )


def subject_id(record_name: str) -> str:
    """``SC4001E0`` -> ``SC400``: strip the night index so nights of one subject group."""
    stem = Path(record_name).stem.split("-")[0]
    return stem[:5] if len(stem) > 5 else stem


def load_recording(
    psg_path: str | Path,
    hypnogram_path: str | Path,
    channel: str = "Fpz-Cz",
    crop_margin_min: float = 30.0,
    epoch_seconds: float = EPOCH_SECONDS,
) -> Recording:
    """Load one PSG/hypnogram pair into epochs + AASM labels."""
    psg_path, hypnogram_path = Path(psg_path), Path(hypnogram_path)
    signals, rates, _ = read_edf(psg_path, channels=[channel])
    label = next(iter(signals))
    fs = rates[label]

    epochs = epoch_signal(signals[label], fs, epoch_seconds)
    hypnogram = hypnogram_from_annotations(
        read_annotations(hypnogram_path), len(epochs), epoch_seconds
    )

    keep = hypnogram >= 0
    if crop_margin_min is not None:
        keep &= crop_wake(hypnogram, crop_margin_min)

    return Recording(
        subject=subject_id(psg_path.name),
        night=Path(psg_path.name).stem.split("-")[0],
        epochs=epochs[keep],
        labels=hypnogram[keep],
        sampling_rate=fs,
        channel=label,
    )


def find_pairs(root: str | Path) -> List[Tuple[Path, Path]]:
    """Pair ``*-PSG.edf`` with its ``*-Hypnogram.edf`` inside ``root``.

    Sleep-EDF gives the two files different trailing characters
    (``SC4001E0-PSG`` vs ``SC4001EC-Hypnogram``), so matching is done on the
    first seven characters of the record id.
    """
    root = Path(root)
    psgs = sorted(root.rglob("*-PSG.edf"))
    hyps = {p.name[:7]: p for p in sorted(root.rglob("*-Hypnogram.edf"))}
    pairs = []
    for psg in psgs:
        hyp = hyps.get(psg.name[:7])
        if hyp is not None:
            pairs.append((psg, hyp))
    return pairs


def load_dataset(
    root: str | Path,
    channel: str = "Fpz-Cz",
    crop_margin_min: float = 30.0,
    limit: int | None = None,
    verbose: bool = True,
) -> List[Recording]:
    """Load every PSG/hypnogram pair under ``root``."""
    pairs = find_pairs(root)
    if not pairs:
        raise FileNotFoundError(
            f"no *-PSG.edf / *-Hypnogram.edf pairs under {root}. "
            "Run `bash scripts/download_sleepedf.sh` first (see docs/DATA.md)."
        )
    if limit:
        pairs = pairs[:limit]

    recordings = []
    for psg, hyp in pairs:
        try:
            rec = load_recording(psg, hyp, channel, crop_margin_min)
        except (ValueError, KeyError) as exc:
            if verbose:
                print(f"  skipping {psg.name}: {exc}")
            continue
        if len(rec) == 0:
            continue
        recordings.append(rec)
        if verbose:
            print(f"  {rec.night}  subject={rec.subject}  epochs={len(rec):5d}")
    return recordings


def class_distribution(labels: np.ndarray) -> Dict[str, float]:
    """Fraction of epochs per stage -- used to report the wake-cropping effect."""
    labels = np.asarray(labels)
    total = max(len(labels), 1)
    counts = np.bincount(labels[labels >= 0], minlength=len(STAGE_NAMES))
    return {name: float(counts[i]) / total for i, name in enumerate(STAGE_NAMES)}


# --------------------------------------------------------------------------- #
# Synthetic data — lets the whole pipeline run end to end without a download
# --------------------------------------------------------------------------- #
_STAGE_SPECTRA = {
    #        delta  theta  alpha  sigma  beta   noise
    "W":   (0.20, 0.30, 1.00, 0.10, 0.70, 1.00),
    "N1":  (0.40, 1.00, 0.35, 0.10, 0.25, 0.90),
    "N2":  (0.70, 0.60, 0.15, 1.00, 0.15, 0.80),   # sigma == spindles
    "N3":  (1.00, 0.35, 0.10, 0.15, 0.08, 0.70),   # delta == slow waves
    "REM": (0.30, 0.90, 0.25, 0.08, 0.35, 0.95),
}
_BAND_CENTRES = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "sigma": 13.5, "beta": 22.0}


def make_synthetic_recording(
    out_dir: str | Path,
    record: str = "SC4001E0",
    n_epochs: int = 200,
    sampling_rate: float = 100.0,
    seed: int = 0,
    wake_pad_epochs: int = 40,
) -> Tuple[Path, Path]:
    """Write a synthetic PSG + hypnogram pair with stage-appropriate spectra.

    Not a substitute for Sleep-EDF -- it exists so that ``somnoscope selftest``
    can exercise loading, training, robustness and explainability in ~30 seconds
    on a machine with no data on it. Each stage gets the band-power profile that
    stage is clinically defined by, so a working model should reach a high kappa
    here and the spectral audit should point at the right bands.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    per_epoch = int(sampling_rate * EPOCH_SECONDS)
    t = np.arange(per_epoch) / sampling_rate

    # A plausible night: wake padding, then cycles through the sleep stages.
    cycle = ["N1"] * 4 + ["N2"] * 12 + ["N3"] * 8 + ["N2"] * 6 + ["REM"] * 6
    body = (cycle * (n_epochs // len(cycle) + 1))[: max(n_epochs - 2 * wake_pad_epochs, 1)]
    stages = ["W"] * wake_pad_epochs + body + ["W"] * wake_pad_epochs

    chunks, annotations = [], []
    for i, stage in enumerate(stages):
        d, th, a, s, b, n = _STAGE_SPECTRA[stage]
        amps = dict(zip(_BAND_CENTRES, (d, th, a, s, b)))
        sig = rng.standard_normal(per_epoch) * 8.0 * n
        for band, centre in _BAND_CENTRES.items():
            f = centre * (1 + 0.08 * rng.standard_normal())
            sig += amps[band] * 25.0 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        if stage == "N2" and i % 3 == 0:                     # discrete spindle bursts
            onset = rng.integers(0, per_epoch - int(sampling_rate))
            window = np.zeros(per_epoch)
            span = slice(onset, onset + int(sampling_rate))
            window[span] = np.hanning(int(sampling_rate))
            sig += 45.0 * window * np.sin(2 * np.pi * 13.5 * t)
        chunks.append(sig.astype(np.float32))
        annotations.append(
            Annotation(i * EPOCH_SECONDS, EPOCH_SECONDS, f"Sleep stage {_aasm_to_edf(stage)}")
        )

    psg = out_dir / f"{record}-PSG.edf"
    hyp = out_dir / f"{record[:7]}C-Hypnogram.edf"
    write_edf(psg, {"EEG Fpz-Cz": np.concatenate(chunks)}, sampling_rate, record_duration=1.0)
    write_edf(
        hyp,
        {"EEG Fpz-Cz": np.zeros(len(stages) * per_epoch, dtype=np.float32)},
        sampling_rate,
        record_duration=EPOCH_SECONDS,
        annotations=annotations,
    )
    return psg, hyp


def _aasm_to_edf(stage: str) -> str:
    return {"W": "W", "N1": "1", "N2": "2", "N3": "4", "REM": "R"}[stage]


def make_synthetic_dataset(
    out_dir: str | Path, n_subjects: int = 6, n_epochs: int = 200, seed: int = 0
) -> List[Tuple[Path, Path]]:
    """Several synthetic subjects, so subject-disjoint splitting has something to split."""
    return [
        make_synthetic_recording(
            out_dir, record=f"SC4{i:02d}1E0", n_epochs=n_epochs, seed=seed + i
        )
        for i in range(n_subjects)
    ]


def stack(recordings: Sequence[Recording]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate recordings into ``(X, y, subject_ids)`` arrays."""
    if not recordings:
        raise ValueError("no recordings to stack")
    X = np.concatenate([r.epochs for r in recordings], axis=0)
    y = np.concatenate([r.labels for r in recordings], axis=0)
    groups = np.concatenate([np.repeat(r.subject, len(r)) for r in recordings])
    return X, y, groups
