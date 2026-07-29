"""Spectral band power -- the clinical reference the model's attention is judged against.

The AASM scoring rules are, at bottom, statements about frequency content:

======  ==============  ===============================================================
stage   defining band   rule (abbreviated)
======  ==============  ===============================================================
W       alpha 8-12 Hz   posterior alpha on eye closure; high EMG, blinks
N1      theta 4-8 Hz    alpha attenuates, low-amplitude mixed-frequency theta appears
N2      sigma 11-16 Hz  sleep spindles (>=0.5 s) and/or K-complexes
N3      delta 0.5-2 Hz  slow-wave activity over >=20% of the epoch (>75 uV peak-to-peak)
REM     theta 4-8 Hz    low-amplitude mixed frequency, sawtooth waves, atonia
======  ==============  ===============================================================

That table is what makes a *spectral* explainability audit possible at all: unlike
most saliency work, there is an independent, pre-registered, clinically agreed
answer for what the model ought to be looking at. ``EXPECTED_BAND`` encodes it,
and ``somnoscope.explain.audit`` scores the model's Grad-CAM attribution against
it rather than eyeballing heatmaps.

W and REM share theta as a marker, which is honest: they are separated in the
clinic by EOG and chin EMG, neither of which a single-channel Fpz-Cz montage has.
The audit reports W and REM alignment but does not treat a miss there as strong
evidence of a broken model.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import signal as sps

__all__ = ["BANDS", "EXPECTED_BAND", "band_power", "relative_band_power",
           "band_power_by_stage", "spectrum"]

# numpy renamed trapz -> trapezoid in 2.0; support both.
_integrate = getattr(np, "trapezoid", None) or np.trapz

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),     # sleep spindles
    "beta": (16.0, 30.0),
}

# The band each stage is clinically defined by. This is the audit's ground truth
# and is fixed before any model is trained.
EXPECTED_BAND: Dict[str, str] = {
    "W": "alpha",
    "N1": "theta",
    "N2": "sigma",
    "N3": "delta",
    "REM": "theta",
}


def spectrum(
    x: np.ndarray, fs: float = 100.0, nperseg: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD. ``x`` may be ``(T,)`` or ``(B, T)``; returns ``(freqs, psd)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    nperseg = nperseg or min(int(fs * 4), x.shape[1])      # 4 s -> 0.25 Hz resolution
    freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    return freqs, psd


def band_power(
    x: np.ndarray, fs: float = 100.0, bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, np.ndarray]:
    """Absolute power per band, integrated over the PSD (uV^2)."""
    bands = bands or BANDS
    freqs, psd = spectrum(x, fs)
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = _integrate(psd[:, mask], freqs[mask], axis=-1) if mask.any() \
            else np.zeros(psd.shape[0])
    return out


def relative_band_power(
    x: np.ndarray, fs: float = 100.0, bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, np.ndarray]:
    """Band power normalised to sum to 1 per epoch.

    Relative power is the right quantity for this audit for the same reason
    amplitude scaling is a shortcut: the clinical rules are about the *shape* of
    the spectrum, not its overall scale.
    """
    absolute = band_power(x, fs, bands)
    total = np.sum(list(absolute.values()), axis=0) + 1e-12
    return {k: v / total for k, v in absolute.items()}


def band_power_by_stage(
    X: np.ndarray,
    y: Sequence[int],
    stage_names: Sequence[str],
    fs: float = 100.0,
) -> Dict[str, Dict[str, float]]:
    """Mean relative band power per stage -- the empirical version of the table above.

    Printed by the audit so the reader can confirm the *data* behaves as the AASM
    rules say before any claim is made about where the model looks.
    """
    y = np.asarray(y)
    out: Dict[str, Dict[str, float]] = {}
    for i, stage in enumerate(stage_names):
        mask = y == i
        if not mask.any():
            out[stage] = {b: float("nan") for b in BANDS}
            continue
        rel = relative_band_power(X[mask], fs)
        out[stage] = {band: float(np.mean(values)) for band, values in rel.items()}
    return out


def dominant_band(profile: Dict[str, float]) -> str:
    """Band carrying the most relative power in a profile."""
    finite = {k: v for k, v in profile.items() if np.isfinite(v)}
    return max(finite, key=finite.get) if finite else "none"
