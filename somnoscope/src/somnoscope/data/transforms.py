"""Wearable-realistic signal degradations, plus the normalisation the model sees.

Clinical PSG is recorded with gelled AgAgCl electrodes on a shielded amplifier.
A headband or an in-ear device is not that. The five degradations below are the
failure modes that separate the two, each parameterised by a severity in
``[0, 1]`` so a model can be swept along a single axis:

===========================  ==================================================
``amplitude_scale``          Gain / impedance mismatch. **The shortcut probe.**
``sensor_noise``             Dry-electrode thermal + contact noise (SNR in dB).
``motion_drift``             Low-frequency baseline wander from head movement.
``powerline_hum``            50/60 Hz mains pickup through poor shielding.
``bandwidth_loss``           Cheap analogue front-end rolling off high frequency.
===========================  ==================================================

Why amplitude scaling is the interesting one
--------------------------------------------
Sleep stage is defined by the *relative* spectral content of the EEG, not its
absolute voltage -- a night recorded through a higher-impedance electrode is
still the same night. So a stage classifier ought to be near-invariant to a pure
gain change. If accuracy collapses when the signal is multiplied by 0.5, the
model has learned "big numbers mean N3" rather than "slow waves mean N3": an
amplitude shortcut that will not survive contact with a real wearable. That makes
this degradation a *probe* rather than just a stress test, and it is why the
default normalisation matters so much (see ``normalize``).
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy import signal as sps

__all__ = [
    "DEGRADATIONS",
    "amplitude_scale",
    "sensor_noise",
    "motion_drift",
    "powerline_hum",
    "bandwidth_loss",
    "apply_degradation",
    "normalize",
]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[None, :] if x.ndim == 1 else x


def _clean(x: np.ndarray, severity: float) -> bool:
    """Severity 0 means *clean* for every degradation, exactly.

    Worth stating in code rather than leaving to each formula: the severity-0 row
    of the robustness table has to be the untouched baseline, or the whole sweep
    is measured against the wrong reference. ``sensor_noise`` at 30 dB and
    ``bandwidth_loss`` at a 45 Hz cutoff are both nearly harmless but not
    identical to the input, and "nearly" is not good enough for a baseline.
    """
    return float(severity) <= 0.0


# --------------------------------------------------------------------------- #
# The five degradations
# --------------------------------------------------------------------------- #
def amplitude_scale(
    x: np.ndarray, severity: float = 0.5, fs: float = 100.0, rng=None
) -> np.ndarray:
    """Multiply by a per-epoch gain factor. Severity 0 -> 1.0x, severity 1 -> up to 4x/0.25x.

    The gain is drawn log-uniformly and symmetrically around unity so that
    attenuation and amplification are equally likely; a model that is genuinely
    scale-invariant should be unaffected at every severity.
    """
    x = _as_2d(x)
    if _clean(x, severity):
        return x
    rng = np.random.default_rng() if rng is None else rng
    max_log_gain = np.log(4.0) * float(severity)
    gains = np.exp(rng.uniform(-max_log_gain, max_log_gain, size=(x.shape[0], 1)))
    return (x * gains).astype(np.float32)


def sensor_noise(
    x: np.ndarray, severity: float = 0.5, fs: float = 100.0, rng=None
) -> np.ndarray:
    """Additive white noise at a target SNR (severity 0 -> 30 dB, severity 1 -> 0 dB)."""
    x = _as_2d(x)
    if _clean(x, severity):
        return x
    rng = np.random.default_rng() if rng is None else rng
    snr_db = 30.0 * (1.0 - float(severity))
    power = np.mean(x**2, axis=1, keepdims=True) + 1e-12
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def motion_drift(
    x: np.ndarray, severity: float = 0.5, fs: float = 100.0, rng=None
) -> np.ndarray:
    """Low-frequency baseline wander (0.05-0.5 Hz) at up to 100% of signal RMS.

    This is the one that contaminates the delta band, so it is the natural threat
    to N3 detection: motion wander and slow-wave activity live next door to each
    other in frequency.
    """
    x = _as_2d(x)
    if _clean(x, severity):
        return x
    rng = np.random.default_rng() if rng is None else rng
    n = x.shape[1]
    t = np.arange(n) / fs
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True)) + 1e-12

    drift = np.zeros_like(x)
    for f in (0.05, 0.13, 0.31, 0.5):
        phase = rng.uniform(0, 2 * np.pi, size=(x.shape[0], 1))
        weight = rng.uniform(0.5, 1.0, size=(x.shape[0], 1))
        drift += weight * np.sin(2 * np.pi * f * t[None, :] + phase)
    drift /= np.max(np.abs(drift), axis=1, keepdims=True) + 1e-12
    return (x + drift * rms * float(severity)).astype(np.float32)


def powerline_hum(
    x: np.ndarray, severity: float = 0.5, fs: float = 100.0, rng=None
) -> np.ndarray:
    """Mains interference at 50 Hz with a 100 Hz harmonic, up to 50% of signal RMS.

    At fs = 100 Hz the 50 Hz fundamental sits exactly at Nyquist, which is
    realistic for Sleep-EDF and deliberately awkward: it aliases into a DC-like
    alternating pattern rather than appearing as a clean spectral line.
    """
    x = _as_2d(x)
    if _clean(x, severity):
        return x
    rng = np.random.default_rng() if rng is None else rng
    n = x.shape[1]
    t = np.arange(n) / fs
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True)) + 1e-12
    phase = rng.uniform(0, 2 * np.pi, size=(x.shape[0], 1))
    hum = np.sin(2 * np.pi * 50.0 * t[None, :] + phase)
    hum += 0.3 * np.sin(2 * np.pi * 100.0 * t[None, :] + phase)
    return (x + 0.5 * float(severity) * rms * hum).astype(np.float32)


def bandwidth_loss(
    x: np.ndarray, severity: float = 0.5, fs: float = 100.0, rng=None
) -> np.ndarray:
    """Low-pass the signal: severity 0 -> 45 Hz cutoff, severity 1 -> 8 Hz cutoff.

    Models a cheap analogue front-end. At high severity this destroys the sigma
    band outright, so N2 (spindles) should degrade well before N3 (slow waves) --
    a directional prediction the audit checks rather than assumes.
    """
    x = _as_2d(x)
    if _clean(x, severity):
        return x
    nyquist = fs / 2.0
    cutoff = 45.0 - 37.0 * float(severity)
    cutoff = float(np.clip(cutoff, 1.0, nyquist * 0.98))
    if cutoff >= nyquist * 0.97:
        return x.astype(np.float32)
    sos = sps.butter(4, cutoff / nyquist, btype="low", output="sos")
    return sps.sosfiltfilt(sos, x, axis=1).astype(np.float32)


DEGRADATIONS: Dict[str, Callable[..., np.ndarray]] = {
    "amplitude_scale": amplitude_scale,
    "sensor_noise": sensor_noise,
    "motion_drift": motion_drift,
    "powerline_hum": powerline_hum,
    "bandwidth_loss": bandwidth_loss,
}


def apply_degradation(
    x: np.ndarray, name: str, severity: float, fs: float = 100.0, seed: int | None = None
) -> np.ndarray:
    """Dispatch by name with a reproducible RNG."""
    if name not in DEGRADATIONS:
        raise KeyError(f"unknown degradation {name!r}; have {sorted(DEGRADATIONS)}")
    rng = np.random.default_rng(seed)
    return DEGRADATIONS[name](x, severity=severity, fs=fs, rng=rng)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalize(x: np.ndarray, mode: str = "per_epoch", clip: float = 20.0) -> np.ndarray:
    """Scale epochs before they reach the model.

    ``per_epoch``
        z-score each 30 s epoch independently. Removes absolute gain, which is
        exactly what closes the amplitude shortcut -- at the cost of discarding
        genuine amplitude information (N3 really does have larger deflections
        than N1). This is the default, and the robustness table quantifies what
        it buys.
    ``global``
        divide by a fixed constant, preserving relative amplitude across epochs.
        Use ``--norm global`` to reproduce the vulnerable baseline.
    ``none``
        raw microvolts.

    ``clip`` bounds the result in units of sigma to stop electrode pops from
    dominating a batch.
    """
    x = _as_2d(x)
    if mode == "per_epoch":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        out = (x - mu) / sd
    elif mode == "global":
        out = x / 50.0                      # ~1 sigma of Sleep-EDF Fpz-Cz in uV
    elif mode == "none":
        out = x
    else:
        raise ValueError(f"unknown normalisation mode {mode!r}")
    if clip:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32)
