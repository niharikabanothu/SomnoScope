"""Degradations behave as advertised, normalisation closes the gain shortcut,
and kappa punishes the majority-class predictor that accuracy rewards."""

from __future__ import annotations

import numpy as np
import pytest

from somnoscope.data.sleepedf import STAGE_NAMES, STAGE_TO_INDEX
from somnoscope.data.transforms import DEGRADATIONS, apply_degradation, normalize
from somnoscope.explain.bandpower import BANDS, EXPECTED_BAND, band_power, relative_band_power
from somnoscope.metrics import HUMAN_KAPPA_FLOOR, kappa_report

FS = 100.0


def make_epoch(freq: float, n: int = 3000, amplitude: float = 50.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (amplitude * np.sin(2 * np.pi * freq * t) + rng.standard_normal(n)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Degradations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_every_degradation_preserves_shape_and_is_a_no_op_at_zero_severity(name):
    x = np.stack([make_epoch(10.0, seed=i) for i in range(4)])
    out = apply_degradation(x, name, severity=0.0, fs=FS, seed=0)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.allclose(out, x, atol=1e-3), f"{name} is not a no-op at severity 0"


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_every_degradation_is_reproducible_given_a_seed(name):
    x = np.stack([make_epoch(10.0, seed=i) for i in range(3)])
    a = apply_degradation(x, name, 0.7, fs=FS, seed=42)
    b = apply_degradation(x, name, 0.7, fs=FS, seed=42)
    assert np.array_equal(a, b)


def test_amplitude_scaling_changes_scale_but_not_spectral_shape():
    """The premise of the shortcut probe: gain is label-preserving."""
    x = make_epoch(13.5)[None, :]                       # a sigma-band epoch
    scaled = apply_degradation(x, "amplitude_scale", 1.0, fs=FS, seed=7)

    assert not np.allclose(scaled, x)                   # something did happen
    before = relative_band_power(x, FS)
    after = relative_band_power(scaled, FS)
    for band in BANDS:
        assert after[band] == pytest.approx(before[band], abs=1e-4), band


def test_per_epoch_normalisation_makes_gain_changes_invisible():
    """This is the mechanism the shortcut probe is designed to test."""
    x = make_epoch(13.5)[None, :]
    scaled = x * 3.7
    assert np.allclose(normalize(x, "per_epoch"), normalize(scaled, "per_epoch"), atol=1e-4)
    # ... and the vulnerable baseline really is vulnerable.
    assert not np.allclose(normalize(x, "global"), normalize(scaled, "global"), atol=1e-2)


def test_sensor_noise_hits_the_requested_snr():
    x = make_epoch(2.0, amplitude=100.0)[None, :]
    noisy = apply_degradation(x, "sensor_noise", 1.0, fs=FS, seed=0)   # target 0 dB
    signal_power = np.mean(x**2)
    noise_power = np.mean((noisy - x) ** 2)
    assert 10 * np.log10(signal_power / noise_power) == pytest.approx(0.0, abs=1.5)


def test_bandwidth_loss_removes_the_sigma_band_that_defines_n2():
    """Absolute sigma power, not relative: with an 8 Hz cutoff almost everything
    is attenuated, so the relative share can stay high while the band itself has
    been gutted. N2 is the stage that should suffer."""
    x = make_epoch(13.5, amplitude=80.0)[None, :]
    filtered = apply_degradation(x, "bandwidth_loss", 1.0, fs=FS, seed=0)   # 8 Hz cutoff
    before = band_power(x, FS)["sigma"][0]
    after = band_power(filtered, FS)["sigma"][0]
    assert after < before / 100


def test_motion_drift_contaminates_delta_which_is_why_it_threatens_n3():
    x = make_epoch(20.0, amplitude=40.0)[None, :]      # beta-dominated to start
    drifted = apply_degradation(x, "motion_drift", 1.0, fs=FS, seed=0)
    assert relative_band_power(drifted, FS)["delta"][0] > relative_band_power(x, FS)["delta"][0]


def test_unknown_degradation_is_rejected():
    with pytest.raises(KeyError):
        apply_degradation(np.zeros((1, 3000), np.float32), "cosmic_rays", 0.5)


# --------------------------------------------------------------------------- #
# Band power sanity — the clinical reference the audit relies on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage,freq", [("N3", 1.5), ("N1", 6.0), ("W", 10.0), ("N2", 13.5)])
def test_a_pure_tone_lands_in_the_band_the_stage_is_defined_by(stage, freq):
    x = make_epoch(freq, amplitude=80.0)[None, :]
    profile = relative_band_power(x, FS)
    dominant = max(profile, key=lambda b: profile[b][0])
    assert dominant == EXPECTED_BAND[stage]


# --------------------------------------------------------------------------- #
# Metrics — why kappa and not accuracy
# --------------------------------------------------------------------------- #
def test_majority_class_predictor_scores_high_accuracy_and_zero_kappa():
    """The exact failure wake cropping exists to prevent."""
    rng = np.random.default_rng(0)
    w = STAGE_TO_INDEX["W"]
    y_true = np.array([w] * 650 + list(rng.integers(1, 5, 350)))
    y_pred = np.full_like(y_true, w)

    report = kappa_report(y_true, y_pred)
    assert report.accuracy == pytest.approx(0.65, abs=0.01)
    assert report.kappa == pytest.approx(0.0, abs=1e-9)
    assert not report.reaches_human_floor
    assert any("constant predictor" in note for note in report.notes)


def test_perfect_prediction_reaches_kappa_one_and_clears_the_human_band():
    y = np.repeat(np.arange(5), 40)
    report = kappa_report(y, y.copy())
    assert report.kappa == pytest.approx(1.0)
    assert report.reaches_human_floor
    assert report.ceiling_fraction > 1.0


def test_report_shape_and_confidence_interval():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 5, 400)
    y_pred = np.where(rng.random(400) < 0.8, y_true, rng.integers(0, 5, 400))

    report = kappa_report(y_true, y_pred, bootstrap=200)
    assert report.confusion.shape == (5, 5)
    assert set(report.per_class_f1) == set(STAGE_NAMES)
    lo, hi = report.kappa_ci
    assert lo < report.kappa < hi
    assert 0.0 <= report.kappa <= 1.0


def test_human_band_constant_matches_the_literature_value_used_in_the_readme():
    assert HUMAN_KAPPA_FLOOR == 0.75


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        kappa_report(np.zeros(10, int), np.zeros(9, int))
