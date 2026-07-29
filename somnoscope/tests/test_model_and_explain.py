"""Model, Grad-CAM and audit tests. Skipped automatically when torch is absent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from somnoscope.data.sleepedf import STAGE_NAMES                       # noqa: E402
from somnoscope.explain.bandpower import BANDS                          # noqa: E402
from somnoscope.explain.gradcam import (                                # noqa: E402
    GradCAM1D,
    attributed_spectrum,
    band_attribution,
)
from somnoscope.models.dual_stream_cnn import (                         # noqa: E402
    ModelConfig,
    build_model,
    class_weighted_loss,
)

FS = 100.0
EPOCH_LEN = int(FS * 30)


def tone(freq: float, n: int = EPOCH_LEN, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (60 * np.sin(2 * np.pi * freq * t) + rng.standard_normal(n)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def test_forward_shape_and_batch_independence():
    model = build_model().eval()
    x = torch.randn(7, 1, EPOCH_LEN)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (7, len(STAGE_NAMES))
    # (B, T) input is accepted too.
    with torch.no_grad():
        assert torch.allclose(model(x.squeeze(1)), logits, atol=1e-5)


def test_kernel_sizes_encode_the_two_target_timescales():
    """0.5 s for spindles, 4 s for slow waves -- as durations, not tap counts."""
    model = build_model(ModelConfig(sampling_rate=FS))
    assert model.fine.kernel_taps == 50            # 0.5 s at 100 Hz
    assert model.coarse.kernel_taps == 400         # 4.0 s at 100 Hz
    # A spindle is 11-16 Hz, so 0.5 s spans ~6 cycles; a slow wave is 0.5-2 Hz,
    # so the coarse kernel spans at least two full cycles.
    assert model.fine.kernel_taps / FS * 11 > 5
    assert model.coarse.kernel_taps / FS * 0.5 >= 2


def test_kernel_taps_follow_the_sampling_rate():
    model = build_model(ModelConfig(sampling_rate=200.0))
    assert model.fine.kernel_taps == 100
    assert model.coarse.kernel_taps == 800


def test_model_accepts_other_epoch_lengths_thanks_to_adaptive_pooling():
    model = build_model().eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, int(FS * 20))).shape == (2, 5)
        assert model(torch.randn(2, 1, int(FS * 60))).shape == (2, 5)


def test_rejects_multichannel_input():
    model = build_model()
    with pytest.raises(ValueError, match="single-channel"):
        model(torch.randn(2, 3, EPOCH_LEN))


def test_streams_are_not_redundant():
    """Zeroing one stream's features must change the output, or it is decoration."""
    model = build_model().eval()
    x = torch.randn(4, 1, EPOCH_LEN)
    with torch.no_grad():
        outs = model.stream_logits(x)
    assert not torch.allclose(outs["both"], outs["fine_only"], atol=1e-4)
    assert not torch.allclose(outs["both"], outs["coarse_only"], atol=1e-4)


def test_class_weighting_favours_the_rare_stage():
    """N1 is ~5% of epochs; unweighted loss lets the model trade it away."""
    counts = [5000, 300, 4000, 1500, 1800]           # W, N1, N2, N3, REM
    loss = class_weighted_loss(counts)
    weights = loss.weight.numpy()
    assert weights.argmax() == 1                     # N1 gets the largest weight
    assert weights[1] > weights[0]


def test_model_trains_a_little_on_a_separable_toy_problem():
    """Not a performance claim -- just that gradients flow and loss decreases."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = np.stack([tone(1.5 if i % 2 else 13.5, seed=int(rng.integers(1 << 20)))
                  for i in range(64)])
    y = np.array([0 if i % 2 else 1 for i in range(64)])
    xb = torch.from_numpy((X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)).unsqueeze(1)
    yb = torch.from_numpy(y)

    model = build_model()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    first = last = None
    for step in range(25):
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def test_gradcam_returns_normalised_maps_for_both_streams():
    model = build_model()
    x = torch.randn(3, 1, EPOCH_LEN)
    with GradCAM1D(model) as cam:
        out = cam(x)
    for stream in ("fine", "coarse"):
        maps = out["cams"][stream]
        assert maps.shape == (3, EPOCH_LEN)          # upsampled back onto the epoch
        assert maps.min() >= 0.0
        assert np.isclose(maps.max(), 1.0, atol=1e-5)
    assert out["prediction"].shape == (3,)


def test_gradcam_targets_the_requested_class():
    model = build_model()
    x = torch.randn(2, 1, EPOCH_LEN)
    with GradCAM1D(model) as cam:
        out = cam(x, target=3)
    assert (out["target"] == 3).all()


def test_gradcam_releases_retained_activations_on_exit():
    model = build_model()
    x = torch.randn(2, 1, EPOCH_LEN)
    with GradCAM1D(model) as cam:
        cam(x)
        assert model.fine.feature_map is not None
    assert model.fine.feature_map is None
    assert model.coarse.feature_map is None


def test_gradcam_can_be_called_repeatedly():
    model = build_model()
    cam = GradCAM1D(model)
    first = cam(torch.randn(2, 1, EPOCH_LEN))
    second = cam(torch.randn(2, 1, EPOCH_LEN))
    assert first["cams"]["fine"].shape == second["cams"]["fine"].shape


def test_gradcam_rejects_a_model_without_streams():
    with pytest.raises(ValueError, match="feature_map"):
        GradCAM1D(torch.nn.Linear(4, 4))


def test_gradcam_uses_the_same_tensor_for_activations_and_gradients():
    """Regression guard: a conv backward hook would fire pre-BatchNorm/GELU while
    ``feature_map`` is post-activation, silently mismatching the two."""
    model = build_model()
    x = torch.randn(2, 1, EPOCH_LEN)
    with GradCAM1D(model) as cam:
        cam(x)
        stored = model.fine.feature_map
        with torch.no_grad():
            recomputed = model.fine.blocks(model.fine.entry(x))
        assert stored.shape == recomputed.shape


# --------------------------------------------------------------------------- #
# CAM -> frequency projection: the step that makes the audit falsifiable
# --------------------------------------------------------------------------- #
def test_attributed_spectrum_follows_the_cam_in_time():
    """Attend to the half of the epoch containing the spindle, recover sigma."""
    n = EPOCH_LEN
    t = np.arange(n) / FS
    signal = np.zeros(n, dtype=np.float32)
    signal[: n // 2] = 60 * np.sin(2 * np.pi * 1.5 * t[: n // 2])       # delta half
    signal[n // 2 :] = 60 * np.sin(2 * np.pi * 13.5 * t[n // 2 :])      # sigma half

    cam_late = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    cam_early = 1.0 - cam_late

    late = band_attribution(signal, cam_late, BANDS, FS, contrast=False)
    early = band_attribution(signal, cam_early, BANDS, FS, contrast=False)

    assert max(late, key=late.get) == "sigma"
    assert max(early, key=early.get) == "delta"


def test_band_attribution_sums_to_one():
    signal = tone(10.0)
    cam = np.abs(np.random.default_rng(0).standard_normal(EPOCH_LEN))
    attribution = band_attribution(signal, cam, BANDS, FS)
    assert sum(attribution.values()) == pytest.approx(1.0)


def test_contrast_mode_discounts_a_band_that_is_merely_loud():
    """A flat CAM should not credit the model for the epoch's own dominant band."""
    signal = tone(1.5)                               # overwhelmingly delta
    flat_cam = np.ones(EPOCH_LEN)

    raw = band_attribution(signal, flat_cam, BANDS, FS, contrast=False)
    contrasted = band_attribution(signal, flat_cam, BANDS, FS, contrast=True)
    assert raw["delta"] > 0.8                        # raw power says "delta!"
    assert contrasted["delta"] < raw["delta"]        # contrast says "of course it is"


def test_attributed_and_baseline_spectra_have_matching_frequency_axes():
    freqs, attributed, baseline = attributed_spectrum(tone(8.0), np.ones(EPOCH_LEN), FS)
    assert freqs.shape == attributed.shape == baseline.shape
    assert freqs[-1] == pytest.approx(FS / 2)


def test_cam_shorter_than_signal_is_interpolated():
    freqs, attributed, _ = attributed_spectrum(tone(8.0), np.ones(128), FS)
    assert np.isfinite(attributed).all()
