"""1-D Grad-CAM for the dual-stream CNN, and the spectral projection that makes it testable.

Grad-CAM (Selvaraju et al., 2017) adapted to time series: take the final conv
feature map of a stream, weight each channel by the gradient of the target logit
averaged over time, sum, ReLU, and stretch back onto the 3000-sample epoch. That
gives a saliency curve over *time*.

Time is the wrong axis for auditing a sleep stager
--------------------------------------------------
A heatmap saying "the model looked at seconds 12-14" is not checkable against
anything. The AASM rules are frequency statements, not timestamps -- and a
saliency picture that a reader interprets by eye is exactly the kind of
explainability that has been shown to be unfalsifiable in practice.

So ``attributed_spectrum`` projects the CAM onto frequency: compute the STFT of
the epoch, weight each time frame by the CAM value there, and sum. The result is
"the spectrum of the parts of the signal the model actually used". That *is*
checkable -- against ``EXPECTED_BAND`` in ``bandpower.py``, one hypothesis per
stage, fixed in advance. The audit reports a hit rate, not a gallery.

Two streams, two answers
------------------------
CAMs are computed per stream, which is the point of the dual-scale design: the
fine (0.5 s) stream should attribute into sigma on N2 epochs, the coarse (4 s)
stream into delta on N3 epochs. If both streams attribute to the same band, the
second stream is not earning its parameters.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sps

__all__ = ["GradCAM1D", "attributed_spectrum", "band_attribution"]


class GradCAM1D:
    """Grad-CAM over the final conv feature map of each :class:`ConvStream`.

    Usage::

        with GradCAM1D(model) as cam:
            out = cam(x, target=None)       # target=None -> predicted class
        out["cams"]["fine"]                 # (B, T) in [0, 1]
        out["cams"]["coarse"]

    Implementation note: gradients come from ``torch.autograd.grad`` against the
    exact tensor each stream stored in ``feature_map``, not from a backward hook on
    the last ``Conv1d``. Those are not the same tensor -- the stored map is
    post-BatchNorm/GELU while a conv hook fires pre-activation -- and pairing
    mismatched activations with gradients is a quiet way to produce a saliency map
    that looks plausible and means nothing. Asking autograd directly makes the
    correspondence impossible to get wrong.
    """

    def __init__(self, model, streams: Sequence[str] = ("fine", "coarse")):
        self.model = model
        self.streams = list(streams)
        for name in self.streams:
            stream = getattr(model, name, None)
            if stream is None or not hasattr(stream, "feature_map"):
                raise ValueError(
                    f"model has no stream {name!r} exposing a feature_map; "
                    "GradCAM1D expects a DualStreamCNN"
                )

    def remove(self) -> None:
        """Drop references to the retained feature maps.

        Nothing is registered on the model, so there are no hooks to unregister --
        but the streams hold onto their last activation map, and on a large batch
        that is real memory. Kept as a method (and a context manager) so callers do
        not have to know which implementation is underneath.
        """
        for name in self.streams:
            stream = getattr(self.model, name, None)
            if stream is not None:
                stream.feature_map = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, target: torch.Tensor | int | None = None) -> Dict:
        """Return per-stream CAMs upsampled to the input length, plus predictions."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        with torch.enable_grad():
            logits = self.model(x)

            if target is None:
                target_idx = logits.argmax(dim=1)
            elif isinstance(target, int):
                target_idx = torch.full((x.size(0),), target, dtype=torch.long, device=x.device)
            else:
                target_idx = target.to(x.device).long()

            maps = []
            for name in self.streams:
                feature_map = getattr(self.model, name).feature_map
                if feature_map is None:
                    raise RuntimeError(f"stream {name!r} did not record a feature map")
                maps.append(feature_map)

            selected = logits.gather(1, target_idx[:, None]).sum()
            grads = torch.autograd.grad(selected, maps, retain_graph=False)

        cams = {}
        length = x.shape[-1]
        for name, activations, grad in zip(self.streams, maps, grads):
            weights = grad.mean(dim=2, keepdim=True)          # GAP over time -> (B, C, 1)
            cam = F.relu((weights * activations.detach()).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=length, mode="linear", align_corners=False)
            cam = cam.squeeze(1)
            peak = cam.amax(dim=1, keepdim=True).clamp_min(1e-8)
            cams[name] = (cam / peak).detach().cpu().numpy()

        return {
            "cams": cams,
            "logits": logits.detach().cpu().numpy(),
            "target": target_idx.detach().cpu().numpy(),
            "prediction": logits.argmax(1).detach().cpu().numpy(),
        }


def attributed_spectrum(
    signal: np.ndarray,
    cam: np.ndarray,
    fs: float = 100.0,
    nperseg: int | None = None,
) -> tuple:
    """Project a time-domain CAM onto frequency via a CAM-weighted STFT.

    Returns ``(freqs, attributed_psd, baseline_psd)``. ``attributed_psd`` weights
    each STFT frame by the mean CAM value over that frame, so it is the spectrum
    of the signal *as the model weighted it*; ``baseline_psd`` weights all frames
    equally. Comparing the two separates "the model looked at the delta band"
    from "this epoch happens to be mostly delta" -- which is the whole difficulty
    with reading saliency maps on a signal that is not spectrally flat.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    cam = np.asarray(cam, dtype=np.float64).reshape(-1)
    if len(cam) != len(signal):
        cam = np.interp(np.linspace(0, 1, len(signal)), np.linspace(0, 1, len(cam)), cam)

    nperseg = nperseg or min(int(fs * 2), len(signal))
    freqs, times, Zxx = sps.stft(
        signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, boundary=None, padded=False
    )
    power = np.abs(Zxx) ** 2                                    # (F, frames)

    frame_centres = np.clip((times * fs).astype(int), 0, len(cam) - 1)
    frame_weights = cam[frame_centres]
    frame_weights = frame_weights / (frame_weights.sum() + 1e-12)

    attributed = power @ frame_weights
    baseline = power.mean(axis=1)
    return freqs, attributed, baseline


def band_attribution(
    signal: np.ndarray,
    cam: np.ndarray,
    bands: Dict[str, tuple],
    fs: float = 100.0,
    contrast: bool = True,
) -> Dict[str, float]:
    """Fraction of the model's attention landing in each frequency band.

    With ``contrast=True`` the attributed spectrum is divided by the unweighted
    spectrum before integration, so the result answers "which bands did the model
    *over*-weight relative to what was there" rather than "which bands were loud".
    """
    freqs, attributed, baseline = attributed_spectrum(signal, cam, fs)
    spectrum_of_interest = attributed / (baseline + 1e-12) if contrast else attributed

    out, total = {}, 0.0
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        value = float(spectrum_of_interest[mask].sum()) if mask.any() else 0.0
        out[name] = value
        total += value
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out
