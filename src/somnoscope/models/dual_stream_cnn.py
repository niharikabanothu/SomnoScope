"""Dual-stream 1-D CNN for single-channel EEG sleep staging.

Architecture rationale
----------------------
The two AASM stages that are hardest to separate are defined by events at
opposite ends of the time scale:

* **Sleep spindles** (N2) are 11-16 Hz bursts lasting **0.5-2 s**.
* **Slow-wave activity** (N3) is 0.5-2 Hz activity -- a single wave is **~1-2 s**,
  and the AASM rule is about the *proportion of a 30 s epoch* it occupies.

A single receptive field has to compromise between them. A short first-layer
kernel resolves spindle onsets but needs many layers before it can see a full
slow wave; a long kernel sees slow waves immediately but smears a spindle into
its own background. So the network runs both:

    stream         first kernel      covers        tuned for
    -------------  ----------------  ------------  ---------------------------
    fine           0.5 s (50 taps)   ~6 spindle    spindles, K-complex onsets,
                                     cycles        EMG-ish high frequency
    coarse         4.0 s (400 taps)  ~2-8 slow     slow-wave activity, delta
                                     waves         rhythm, REM theta

Both streams are convolutional stacks that end in an adaptive pool, so the epoch
length is not baked into the weights. Their pooled features are concatenated and
classified by a linear head.

This is the DeepSleepNet representation idea (Supratak et al., 2017) with the
kernel sizes stated as *durations* rather than tap counts, so changing the
sampling rate changes the taps and keeps the physiology fixed. The sequential
residual-BiLSTM stage of DeepSleepNet is deliberately **not** included: this repo
is an audit of what a single epoch supports, and a temporal model would let
context paper over exactly the per-epoch failures the robustness sweep is looking
for.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["StreamConfig", "ModelConfig", "ConvStream", "DualStreamCNN", "build_model"]


@dataclass
class StreamConfig:
    """One branch, with the first layer expressed in seconds rather than taps."""

    kernel_seconds: float
    stride_seconds: float
    n_filters: int = 64
    n_blocks: int = 3
    block_kernel: int = 7
    first_pool: int = 8
    block_pool: int = 4
    dropout: float = 0.3

    def taps(self, fs: float) -> Tuple[int, int]:
        kernel = max(int(round(self.kernel_seconds * fs)), 3)
        stride = max(int(round(self.stride_seconds * fs)), 1)
        return kernel, stride


@dataclass
class ModelConfig:
    sampling_rate: float = 100.0
    n_classes: int = 5
    pool_out: int = 4
    head_dropout: float = 0.5
    # 0.5 s kernel -> sleep spindles (11-16 Hz).
    fine: StreamConfig = None
    # 4 s kernel -> slow-wave activity (0.5-2 Hz).
    coarse: StreamConfig = None

    def __post_init__(self):
        if self.fine is None:
            self.fine = StreamConfig(
                kernel_seconds=0.5, stride_seconds=0.0625, first_pool=8, block_pool=4
            )
        if self.coarse is None:
            self.coarse = StreamConfig(
                kernel_seconds=4.0, stride_seconds=0.5, first_pool=4, block_pool=2
            )
        if isinstance(self.fine, dict):
            self.fine = StreamConfig(**self.fine)
        if isinstance(self.coarse, dict):
            self.coarse = StreamConfig(**self.coarse)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


class ConvStream(nn.Module):
    """Wide-kernel entry layer, then narrow-kernel blocks, then an adaptive pool.

    ``same``-style padding is used throughout the narrow blocks so the temporal
    axis only shrinks where a pool says it should. That keeps the Grad-CAM
    upsampling in ``somnoscope.explain`` a clean linear stretch back onto the raw
    30 s epoch, with no accumulated edge offset to correct for.
    """

    def __init__(self, cfg: StreamConfig, fs: float, pool_out: int):
        super().__init__()
        kernel, stride = cfg.taps(fs)
        self.kernel_taps, self.stride_taps = kernel, stride
        self.cfg = cfg

        self.entry = nn.Sequential(
            nn.Conv1d(1, cfg.n_filters, kernel_size=kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(cfg.n_filters),
            nn.GELU(),
            nn.MaxPool1d(cfg.first_pool, cfg.first_pool, ceil_mode=True),
            nn.Dropout(cfg.dropout),
        )

        blocks: List[nn.Module] = []
        channels = cfg.n_filters
        for i in range(cfg.n_blocks):
            out_channels = cfg.n_filters * 2 if i == cfg.n_blocks - 1 else cfg.n_filters
            blocks += [
                nn.Conv1d(channels, out_channels, kernel_size=cfg.block_kernel,
                          padding=cfg.block_kernel // 2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ]
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.tail = nn.Sequential(
            nn.MaxPool1d(cfg.block_pool, cfg.block_pool, ceil_mode=True),
            nn.Dropout(cfg.dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_out)
        self.out_channels = channels
        self.out_features = channels * pool_out

        # Grad-CAM hooks attach here: the last conv activation map, still on the
        # temporal axis and still linearly related to sample position.
        self.feature_map: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.entry(x)
        h = self.blocks(h)
        self.feature_map = h                 # (B, C, T') -- retained for explainability
        h = self.tail(h)
        return self.pool(h).flatten(1)

    def receptive_field_seconds(self, fs: float) -> float:
        """Approximate temporal support of one unit in the final conv map."""
        rf = self.kernel_taps
        jump = self.stride_taps
        rf += (self.cfg.first_pool - 1) * jump
        jump *= self.cfg.first_pool
        for _ in range(self.cfg.n_blocks):
            rf += (self.cfg.block_kernel - 1) * jump
        return rf / fs


class DualStreamCNN(nn.Module):
    """Two parallel :class:`ConvStream` branches with a concatenated linear head."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.fine = ConvStream(self.cfg.fine, self.cfg.sampling_rate, self.cfg.pool_out)
        self.coarse = ConvStream(self.cfg.coarse, self.cfg.sampling_rate, self.cfg.pool_out)

        n_features = self.fine.out_features + self.coarse.out_features
        self.head = nn.Sequential(
            nn.Dropout(self.cfg.head_dropout),
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(self.cfg.head_dropout * 0.5),
            nn.Linear(128, self.cfg.n_classes),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 1, T)`` or ``(B, T)`` -> logits ``(B, n_classes)``."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"expected (B, 1, T) single-channel input, got {tuple(x.shape)}")
        return self.head(torch.cat([self.fine(x), self.coarse(x)], dim=1))

    def stream_logits(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Ablate one stream at a time by zeroing its features.

        Used by ``somnoscope evaluate --stream-ablation`` to show that the two
        kernel scales are not redundant: N2 should lean on the fine stream and N3
        on the coarse one.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        fine, coarse = self.fine(x), self.coarse(x)
        zeros_f, zeros_c = torch.zeros_like(fine), torch.zeros_like(coarse)
        return {
            "both": self.head(torch.cat([fine, coarse], dim=1)),
            "fine_only": self.head(torch.cat([fine, zeros_c], dim=1)),
            "coarse_only": self.head(torch.cat([zeros_f, coarse], dim=1)),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x).argmax(dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        fs = self.cfg.sampling_rate
        return (
            f"DualStreamCNN  fs={fs:g} Hz  params={self.n_parameters():,}\n"
            f"  fine   : kernel {self.fine.kernel_taps:4d} taps "
            f"({self.cfg.fine.kernel_seconds:g} s)  RF ~{self.fine.receptive_field_seconds(fs):.1f} s"
            f"  -> spindles 11-16 Hz\n"
            f"  coarse : kernel {self.coarse.kernel_taps:4d} taps "
            f"({self.cfg.coarse.kernel_seconds:g} s)  RF ~{self.coarse.receptive_field_seconds(fs):.1f} s"
            f"  -> slow-wave activity 0.5-2 Hz"
        )


def build_model(cfg: ModelConfig | Dict | None = None) -> DualStreamCNN:
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return DualStreamCNN(cfg)


def class_weighted_loss(class_counts, device=None, beta: float = 0.999) -> nn.Module:
    """Effective-number class weighting (Cui et al., 2019).

    Sleep-EDF is imbalanced even after wake cropping -- N1 is typically ~5% of
    epochs and is the stage humans agree on least. Plain cross-entropy lets the
    model trade N1 away almost for free; kappa notices, accuracy does not.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    effective = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective
    weights = weights / weights.sum() * len(weights)
    if device is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)


def logits_to_proba(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)
