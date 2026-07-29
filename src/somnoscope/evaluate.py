"""Load a checkpoint and score it: metrics, stream ablation, optional plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from .data.sleepedf import STAGE_NAMES, load_dataset, stack
from .data.transforms import normalize
from .metrics import format_report, kappa_report
from .models.dual_stream_cnn import ModelConfig, build_model

__all__ = ["load_checkpoint", "evaluate_arrays", "stream_ablation", "plot_confusion"]


def load_checkpoint(path: str | Path, device=None):
    """Restore a model saved by ``train.cross_validate``."""
    path = Path(path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_cfg"]) if isinstance(blob.get("model_cfg"), dict) \
        else blob.get("model_cfg") or ModelConfig()
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, {"norm": blob.get("norm", "per_epoch"), "fs": blob.get("fs", 100.0),
                   "kappa": blob.get("kappa"), "device": device}


@torch.no_grad()
def _logits(model, X: np.ndarray, norm: str, device, batch_size: int = 256) -> np.ndarray:
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(normalize(X[start : start + batch_size], norm)).unsqueeze(1)
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def evaluate_arrays(
    model, X: np.ndarray, y: np.ndarray, norm: str = "per_epoch", device=None,
    bootstrap: int = 500, title: str = "Held-out evaluation", verbose: bool = True,
):
    device = device or next(model.parameters()).device
    y_pred = _logits(model, X, norm, device).argmax(1)
    report = kappa_report(y, y_pred, bootstrap=bootstrap)
    if verbose:
        print(format_report(report, title))
    return report


@torch.no_grad()
def stream_ablation(
    model, X: np.ndarray, y: np.ndarray, norm: str = "per_epoch", device=None,
    batch_size: int = 256, verbose: bool = True,
) -> Dict:
    """Score the model with each stream zeroed, to test that both scales earn their keep.

    Prediction: N2 (spindles, 0.5-2 s events) should lose more F1 when the fine
    stream is ablated, and N3 (slow-wave activity, 0.5-2 Hz) more when the coarse
    stream is. If ablating either stream costs the same everywhere, the second
    scale is redundant and the architecture is not doing what it claims.
    """
    device = device or next(model.parameters()).device
    model.eval()
    collected: Dict[str, list] = {"both": [], "fine_only": [], "coarse_only": []}
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(normalize(X[start : start + batch_size], norm)).unsqueeze(1)
        outs = model.stream_logits(xb.to(device))
        for key, logits in outs.items():
            collected[key].append(logits.argmax(1).cpu().numpy())

    reports = {k: kappa_report(y, np.concatenate(v)) for k, v in collected.items()}
    result = {
        k: {"kappa": r.kappa, "accuracy": r.accuracy, "per_class_f1": r.per_class_f1}
        for k, r in reports.items()
    }
    base = reports["both"]
    result["delta_f1_vs_both"] = {
        k: {s: reports[k].per_class_f1[s] - base.per_class_f1[s] for s in STAGE_NAMES}
        for k in ("fine_only", "coarse_only")
    }
    if verbose:
        print("\n=== Stream ablation ".ljust(72, "="))
        print("  " + "config".ljust(14) + "kappa".rjust(8)
              + "".join(f"{s:>8}" for s in STAGE_NAMES) + "   (per-stage F1)")
        for key in ("both", "fine_only", "coarse_only"):
            r = reports[key]
            cells = "".join(f"{r.per_class_f1[s]:8.3f}" for s in STAGE_NAMES)
            print(f"  {key:<14}{r.kappa:8.3f}{cells}")
        n2 = result["delta_f1_vs_both"]
        print(f"\n  N2 F1 without the fine stream   {n2['coarse_only']['N2']:+.3f}"
              "   (spindles: expect a large drop)")
        print(f"  N3 F1 without the coarse stream {n2['fine_only']['N3']:+.3f}"
              "   (slow waves: expect a large drop)")
        print("=" * 72)
    return result


def plot_confusion(report, out_path: str | Path, title: str = "Confusion matrix") -> Path:
    """Row-normalised confusion heatmap (matplotlib, no seaborn)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(report.confusion, dtype=float)
    rows = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    im = ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(STAGE_NAMES)), STAGE_NAMES)
    ax.set_yticks(range(len(STAGE_NAMES)), STAGE_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expert label")
    ax.set_title(f"{title}\nkappa = {report.kappa:.3f}")
    for i in range(len(STAGE_NAMES)):
        for j in range(len(STAGE_NAMES)):
            ax.text(j, i, f"{rows[i, j]:.2f}", ha="center", va="center",
                    color="white" if rows[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    channel: str = "Fpz-Cz",
    crop_margin_min: float = 30.0,
    subjects: Sequence[str] | None = None,
    ablation: bool = False,
    out_dir: str | Path | None = None,
) -> Dict:
    """Convenience path used by the CLI: checkpoint + data directory -> report."""
    model, meta = load_checkpoint(checkpoint)
    recordings = load_dataset(data_root, channel, crop_margin_min, verbose=False)
    if subjects:
        wanted = set(subjects)
        recordings = [r for r in recordings if r.subject in wanted or r.night in wanted]
        if not recordings:
            raise ValueError(f"no recordings matched {sorted(wanted)}")
    X, y, _ = stack(recordings)

    report = evaluate_arrays(model, X, y, meta["norm"], meta["device"])
    result = {"metrics": report.to_dict(), "checkpoint": str(checkpoint)}
    if ablation:
        result["ablation"] = stream_ablation(model, X, y, meta["norm"], meta["device"])
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion(report, out_dir / "confusion.png")
        (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result
