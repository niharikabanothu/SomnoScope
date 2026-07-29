"""Robustness sweep over five wearable-realistic degradations.

Each degradation is applied to the held-out test epochs at severities
``0.0, 0.25, 0.5, 0.75, 1.0`` and the model is re-scored. Nothing is retrained --
the question is what the *deployed* model does when the signal stops looking like
clinical PSG.

The amplitude shortcut probe
----------------------------
``amplitude_scale`` is scored differently from the other four. Multiplying an EEG
epoch by a constant does not change its sleep stage: stage is defined by relative
spectral content, and a gain change is exactly what a different electrode
impedance or a different amplifier produces. So a correct model should be
*invariant* here, and any degradation under gain change is not "the task got
harder" -- it is evidence the model is reading absolute amplitude as a proxy
feature.

``shortcut_index`` reports that directly: the mean kappa drop under amplitude
scaling relative to the drop under an equally-severe perturbation that genuinely
destroys information (``sensor_noise``). An index near 0 means the model ignores
gain, as it should. An index near or above 1 means gain is doing as much damage
as noise that actually removes signal, which is the amplitude shortcut.

The most useful thing this probe does is make the normalisation choice
falsifiable. ``--norm per_epoch`` z-scores every epoch and should drive the index
to ~0; ``--norm global`` preserves absolute amplitude and should not. Running the
sweep both ways turns "we normalised the data" from a methods-section sentence
into a measured number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from .data.sleepedf import STAGE_NAMES
from .data.transforms import DEGRADATIONS, apply_degradation, normalize
from .metrics import kappa_report

__all__ = [
    "SEVERITIES",
    "sweep_degradation",
    "robustness_sweep",
    "shortcut_index",
    "format_robustness_table",
]

SEVERITIES: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)


@torch.no_grad()
def _score(model, X: np.ndarray, y: np.ndarray, norm: str, device, batch_size: int = 256):
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        chunk = normalize(X[start : start + batch_size], norm)
        xb = torch.from_numpy(chunk).unsqueeze(1).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return kappa_report(y, np.concatenate(preds))


def sweep_degradation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    norm: str = "per_epoch",
    fs: float = 100.0,
    severities: Sequence[float] = SEVERITIES,
    device=None,
    seed: int = 0,
) -> List[Dict]:
    """Score one degradation across the severity axis."""
    device = device or next(model.parameters()).device
    rows = []
    for sev in severities:
        Xd = X if sev == 0 else apply_degradation(X, name, sev, fs=fs, seed=seed)
        report = _score(model, Xd, y, norm, device)
        rows.append({
            "degradation": name,
            "severity": float(sev),
            "kappa": report.kappa,
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "per_class_f1": report.per_class_f1,
        })
    return rows


def robustness_sweep(
    model,
    X: np.ndarray,
    y: np.ndarray,
    norm: str = "per_epoch",
    fs: float = 100.0,
    degradations: Sequence[str] = tuple(DEGRADATIONS),
    severities: Sequence[float] = SEVERITIES,
    device=None,
    seed: int = 0,
    out_path: str | Path | None = None,
    verbose: bool = True,
) -> Dict:
    """Full sweep over all degradations, plus the shortcut analysis."""
    device = device or next(model.parameters()).device
    baseline = _score(model, X, y, norm, device)
    rows: List[Dict] = []
    for name in degradations:
        if verbose:
            print(f"  sweeping {name} ...")
        rows.extend(sweep_degradation(model, X, y, name, norm, fs, severities, device, seed))

    result = {
        "norm": norm,
        "baseline_kappa": baseline.kappa,
        "baseline_accuracy": baseline.accuracy,
        "severities": list(severities),
        "rows": rows,
        "retention": _retention(rows, baseline.kappa),
        "shortcut": shortcut_index(rows, baseline.kappa),
        "stage_sensitivity": _stage_sensitivity(rows, baseline),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    if verbose:
        print(format_robustness_table(result))
    return result


def _retention(rows: Sequence[Dict], baseline_kappa: float) -> Dict[str, float]:
    """Kappa retained at maximum severity, as a fraction of the clean baseline."""
    out = {}
    for row in rows:
        if row["severity"] == max(r["severity"] for r in rows):
            out[row["degradation"]] = float(row["kappa"] / max(baseline_kappa, 1e-6))
    return out


def shortcut_index(
    rows: Sequence[Dict],
    baseline_kappa: float,
    probe: str = "amplitude_scale",
    reference: str = "sensor_noise",
) -> Dict[str, float]:
    """Damage from a label-preserving gain change, relative to real signal loss.

    ``index = mean kappa drop under gain change / mean kappa drop under noise``

    * ~0.0  -- the model ignores absolute amplitude (what we want).
    * ~0.5  -- gain change costs half as much as destroying the signal: partial
      reliance on amplitude.
    * >=1.0 -- rescaling the signal is as damaging as burying it in noise. The
      model is keying on amplitude, and it will not transfer to a wearable with a
      different gain.
    """
    def mean_drop(name: str) -> float:
        drops = [baseline_kappa - r["kappa"] for r in rows
                 if r["degradation"] == name and r["severity"] > 0]
        return float(np.mean(drops)) if drops else float("nan")

    probe_drop = mean_drop(probe)
    reference_drop = mean_drop(reference)
    index = float(probe_drop / reference_drop) if reference_drop > 1e-6 else float("nan")

    if not np.isfinite(index):
        verdict = "not computable (reference degradation caused no measurable drop)"
    elif index < 0.15:
        verdict = "amplitude-invariant: no shortcut detected"
    elif index < 0.5:
        verdict = "mild amplitude dependence"
    elif index < 1.0:
        verdict = "substantial amplitude dependence -- shortcut likely"
    else:
        verdict = "amplitude shortcut: gain change is as damaging as signal loss"

    return {
        "probe": probe,
        "reference": reference,
        "probe_mean_kappa_drop": probe_drop,
        "reference_mean_kappa_drop": reference_drop,
        "index": index,
        "verdict": verdict,
    }


def _stage_sensitivity(rows: Sequence[Dict], baseline) -> Dict[str, Dict[str, float]]:
    """Which stage each degradation hurts most, at maximum severity.

    The interesting confirmations here are directional: ``bandwidth_loss`` strips
    the sigma band and should cost N2 (spindles) more than N3, while
    ``motion_drift`` contaminates delta and should cost N3 more than N2. If those
    come out the other way round, the model was not using the bands it claims to.
    """
    top = max(r["severity"] for r in rows)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["severity"] != top:
            continue
        out[row["degradation"]] = {
            stage: float(row["per_class_f1"][stage] - baseline.per_class_f1[stage])
            for stage in STAGE_NAMES
        }
    return out


def format_robustness_table(result: Dict) -> str:
    severities = result["severities"]
    lines = [
        "",
        "=== Robustness sweep ".ljust(72, "="),
        f"  clean baseline kappa {result['baseline_kappa']:.4f}   normalisation: {result['norm']}",
        "",
        "  kappa by severity",
        "  " + "degradation".ljust(18) + "".join(f"{s:>8.2f}" for s in severities) + "   retained",
        "  " + "-" * (18 + 8 * len(severities) + 11),
    ]
    by_name: Dict[str, Dict[float, float]] = {}
    for row in result["rows"]:
        by_name.setdefault(row["degradation"], {})[row["severity"]] = row["kappa"]
    for name, values in by_name.items():
        cells = "".join(f"{values.get(s, float('nan')):8.3f}" for s in severities)
        tag = "  <- probe" if name == result["shortcut"]["probe"] else ""
        lines.append(f"  {name:<18}{cells}   {result['retention'].get(name, float('nan')):7.1%}{tag}")

    sc = result["shortcut"]
    lines += [
        "",
        "  amplitude shortcut probe",
        f"    mean kappa drop, {sc['probe']:<16} {sc['probe_mean_kappa_drop']:.4f}",
        f"    mean kappa drop, {sc['reference']:<16} {sc['reference_mean_kappa_drop']:.4f}",
        f"    shortcut index                    {sc['index']:.3f}",
        f"    verdict: {sc['verdict']}",
        "",
        "  per-stage F1 change at maximum severity",
        "  " + " " * 18 + "".join(f"{s:>8}" for s in STAGE_NAMES),
    ]
    for name, deltas in result["stage_sensitivity"].items():
        cells = "".join(f"{deltas[s]:+8.3f}" for s in STAGE_NAMES)
        lines.append(f"  {name:<18}{cells}")
    lines.append("=" * 72)
    return "\n".join(lines)
