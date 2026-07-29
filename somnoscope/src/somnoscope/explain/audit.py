"""The spectral explainability audit: does attention land where the AASM rules say it should?

For every correctly-classified test epoch we compute a per-stream Grad-CAM, project
it onto frequency, and check which band the model over-weighted. Each stage has one
pre-registered expected band (``EXPECTED_BAND`` in ``bandpower.py``), so the output
is a **hit rate** -- a number that can be wrong -- rather than a page of heatmaps.

Reported per stage:

``dominant_band``        the band the model over-weighted most, per stream
``hit``                  whether that matches the clinically defining band
``expected_band_share``  how much of the attribution mass fell in the right band
``data_profile``         the epoch's actual relative band power, for contrast

The contrast matters. If N3 epochs are 60% delta by raw power, a model attending
to delta proves nothing on its own -- there is nothing else to attend to. The
audit therefore scores *over*-weighting relative to the epoch's own spectrum
(``band_attribution(..., contrast=True)``), and prints the data profile alongside
so the comparison is visible rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..data.sleepedf import STAGE_NAMES
from ..data.transforms import normalize
from .bandpower import BANDS, EXPECTED_BAND, band_power_by_stage, dominant_band
from .gradcam import GradCAM1D, band_attribution

__all__ = ["spectral_audit", "format_audit"]


def spectral_audit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    fs: float = 100.0,
    norm: str = "per_epoch",
    max_per_stage: int = 200,
    correct_only: bool = True,
    device=None,
    seed: int = 0,
    out_path: str | Path | None = None,
    verbose: bool = True,
) -> Dict:
    """Run the audit over a sample of epochs per stage."""
    device = device or next(model.parameters()).device
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    data_profile = band_power_by_stage(X, y, STAGE_NAMES, fs)
    results: Dict[str, Dict] = {}
    cam_tool = GradCAM1D(model)

    try:
        for stage_idx, stage in enumerate(STAGE_NAMES):
            idx = np.flatnonzero(y == stage_idx)
            if idx.size == 0:
                continue
            if idx.size > max_per_stage:
                idx = rng.choice(idx, max_per_stage, replace=False)

            raw = X[idx]
            xb = torch.from_numpy(normalize(raw, norm)).unsqueeze(1).to(device)
            out = cam_tool(xb, target=stage_idx)

            if correct_only:
                keep = out["prediction"] == stage_idx
                if keep.sum() < 5:            # too few to say anything; use them all
                    keep = np.ones(len(idx), dtype=bool)
            else:
                keep = np.ones(len(idx), dtype=bool)

            per_stream = {}
            for stream, cams in out["cams"].items():
                attributions: List[Dict[str, float]] = [
                    band_attribution(raw[i], cams[i], BANDS, fs)
                    for i in np.flatnonzero(keep)
                ]
                mean_attr = {
                    band: float(np.mean([a[band] for a in attributions])) for band in BANDS
                }
                top = dominant_band(mean_attr)
                per_stream[stream] = {
                    "attribution": mean_attr,
                    "dominant_band": top,
                    "expected_band": EXPECTED_BAND[stage],
                    "expected_band_share": mean_attr[EXPECTED_BAND[stage]],
                    "hit": top == EXPECTED_BAND[stage],
                    "n_epochs": int(keep.sum()),
                }

            results[stage] = {
                "streams": per_stream,
                "data_profile": data_profile[stage],
                "data_dominant_band": dominant_band(data_profile[stage]),
                "any_stream_hit": any(s["hit"] for s in per_stream.values()),
            }
    finally:
        cam_tool.remove()

    scored = [r for stage, r in results.items() if stage in ("N1", "N2", "N3")]
    audit = {
        "per_stage": results,
        "hit_rate_all_stages": float(
            np.mean([r["any_stream_hit"] for r in results.values()]) if results else float("nan")
        ),
        "hit_rate_scored_stages": float(
            np.mean([r["any_stream_hit"] for r in scored]) if scored else float("nan")
        ),
        "stream_specialisation": _stream_specialisation(results),
        "caveat": (
            "W and REM are excluded from the scored hit rate: both are marked by theta "
            "on a frontal derivation and are separated clinically by EOG and chin EMG, "
            "which a single-channel Fpz-Cz montage does not have."
        ),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(audit, indent=2))
    if verbose:
        print(format_audit(audit))
    return audit


def _stream_specialisation(results: Dict) -> Dict[str, str]:
    """Check the dual-stream premise: fine -> spindles (N2), coarse -> slow waves (N3)."""
    out = {}
    for stage, expect_stream in (("N2", "fine"), ("N3", "coarse")):
        entry = results.get(stage)
        if not entry:
            continue
        streams = entry["streams"]
        expected = EXPECTED_BAND[stage]
        shares = {name: s["expected_band_share"] for name, s in streams.items()}
        if not shares:
            continue
        leader = max(shares, key=shares.get)
        out[stage] = (
            f"{expected} attribution led by the {leader} stream "
            f"({shares[leader]:.2f} vs {min(shares.values()):.2f}) -- "
            + ("as designed" if leader == expect_stream
               else f"expected the {expect_stream} stream to lead")
        )
    return out


def format_audit(audit: Dict) -> str:
    lines = [
        "",
        "=== Spectral explainability audit ".ljust(72, "="),
        "  Grad-CAM projected onto frequency, contrasted against each epoch's own",
        "  spectrum, then compared with the band the AASM rules define the stage by.",
        "",
        "  " + "stage".ljust(6) + "stream".ljust(9) + "expected".ljust(10)
        + "attended".ljust(10) + "share".rjust(7) + "   hit",
        "  " + "-" * 52,
    ]
    for stage, entry in audit["per_stage"].items():
        for stream, s in entry["streams"].items():
            lines.append(
                f"  {stage:<6}{stream:<9}{s['expected_band']:<10}{s['dominant_band']:<10}"
                f"{s['expected_band_share']:7.2f}   {'yes' if s['hit'] else 'no'}"
            )
        lines.append(
            f"  {'':<6}{'(data)':<9}{'':<10}{entry['data_dominant_band']:<10}"
            f"{'':>7}   raw band power of these epochs"
        )
    lines += [
        "",
        f"  hit rate, N1/N2/N3   {audit['hit_rate_scored_stages']:.2f}",
        f"  hit rate, all stages {audit['hit_rate_all_stages']:.2f}",
    ]
    for stage, note in audit["stream_specialisation"].items():
        lines.append(f"  {stage}: {note}")
    lines += ["", f"  caveat: {audit['caveat']}", "=" * 72]
    return "\n".join(lines)
