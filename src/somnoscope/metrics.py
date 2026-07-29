"""Scoring, with Cohen's kappa as the headline number.

Why not accuracy
----------------
On uncropped Sleep-EDF, ~60-70% of epochs are Wake. A constant "Wake" predictor
therefore scores ~0.65 accuracy while containing no sleep-staging ability at all.
Cohen's kappa corrects for exactly that chance agreement, and it is also the
metric the sleep literature uses to quantify how much two *human* scorers agree.

The human ceiling
-----------------
Published inter-scorer reliability for 5-stage AASM scoring sits at roughly
**kappa 0.75-0.80** (Danker-Hopfe et al., 2009, ~0.76 for R&K/AASM across
European centres; Rosenberg & Van Hout, 2013, ~0.76 in the AASM inter-scorer
programme; Younes et al., 2016, similar). That band is the reference this repo
reports against, because a model at kappa 0.78 is not "78% good" -- it is
performing at the level two trained humans agree with each other, which is the
practical ceiling for a task whose ground truth is itself a human opinion.

``kappa_report`` therefore returns the raw kappa, its position within that band,
and the per-stage breakdown, since the aggregate hides the fact that N1 is where
both humans and models do worst (human pairwise agreement on N1 alone is often
below 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .data.sleepedf import STAGE_NAMES

__all__ = [
    "HUMAN_KAPPA_FLOOR",
    "HUMAN_KAPPA_CEILING",
    "StageReport",
    "kappa_report",
    "bootstrap_kappa_ci",
    "format_report",
    "confusion_table",
]

HUMAN_KAPPA_FLOOR = 0.75
HUMAN_KAPPA_CEILING = 0.80


@dataclass
class StageReport:
    accuracy: float
    kappa: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    support: Dict[str, int]
    confusion: np.ndarray
    kappa_ci: tuple | None = None
    notes: List[str] = field(default_factory=list)

    @property
    def reaches_human_floor(self) -> bool:
        return self.kappa >= HUMAN_KAPPA_FLOOR

    @property
    def ceiling_fraction(self) -> float:
        """Kappa as a fraction of the midpoint of the human agreement band."""
        midpoint = (HUMAN_KAPPA_FLOOR + HUMAN_KAPPA_CEILING) / 2.0
        return float(self.kappa / midpoint)

    def to_dict(self) -> Dict:
        d = {
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "macro_f1": self.macro_f1,
            "per_class_f1": self.per_class_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "support": self.support,
            "confusion": self.confusion.tolist(),
            "human_band": [HUMAN_KAPPA_FLOOR, HUMAN_KAPPA_CEILING],
            "ceiling_fraction": self.ceiling_fraction,
        }
        if self.kappa_ci:
            d["kappa_ci95"] = list(self.kappa_ci)
        return d


def kappa_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] = STAGE_NAMES,
    bootstrap: int = 0,
    seed: int = 0,
) -> StageReport:
    """Full scoring pass. Set ``bootstrap`` > 0 for a 95% CI on kappa."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    idx = list(range(len(labels)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, zero_division=0
    )
    report = StageReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred, labels=idx)),
        macro_f1=float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        per_class_f1={labels[i]: float(f1[i]) for i in idx},
        per_class_precision={labels[i]: float(precision[i]) for i in idx},
        per_class_recall={labels[i]: float(recall[i]) for i in idx},
        support={labels[i]: int(support[i]) for i in idx},
        confusion=confusion_matrix(y_true, y_pred, labels=idx),
    )

    # The majority-class sanity check that motivates using kappa at all.
    counts = np.bincount(y_true, minlength=len(labels))
    majority = counts.max() / max(counts.sum(), 1)
    report.notes.append(
        f"majority class = {labels[int(counts.argmax())]} at {majority:.1%} of epochs; "
        f"a constant predictor would score accuracy {majority:.3f} and kappa 0.000"
    )
    if report.kappa >= HUMAN_KAPPA_FLOOR:
        report.notes.append(
            f"kappa {report.kappa:.3f} is within/above the human inter-scorer band "
            f"[{HUMAN_KAPPA_FLOOR}, {HUMAN_KAPPA_CEILING}]"
        )
    else:
        gap = HUMAN_KAPPA_FLOOR - report.kappa
        report.notes.append(
            f"kappa {report.kappa:.3f} is {gap:.3f} below the human band floor "
            f"({HUMAN_KAPPA_FLOOR})"
        )

    if bootstrap:
        report.kappa_ci = bootstrap_kappa_ci(y_true, y_pred, n=bootstrap, seed=seed)
    return report


def bootstrap_kappa_ci(
    y_true: Sequence[int], y_pred: Sequence[int], n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple:
    """Percentile bootstrap CI over epochs.

    Note this treats epochs as independent, which they are not -- neighbouring
    epochs come from the same night. It is a lower bound on the true interval and
    is reported as such; the subject-level spread across CV folds is the honest
    version and is what ``train.py`` prints.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    idx = list(range(len(STAGE_NAMES)))
    draws = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, n_samples, n_samples)
        draws[i] = cohen_kappa_score(y_true[pick], y_pred[pick], labels=idx)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def confusion_table(confusion: np.ndarray, labels: Sequence[str] = STAGE_NAMES) -> str:
    """Row-normalised confusion matrix as fixed-width text (rows = truth)."""
    confusion = np.asarray(confusion, dtype=float)
    rows = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    width = max(len(x) for x in labels) + 2
    head = " " * (width + 2) + "".join(f"{name:>7}" for name in labels)
    lines = [head, " " * (width + 2) + "-" * (7 * len(labels))]
    for i, label in enumerate(labels):
        cells = "".join(f"{rows[i, j]:7.2f}" for j in range(len(labels)))
        lines.append(f"{label:>{width}} |{cells}   (n={int(confusion[i].sum())})")
    return "\n".join(lines)


def format_report(report: StageReport, title: str = "Evaluation") -> str:
    """Human-readable summary block, printed by every CLI subcommand."""
    bar_pos = min(max(report.kappa / HUMAN_KAPPA_CEILING, 0.0), 1.2)
    bar = "#" * int(bar_pos * 40)
    lines = [
        "",
        f"=== {title} ".ljust(72, "="),
        f"  accuracy     {report.accuracy:.4f}",
        f"  macro F1     {report.macro_f1:.4f}",
        f"  Cohen kappa  {report.kappa:.4f}"
        + (f"   95% CI [{report.kappa_ci[0]:.3f}, {report.kappa_ci[1]:.3f}]"
           if report.kappa_ci else ""),
        "",
        f"  human inter-scorer band: [{HUMAN_KAPPA_FLOOR:.2f}, {HUMAN_KAPPA_CEILING:.2f}]",
        f"  |{bar:<40}|  {report.ceiling_fraction:.0%} of the band midpoint",
        "",
        "  per-stage F1 / precision / recall / n",
    ]
    for stage in STAGE_NAMES:
        lines.append(
            f"    {stage:<4} {report.per_class_f1[stage]:.3f}  "
            f"{report.per_class_precision[stage]:.3f}  "
            f"{report.per_class_recall[stage]:.3f}  "
            f"{report.support[stage]:>6d}"
        )
    lines += ["", "  confusion (row-normalised, rows = expert label)",
              confusion_table(report.confusion), ""]
    for note in report.notes:
        lines.append(f"  note: {note}")
    lines.append("=" * 72)
    return "\n".join(lines)
