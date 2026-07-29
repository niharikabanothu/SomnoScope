"""Training: subject-disjoint splits, class-balanced loss, kappa-selected checkpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data.sleepedf import STAGE_NAMES, class_distribution, load_dataset, stack
from .data.splits import subject_kfold, subject_split, verify_disjoint
from .data.transforms import DEGRADATIONS, apply_degradation, normalize
from .metrics import format_report, kappa_report
from .models.dual_stream_cnn import ModelConfig, build_model, class_weighted_loss

__all__ = ["TrainConfig", "EpochDataset", "train_model", "cross_validate", "run_training"]


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    norm: str = "per_epoch"
    seed: int = 0
    patience: int = 8
    device: str = "auto"
    augment: bool = True
    augment_severity: float = 0.35
    augment_kinds: Sequence[str] = field(
        # Deliberately excludes amplitude_scale: augmenting with the shortcut
        # probe would train the vulnerability away and make the probe
        # uninformative. Amplitude invariance has to come from normalisation, and
        # the robustness sweep has to be able to tell whether it actually did.
        default_factory=lambda: ("sensor_noise", "motion_drift", "bandwidth_loss")
    )
    num_workers: int = 0

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class EpochDataset(Dataset):
    """30 s epochs, normalised on the fly, optionally augmented during training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm: str = "per_epoch",
        fs: float = 100.0,
        augment: bool = False,
        severity: float = 0.35,
        kinds: Sequence[str] = (),
        seed: int = 0,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.norm = norm
        self.fs = fs
        self.augment = augment
        self.severity = severity
        self.kinds = [k for k in kinds if k in DEGRADATIONS]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.augment and self.kinds and self.rng.random() < 0.5:
            kind = self.kinds[self.rng.integers(len(self.kinds))]
            sev = float(self.rng.uniform(0.0, self.severity))
            x = apply_degradation(x, kind, sev, fs=self.fs, seed=int(self.rng.integers(1 << 31)))[0]
        x = normalize(x, self.norm)[0]
        return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), int(self.y[i])


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own augmentation RNG.

    Workers are forked copies of the dataset object, so without this every worker
    replays the *same* sequence of degradations -- the augmentation looks random
    per batch but is duplicated across workers, quietly reducing its effective
    diversity by a factor of ``num_workers``.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(
            (torch.initial_seed() + worker_id) % (2**32)
        )


def _loader(dataset: EpochDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        worker_init_fn=_worker_init if workers > 0 else None,
    )


@torch.no_grad()
def predict_all(model, loader, device) -> tuple:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(np.asarray(yb))
    return np.concatenate(trues), np.concatenate(preds)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
    fs: float = 100.0,
    verbose: bool = True,
) -> Dict:
    """Train one model. Returns the fitted model plus its training history.

    Model selection is on **validation kappa**, not validation loss or accuracy.
    With a 5-class imbalanced problem those disagree often, and the run that
    minimises loss is frequently the one that has quietly stopped predicting N1.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.resolve_device()

    model_cfg = model_cfg or ModelConfig(sampling_rate=fs)
    model = build_model(model_cfg).to(device)
    if verbose:
        print(model.describe())

    counts = np.bincount(y_train, minlength=len(STAGE_NAMES))
    criterion = class_weighted_loss(counts, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = _loader(
        EpochDataset(X_train, y_train, cfg.norm, fs, cfg.augment, cfg.augment_severity,
                     cfg.augment_kinds, cfg.seed),
        cfg.batch_size, True, cfg.num_workers,
    )
    val_loader = _loader(
        EpochDataset(X_val, y_val, cfg.norm, fs), cfg.batch_size, False, cfg.num_workers
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=max(len(train_loader), 1)
    )

    history: List[Dict] = []
    best = {"kappa": -np.inf, "state": None, "epoch": -1}
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        started, total_loss, seen = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss) * len(yb)
            seen += len(yb)

        y_true, y_pred = predict_all(model, val_loader, device)
        report = kappa_report(y_true, y_pred)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_kappa": report.kappa,
            "val_accuracy": report.accuracy,
            "val_macro_f1": report.macro_f1,
            "seconds": time.time() - started,
        }
        history.append(record)
        if verbose:
            print(
                f"  epoch {epoch + 1:3d}/{cfg.epochs}  loss {record['train_loss']:.4f}"
                f"  val kappa {report.kappa:.4f}  acc {report.accuracy:.4f}"
                f"  macroF1 {report.macro_f1:.4f}  ({record['seconds']:.1f}s)"
            )

        if report.kappa > best["kappa"] + 1e-5:
            best = {
                "kappa": report.kappa,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1} (best epoch {best['epoch'] + 1})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"model": model, "history": history, "best_epoch": best["epoch"],
            "best_val_kappa": best["kappa"], "device": str(device), "model_cfg": model_cfg}


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: TrainConfig,
    n_folds: int = 5,
    fs: float = 100.0,
    out_dir: str | Path = "runs/cv",
    verbose: bool = True,
) -> Dict:
    """Subject-disjoint k-fold CV. The fold spread is the honest error bar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_reports, kappas, oof_true, oof_pred = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(subject_kfold(groups, n_folds, cfg.seed)):
        verify_disjoint(groups, train_idx, test_idx)
        # Carve a validation set out of the *training* subjects only.
        inner_train, inner_val, _ = subject_split(
            groups[train_idx], test_fraction=0.0, val_fraction=0.15, seed=cfg.seed + fold
        )
        tr = train_idx[inner_train]
        va = train_idx[inner_val] if len(inner_val) else train_idx[: max(len(train_idx) // 10, 1)]
        verify_disjoint(groups, tr, va, test_idx)

        if verbose:
            print(f"\n--- fold {fold + 1}/{n_folds} "
                  f"(train {len(tr)}, val {len(va)}, test {len(test_idx)} epochs; "
                  f"test subjects: {sorted(set(groups[test_idx]))}) ---")

        result = train_model(X[tr], y[tr], X[va], y[va], cfg, fs=fs, verbose=verbose)
        model, device = result["model"], torch.device(result["device"])
        test_loader = _loader(EpochDataset(X[test_idx], y[test_idx], cfg.norm, fs),
                              cfg.batch_size, False, cfg.num_workers)
        y_true, y_pred = predict_all(model, test_loader, device)
        report = kappa_report(y_true, y_pred)

        if verbose:
            print(format_report(report, f"fold {fold + 1} (held-out subjects)"))
        torch.save(
            {"state_dict": model.state_dict(), "model_cfg": result["model_cfg"].to_dict(),
             "norm": cfg.norm, "fs": fs, "fold": fold, "kappa": report.kappa},
            out_dir / f"fold{fold}.pt",
        )
        fold_reports.append(report.to_dict())
        kappas.append(report.kappa)
        oof_true.append(y_true)
        oof_pred.append(y_pred)

    pooled = kappa_report(np.concatenate(oof_true), np.concatenate(oof_pred), bootstrap=500)
    summary = {
        "fold_kappas": kappas,
        "kappa_mean": float(np.mean(kappas)),
        "kappa_std": float(np.std(kappas)),
        "pooled": pooled.to_dict(),
        "folds": fold_reports,
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(format_report(pooled, "pooled out-of-fold (all held-out subjects)"))
        print(f"  fold kappa: {np.mean(kappas):.4f} +/- {np.std(kappas):.4f} "
              f"({', '.join(f'{k:.3f}' for k in kappas)})")
    return summary


def run_training(
    data_root: str | Path,
    out_dir: str | Path = "runs/default",
    channel: str = "Fpz-Cz",
    crop_margin_min: float | None = 30.0,
    cfg: TrainConfig | None = None,
    n_folds: int = 5,
    limit: int | None = None,
    verbose: bool = True,
) -> Dict:
    """Load -> report the wake-cropping effect -> cross-validate -> save."""
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"loading recordings from {data_root} (channel={channel})")
    raw = load_dataset(data_root, channel, crop_margin_min=None, limit=limit, verbose=verbose)
    _, y_raw, _ = stack(raw)
    before = class_distribution(y_raw)

    recordings = load_dataset(data_root, channel, crop_margin_min, limit=limit, verbose=False)
    X, y, groups = stack(recordings)
    after = class_distribution(y)
    fs = recordings[0].sampling_rate

    if verbose:
        print("\nclass balance, before -> after wake cropping "
              f"(margin {crop_margin_min} min):")
        for stage in STAGE_NAMES:
            print(f"  {stage:<4} {before[stage]:6.1%} -> {after[stage]:6.1%}")
        print(f"  epochs {len(y_raw):,} -> {len(y):,}   "
              f"subjects {len(set(groups))}   fs {fs:g} Hz\n")

    summary = cross_validate(X, y, groups, cfg, n_folds, fs, out_dir / "cv", verbose)
    summary["class_balance"] = {"before_crop": before, "after_crop": after,
                                "n_epochs_before": int(len(y_raw)), "n_epochs_after": int(len(y))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "dataset_index.npz", y=y, groups=groups)
    return summary
