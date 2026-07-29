"""Command line interface.

    somnoscope selftest                     # end-to-end on synthetic data, no download
    somnoscope train      --data data/sleep-edf
    somnoscope evaluate   --checkpoint runs/default/cv/fold0.pt --data data/sleep-edf
    somnoscope robustness --checkpoint runs/default/cv/fold0.pt --data data/sleep-edf
    somnoscope explain    --checkpoint runs/default/cv/fold0.pt --data data/sleep-edf
    somnoscope audit      --checkpoint runs/default/cv/fold0.pt --data data/sleep-edf
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path



def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="directory containing Sleep-EDF *.edf files")
    parser.add_argument("--channel", default="Fpz-Cz")
    parser.add_argument("--crop-margin-min", type=float, default=30.0,
                        help="minutes of Wake kept either side of the sleep period "
                             "(use a large value to disable cropping)")
    parser.add_argument("--limit", type=int, default=None, help="load at most N recordings")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="somnoscope",
        description="Sleep staging with a robustness and spectral-explainability audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ------------------------------------------------------------
    p = sub.add_parser("train", help="subject-disjoint cross-validated training")
    _add_data_args(p)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--norm", default="per_epoch", choices=["per_epoch", "global", "none"],
                   help="per_epoch z-scores each epoch and closes the amplitude shortcut; "
                        "global keeps absolute amplitude (the vulnerable baseline)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")

    # ---- evaluate ---------------------------------------------------------
    p = sub.add_parser("evaluate", help="score a checkpoint on held-out data")
    _add_data_args(p)
    _add_model_args(p)
    p.add_argument("--out", default=None)
    p.add_argument("--ablation", action="store_true", help="also run the stream ablation")

    # ---- robustness -------------------------------------------------------
    p = sub.add_parser("robustness", help="five-degradation sweep + amplitude shortcut probe")
    _add_data_args(p)
    _add_model_args(p)
    p.add_argument("--out", default="runs/default/robustness.json")
    p.add_argument("--severities", type=float, nargs="+", default=None)

    # ---- explain / audit --------------------------------------------------
    for name, helptext in (("explain", "Grad-CAM + band-power spectral audit"),
                           ("audit", "alias for `explain`")):
        p = sub.add_parser(name, help=helptext)
        _add_data_args(p)
        _add_model_args(p)
        p.add_argument("--out", default="runs/default/audit.json")
        p.add_argument("--max-per-stage", type=int, default=200)
        p.add_argument("--include-misclassified", action="store_true")

    # ---- selftest ---------------------------------------------------------
    p = sub.add_parser("selftest", help="run the whole pipeline on synthetic data")
    p.add_argument("--out", default=None, help="keep outputs here instead of a temp dir")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--subjects", type=int, default=6)
    p.add_argument("--norm", default="per_epoch", choices=["per_epoch", "global", "none"])
    p.add_argument("--device", default="auto")
    return parser


# --------------------------------------------------------------------------- #
def _load(args):
    from .data.sleepedf import load_dataset, stack

    recordings = load_dataset(args.data, args.channel, args.crop_margin_min,
                              limit=args.limit, verbose=False)
    X, y, groups = stack(recordings)
    return X, y, groups, recordings[0].sampling_rate


def cmd_train(args) -> int:
    from .train import TrainConfig, run_training

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      norm=args.norm, seed=args.seed, augment=not args.no_augment,
                      device=args.device)
    summary = run_training(args.data, args.out, args.channel, args.crop_margin_min,
                           cfg, args.folds, args.limit)
    print(f"\nwrote {Path(args.out) / 'summary.json'}")
    print(f"kappa {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f} across folds")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate_checkpoint

    evaluate_checkpoint(args.checkpoint, args.data, args.channel, args.crop_margin_min,
                        ablation=args.ablation, out_dir=args.out)
    return 0


def cmd_robustness(args) -> int:
    from .evaluate import load_checkpoint
    from .robustness import SEVERITIES, robustness_sweep

    model, meta = load_checkpoint(args.checkpoint)
    X, y, _, fs = _load(args)
    robustness_sweep(model, X, y, meta["norm"], fs,
                     severities=args.severities or SEVERITIES,
                     device=meta["device"], out_path=args.out)
    print(f"\nwrote {args.out}")
    return 0


def cmd_explain(args) -> int:
    from .evaluate import load_checkpoint
    from .explain.audit import spectral_audit

    model, meta = load_checkpoint(args.checkpoint)
    X, y, _, fs = _load(args)
    spectral_audit(model, X, y, fs=fs, norm=meta["norm"],
                   max_per_stage=args.max_per_stage,
                   correct_only=not args.include_misclassified,
                   device=meta["device"], out_path=args.out)
    print(f"\nwrote {args.out}")
    return 0


def cmd_selftest(args) -> int:
    """End-to-end smoke run on synthetic data.

    Exists so the repo is checkable without a 8 GB PhysioNet download: it builds
    synthetic nights whose spectra follow the AASM stage definitions, trains for a
    few epochs, and runs the robustness sweep and the spectral audit. It proves
    the plumbing, not the science -- the numbers it prints mean nothing about real
    sleep staging.
    """
    from .data.sleepedf import make_synthetic_dataset, load_dataset, stack
    from .data.splits import subject_split, verify_disjoint
    from .evaluate import stream_ablation
    from .explain.audit import spectral_audit
    from .metrics import format_report, kappa_report
    from .robustness import robustness_sweep
    from .train import TrainConfig, train_model
    import torch

    work = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="somnoscope-"))
    data_dir = work / "synthetic"
    print(f"[selftest] generating {args.subjects} synthetic subjects in {data_dir}")
    make_synthetic_dataset(data_dir, n_subjects=args.subjects, n_epochs=200)

    recordings = load_dataset(data_dir, "Fpz-Cz", crop_margin_min=30.0, verbose=False)
    X, y, groups = stack(recordings)
    fs = recordings[0].sampling_rate
    print(f"[selftest] {len(y)} epochs, {len(set(groups))} subjects, fs={fs:g} Hz")

    train_idx, val_idx, test_idx = subject_split(groups, 0.34, 0.17, seed=0)
    verify_disjoint(groups, train_idx, val_idx, test_idx)
    print(f"[selftest] split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=64, norm=args.norm, device=args.device)
    result = train_model(X[train_idx], y[train_idx], X[val_idx], y[val_idx], cfg, fs=fs)
    model = result["model"]
    device = torch.device(result["device"])

    from .data.transforms import normalize
    with torch.no_grad():
        logits = model(torch.from_numpy(normalize(X[test_idx], args.norm)).unsqueeze(1).to(device))
    report = kappa_report(y[test_idx], logits.argmax(1).cpu().numpy())
    print(format_report(report, "selftest, held-out synthetic subjects"))

    stream_ablation(model, X[test_idx], y[test_idx], args.norm, device)
    robustness_sweep(model, X[test_idx], y[test_idx], args.norm, fs, device=device,
                     out_path=work / "robustness.json")
    spectral_audit(model, X[test_idx], y[test_idx], fs=fs, norm=args.norm,
                   max_per_stage=60, device=device, out_path=work / "audit.json")

    print(f"\n[selftest] complete. artefacts in {work}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "robustness": cmd_robustness,
        "explain": cmd_explain,
        "audit": cmd_explain,
        "selftest": cmd_selftest,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
