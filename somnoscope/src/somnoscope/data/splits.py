"""Subject-disjoint splitting.

The single most common way to overstate sleep-staging performance is to shuffle
epochs and split at random. Consecutive 30 s epochs from one night are enormously
correlated -- same electrode impedance, same skull, same person -- so a random
split leaks the test subject into training and inflates kappa substantially.
Every split produced here is disjoint at the *subject* level, not the night level
and not the epoch level: both of a subject's nights always land on the same side.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np

__all__ = ["subject_split", "subject_kfold", "verify_disjoint"]


def _unique_subjects(groups: Sequence[str]) -> np.ndarray:
    return np.array(sorted(set(map(str, groups))))


def subject_split(
    groups: Sequence[str],
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split epoch indices into train/val/test with no subject appearing twice."""
    groups = np.asarray(list(map(str, groups)))
    subjects = _unique_subjects(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    # A requested fraction of exactly 0 means "no such split"; any positive
    # fraction is rounded up to at least one subject so a small cohort still
    # yields a usable held-out set instead of silently rounding to zero.
    n_test = 0 if test_fraction <= 0 else max(int(round(n * test_fraction)), 1 if n > 2 else 0)
    n_val = 0 if val_fraction <= 0 else max(
        int(round(n * val_fraction)), 1 if n - n_test > 2 else 0
    )
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test : n_test + n_val])

    is_test = np.isin(groups, list(test_s)) if test_s else np.zeros(len(groups), bool)
    is_val = np.isin(groups, list(val_s)) if val_s else np.zeros(len(groups), bool)
    train = np.flatnonzero(~is_test & ~is_val)
    return train, np.flatnonzero(is_val), np.flatnonzero(is_test)


def subject_kfold(
    groups: Sequence[str], n_folds: int = 5, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` for subject-disjoint k-fold cross-validation."""
    groups = np.asarray(list(map(str, groups)))
    subjects = _unique_subjects(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_folds = int(min(n_folds, len(subjects)))
    if n_folds < 2:
        raise ValueError(f"need >= 2 subjects to cross-validate, got {len(subjects)}")

    for fold in np.array_split(subjects, n_folds):
        is_test = np.isin(groups, fold)
        yield np.flatnonzero(~is_test), np.flatnonzero(is_test)


def verify_disjoint(groups: Sequence[str], *index_sets: Sequence[int]) -> bool:
    """Assert that no subject appears in more than one index set. Called by the trainer."""
    groups = np.asarray(list(map(str, groups)))
    seen: List[set] = [set(groups[np.asarray(idx, dtype=int)]) for idx in index_sets if len(idx)]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            overlap = seen[i] & seen[j]
            if overlap:
                raise AssertionError(f"subject leak between splits {i} and {j}: {sorted(overlap)}")
    return True
