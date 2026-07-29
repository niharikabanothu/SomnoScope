"""EDF round-trip, AASM label mapping, wake cropping and subject-disjoint splits."""

from __future__ import annotations

import numpy as np
import pytest

from somnoscope.data.edf import Annotation, read_annotations, read_edf, write_edf
from somnoscope.data.sleepedf import (
    STAGE_TO_INDEX,
    crop_wake,
    epoch_signal,
    find_pairs,
    hypnogram_from_annotations,
    load_recording,
    make_synthetic_dataset,
    make_synthetic_recording,
    stage_from_annotation,
    subject_id,
)
from somnoscope.data.splits import subject_kfold, subject_split, verify_disjoint


# --------------------------------------------------------------------------- #
# EDF
# --------------------------------------------------------------------------- #
def test_edf_roundtrip_preserves_signal(tmp_path):
    fs = 100.0
    t = np.arange(int(fs * 10)) / fs
    original = (50 * np.sin(2 * np.pi * 2 * t)).astype(np.float32)

    path = write_edf(tmp_path / "x.edf", {"EEG Fpz-Cz": original}, fs)
    signals, rates, header = read_edf(path)

    assert rates["EEG Fpz-Cz"] == pytest.approx(fs)
    assert header.n_records == 10
    # int16 quantisation over a 1000 uV span is ~0.03 uV per step.
    assert np.max(np.abs(signals["EEG Fpz-Cz"] - original)) < 0.1


def test_edf_channel_selection_is_substring_and_case_insensitive(tmp_path):
    fs = 100.0
    data = {"EEG Fpz-Cz": np.zeros(int(fs * 5), np.float32),
            "EOG horizontal": np.ones(int(fs * 5), np.float32)}
    path = write_edf(tmp_path / "multi.edf", data, fs)

    picked, _, _ = read_edf(path, channels=["fpz-cz"])
    assert list(picked) == ["EEG Fpz-Cz"]

    with pytest.raises(KeyError):
        read_edf(path, channels=["C3-A2"])


def test_annotations_roundtrip(tmp_path):
    fs = 100.0
    annotations = [
        Annotation(0.0, 30.0, "Sleep stage W"),
        Annotation(30.0, 60.0, "Sleep stage 2"),
        Annotation(90.0, 30.0, "Movement time"),
    ]
    path = write_edf(
        tmp_path / "hyp.edf",
        {"EEG Fpz-Cz": np.zeros(int(fs * 120), np.float32)},
        fs,
        record_duration=30.0,
        annotations=annotations,
    )
    got = read_annotations(path)
    texts = [a.text for a in got]
    assert "Sleep stage W" in texts
    assert "Sleep stage 2" in texts
    assert "Movement time" in texts
    stage_w = next(a for a in got if a.text == "Sleep stage W")
    assert stage_w.onset == pytest.approx(0.0)
    assert stage_w.duration == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# AASM mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sleep stage W", "W"),
        ("Sleep stage 1", "N1"),
        ("Sleep stage 2", "N2"),
        ("Sleep stage 3", "N3"),
        ("Sleep stage 4", "N3"),       # R&K S3+S4 merge into AASM N3
        ("Sleep stage R", "REM"),
    ],
)
def test_stage_mapping(text, expected):
    assert stage_from_annotation(text) == STAGE_TO_INDEX[expected]


@pytest.mark.parametrize("text", ["Sleep stage ?", "Movement time", "Lights off"])
def test_unscorable_stages_are_dropped_not_relabelled_as_wake(text):
    assert stage_from_annotation(text) is None


def test_hypnogram_expansion_marks_unscored_epochs_negative():
    annotations = [Annotation(0.0, 60.0, "Sleep stage W"),
                   Annotation(120.0, 30.0, "Sleep stage 2")]
    hypnogram = hypnogram_from_annotations(annotations, n_epochs=6)
    assert list(hypnogram) == [0, 0, -1, -1, STAGE_TO_INDEX["N2"], -1]


# --------------------------------------------------------------------------- #
# Wake cropping — the anti-triviality step
# --------------------------------------------------------------------------- #
def test_crop_wake_removes_the_long_wake_tails():
    w, n2 = STAGE_TO_INDEX["W"], STAGE_TO_INDEX["N2"]
    labels = np.array([w] * 300 + [n2] * 100 + [w] * 300)

    keep = crop_wake(labels, crop_margin_min=30.0)   # 30 min == 60 epochs

    assert keep.sum() == 60 + 100 + 60
    assert keep[240:460].all()                        # sleep plus both margins
    assert not keep[:240].any() and not keep[460:].any()
    # And the point of the exercise: Wake stops being an overwhelming majority.
    before = (labels == w).mean()
    after = (labels[keep] == w).mean()
    assert before > 0.85 and after < 0.60


def test_crop_wake_keeps_everything_when_there_is_no_sleep():
    labels = np.zeros(50, dtype=int)                  # all Wake
    assert crop_wake(labels).all()


def test_epoch_signal_drops_the_ragged_tail():
    signal = np.arange(3000 * 4 + 17, dtype=np.float32)
    epochs = epoch_signal(signal, 100.0)
    assert epochs.shape == (4, 3000)
    with pytest.raises(ValueError):
        epoch_signal(np.zeros(10), 100.0)


# --------------------------------------------------------------------------- #
# Splits — the leak that inflates every sleep-staging paper
# --------------------------------------------------------------------------- #
def test_subject_split_is_disjoint():
    groups = np.repeat([f"SC4{i:02d}" for i in range(10)], 50)
    train, val, test = subject_split(groups, 0.2, 0.1, seed=1)

    assert verify_disjoint(groups, train, val, test)
    assert len(train) + len(val) + len(test) == len(groups)
    assert set(groups[test]).isdisjoint(set(groups[train]))


def test_subject_split_with_zero_test_fraction_holds_out_nothing():
    groups = np.repeat([f"S{i}" for i in range(8)], 10)
    train, val, test = subject_split(groups, test_fraction=0.0, val_fraction=0.25, seed=0)
    assert len(test) == 0
    assert len(val) > 0
    assert verify_disjoint(groups, train, val)


def test_kfold_covers_every_subject_exactly_once():
    groups = np.repeat([f"S{i}" for i in range(10)], 20)
    seen = []
    for train_idx, test_idx in subject_kfold(groups, n_folds=5, seed=0):
        verify_disjoint(groups, train_idx, test_idx)
        seen.extend(set(groups[test_idx]))
    assert sorted(seen) == sorted(set(groups))


def test_verify_disjoint_raises_on_a_leak():
    groups = np.array(["A", "A", "B", "B"])
    with pytest.raises(AssertionError, match="subject leak"):
        verify_disjoint(groups, [0, 1, 2], [2, 3])


def test_subject_id_groups_both_nights_of_one_subject():
    assert subject_id("SC4001E0-PSG.edf") == subject_id("SC4002E0-PSG.edf") == "SC400"


# --------------------------------------------------------------------------- #
# Synthetic end-to-end load
# --------------------------------------------------------------------------- #
def test_synthetic_recording_loads_with_sensible_labels(tmp_path):
    psg, hyp = make_synthetic_recording(tmp_path, n_epochs=120, seed=3)
    recording = load_recording(psg, hyp, channel="Fpz-Cz", crop_margin_min=30.0)

    assert recording.sampling_rate == pytest.approx(100.0)
    assert recording.epochs.shape[1] == 3000
    assert len(recording) == len(recording.epochs)
    assert set(np.unique(recording.labels)).issubset(set(range(5)))
    assert {STAGE_TO_INDEX["N2"], STAGE_TO_INDEX["N3"]}.issubset(set(np.unique(recording.labels)))


def test_find_pairs_matches_psg_to_hypnogram(tmp_path):
    make_synthetic_dataset(tmp_path, n_subjects=3, n_epochs=60)
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 3
    for psg, hyp in pairs:
        assert psg.name[:7] == hyp.name[:7]
