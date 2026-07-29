"""Dependency-free EDF / EDF+ reader (and a minimal writer for test fixtures).

Sleep-EDF ships two files per night:

    SC4001E0-PSG.edf        polysomnogram   (EEG / EOG / EMG / event marker)
    SC4001EC-Hypnogram.edf  EDF+ annotations only (the expert scoring)

Rather than pull in ``mne`` or ``pyedflib`` just to read a well-specified 256-byte
header, this module implements the EDF spec directly. That keeps the install
light and, more usefully, makes the exact physical-unit scaling visible in the
code -- which matters here, because the amplitude-scaling shortcut probe in
``somnoscope.robustness`` is only meaningful if you know the signal really is in
microvolts before you start rescaling it.

EDF spec: Kemp et al. (1992); EDF+ annotations: Kemp & Olivan (2003).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "EdfHeader",
    "SignalHeader",
    "Annotation",
    "read_edf_header",
    "read_edf",
    "read_annotations",
    "write_edf",
]

ANNOTATION_LABEL = "EDF Annotations"


# --------------------------------------------------------------------------- #
# Header containers
# --------------------------------------------------------------------------- #
@dataclass
class SignalHeader:
    label: str
    transducer: str
    physical_dim: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    prefilter: str
    n_samples_per_record: int

    @property
    def is_annotation(self) -> bool:
        return self.label.strip() == ANNOTATION_LABEL

    def gain(self) -> Tuple[float, float]:
        """Return ``(scale, offset)`` mapping digital -> physical units."""
        d_span = self.digital_max - self.digital_min
        p_span = self.physical_max - self.physical_min
        if d_span == 0:
            return 1.0, 0.0
        scale = p_span / d_span
        offset = self.physical_min - self.digital_min * scale
        return scale, offset


@dataclass
class EdfHeader:
    version: str
    patient_id: str
    recording_id: str
    start_date: str
    start_time: str
    header_bytes: int
    reserved: str
    n_records: int
    record_duration: float
    n_signals: int
    signals: List[SignalHeader] = field(default_factory=list)

    @property
    def is_edf_plus(self) -> bool:
        return self.reserved.strip().startswith("EDF+")

    def sampling_rate(self, label: str) -> float:
        for s in self.signals:
            if s.label.strip() == label.strip():
                return s.n_samples_per_record / self.record_duration
        raise KeyError(f"signal {label!r} not in {[s.label for s in self.signals]}")


@dataclass
class Annotation:
    onset: float          # seconds from recording start
    duration: float       # seconds (0.0 when unspecified)
    text: str


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _ascii(buf: bytes) -> str:
    return buf.decode("ascii", errors="replace").strip()


def _num(buf: bytes, cast=float, default=0):
    txt = _ascii(buf)
    if not txt:
        return cast(default)
    try:
        return cast(float(txt))
    except ValueError:
        return cast(default)


def read_edf_header(path: str | Path) -> EdfHeader:
    """Parse the fixed 256-byte header plus the variable per-signal header."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = fh.read(256)
        if len(raw) < 256:
            raise ValueError(f"{path} is too short to be an EDF file")

        header = EdfHeader(
            version=_ascii(raw[0:8]),
            patient_id=_ascii(raw[8:88]),
            recording_id=_ascii(raw[88:168]),
            start_date=_ascii(raw[168:176]),
            start_time=_ascii(raw[176:184]),
            header_bytes=_num(raw[184:192], int),
            reserved=_ascii(raw[192:236]),
            n_records=_num(raw[236:244], int, -1),
            record_duration=_num(raw[244:252], float, 1.0),
            n_signals=_num(raw[252:256], int),
        )

        ns = header.n_signals
        if ns <= 0:
            raise ValueError(f"{path}: header declares {ns} signals")

        def block(width: int) -> List[bytes]:
            data = fh.read(width * ns)
            return [data[i * width : (i + 1) * width] for i in range(ns)]

        labels = block(16)
        transducers = block(80)
        dims = block(8)
        pmins = block(8)
        pmaxs = block(8)
        dmins = block(8)
        dmaxs = block(8)
        prefilters = block(80)
        nsamps = block(8)
        _reserved = block(32)

        for i in range(ns):
            header.signals.append(
                SignalHeader(
                    label=_ascii(labels[i]),
                    transducer=_ascii(transducers[i]),
                    physical_dim=_ascii(dims[i]),
                    physical_min=_num(pmins[i], float),
                    physical_max=_num(pmaxs[i], float, 1.0),
                    digital_min=_num(dmins[i], float, -32768),
                    digital_max=_num(dmaxs[i], float, 32767),
                    prefilter=_ascii(prefilters[i]),
                    n_samples_per_record=_num(nsamps[i], int),
                )
            )
    return header


def _read_records(path: Path, header: EdfHeader) -> np.ndarray:
    """Return the raw int16 data-record block, shape ``(n_records, samples_per_record)``."""
    per_record = sum(s.n_samples_per_record for s in header.signals)
    with path.open("rb") as fh:
        fh.seek(header.header_bytes)
        buf = fh.read()

    total = len(buf) // 2
    n_records = header.n_records
    if n_records is None or n_records <= 0:            # -1 == "unknown", legal in EDF
        n_records = total // per_record
    n_records = min(n_records, total // per_record)
    if n_records == 0:
        raise ValueError(f"{path}: no complete data records found")

    flat = np.frombuffer(buf, dtype="<i2", count=n_records * per_record)
    return flat.reshape(n_records, per_record)


def read_edf(
    path: str | Path,
    channels: Sequence[str] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], EdfHeader]:
    """Read an EDF/EDF+ file.

    Parameters
    ----------
    path : file to read.
    channels : subset of signal labels to return. Matching is case-insensitive
        and substring-based, so ``"Fpz-Cz"`` finds ``"EEG Fpz-Cz"``. ``None``
        returns every non-annotation signal.

    Returns
    -------
    signals : ``{label: float32 array}`` in physical units (microvolts for EEG).
    rates   : ``{label: sampling rate in Hz}``.
    header  : the parsed :class:`EdfHeader`.
    """
    path = Path(path)
    header = read_edf_header(path)
    records = _read_records(path, header)

    # Column offset of each signal inside one data record.
    offsets, cursor = [], 0
    for s in header.signals:
        offsets.append(cursor)
        cursor += s.n_samples_per_record

    wanted = None
    if channels is not None:
        wanted = [c.strip().lower() for c in channels]

    signals: Dict[str, np.ndarray] = {}
    rates: Dict[str, float] = {}
    for sig, off in zip(header.signals, offsets):
        if sig.is_annotation or sig.n_samples_per_record == 0:
            continue
        label = sig.label.strip()
        if wanted is not None and not any(w in label.lower() for w in wanted):
            continue

        digital = records[:, off : off + sig.n_samples_per_record].astype(np.float32)
        scale, offset = sig.gain()
        signals[label] = (digital * scale + offset).reshape(-1)
        rates[label] = sig.n_samples_per_record / header.record_duration

    if wanted is not None and not signals:
        raise KeyError(
            f"none of {list(channels)} matched signals "
            f"{[s.label.strip() for s in header.signals]} in {path.name}"
        )
    return signals, rates, header


_TAL_SPLIT = re.compile(rb"\x00+")


def read_annotations(path: str | Path) -> List[Annotation]:
    """Read EDF+ TAL annotations (Sleep-EDF stores hypnograms this way).

    Each Time-stamped Annotations List looks like::

        +onset[\\x15duration]\\x14text\\x14[text\\x14...]\\x00
    """
    path = Path(path)
    header = read_edf_header(path)
    records = _read_records(path, header)

    offsets, cursor = [], 0
    for s in header.signals:
        offsets.append(cursor)
        cursor += s.n_samples_per_record

    out: List[Annotation] = []
    for sig, off in zip(header.signals, offsets):
        if not sig.is_annotation:
            continue
        raw = records[:, off : off + sig.n_samples_per_record].astype("<i2").tobytes()
        for tal in _TAL_SPLIT.split(raw):
            if not tal.strip():
                continue
            fields = tal.split(b"\x14")
            stamp = fields[0]
            if not stamp.startswith((b"+", b"-")):
                continue
            if b"\x15" in stamp:
                onset_b, dur_b = stamp.split(b"\x15", 1)
            else:
                onset_b, dur_b = stamp, b"0"
            try:
                onset = float(onset_b)
                duration = float(dur_b or 0)
            except ValueError:
                continue
            for text in fields[1:]:
                label = text.decode("utf-8", errors="replace").strip()
                if label:                      # the bare timekeeping TAL has no text
                    out.append(Annotation(onset, duration, label))
    out.sort(key=lambda a: a.onset)
    return out


# --------------------------------------------------------------------------- #
# Minimal writer — used to build synthetic fixtures for the test suite
# --------------------------------------------------------------------------- #
def _pad(text: str, width: int) -> bytes:
    return text[:width].ljust(width).encode("ascii", errors="replace")


def write_edf(
    path: str | Path,
    signals: Dict[str, np.ndarray],
    sampling_rate: float,
    record_duration: float = 1.0,
    physical_range: Tuple[float, float] = (-500.0, 500.0),
    annotations: Sequence[Annotation] | None = None,
    patient_id: str = "X X X Synthetic",
    recording_id: str = "Startdate 01-JAN-2026 X X somnoscope",
) -> Path:
    """Write a small EDF+ file. Only what the test fixtures need, not a full writer."""
    path = Path(path)
    spr = int(round(sampling_rate * record_duration))
    lengths = {len(v) for v in signals.values()}
    if len(lengths) != 1:
        raise ValueError("all signals must have the same length")
    n_samples = lengths.pop()
    n_records = n_samples // spr
    if n_records == 0:
        raise ValueError("signal shorter than one data record")

    pmin, pmax = physical_range
    dmin, dmax = -32768, 32767
    scale = (pmax - pmin) / (dmax - dmin)

    ann_bytes: List[bytes] = []
    if annotations is not None:
        per_record: Dict[int, List[Annotation]] = {}
        for a in annotations:
            per_record.setdefault(int(a.onset // record_duration), []).append(a)
        for r in range(n_records):
            tal = b"+%s\x14\x14\x00" % f"{r * record_duration:g}".encode()
            for a in per_record.get(r, []):
                tal += (
                    b"+%s\x15%s\x14%s\x14\x00"
                    % (
                        f"{a.onset:g}".encode(),
                        f"{a.duration:g}".encode(),
                        a.text.encode("utf-8"),
                    )
                )
            ann_bytes.append(tal)
        width = max(len(t) for t in ann_bytes)
        width += width % 2                       # int16 pairs
        ann_bytes = [t.ljust(width, b"\x00") for t in ann_bytes]
        ann_spr = width // 2
    else:
        ann_spr = 0

    labels = list(signals)
    n_signals = len(labels) + (1 if ann_spr else 0)
    header_bytes = 256 * (n_signals + 1)

    with path.open("wb") as fh:
        fh.write(_pad("0", 8))
        fh.write(_pad(patient_id, 80))
        fh.write(_pad(recording_id, 80))
        fh.write(_pad("01.01.26", 8))
        fh.write(_pad("00.00.00", 8))
        fh.write(_pad(str(header_bytes), 8))
        fh.write(_pad("EDF+C", 44))
        fh.write(_pad(str(n_records), 8))
        fh.write(_pad(f"{record_duration:g}", 8))
        fh.write(_pad(str(n_signals), 4))

        all_labels = labels + ([ANNOTATION_LABEL] if ann_spr else [])
        for lab in all_labels:
            fh.write(_pad(lab, 16))
        for lab in all_labels:
            fh.write(_pad("AgAgCl electrode" if lab != ANNOTATION_LABEL else "", 80))
        for lab in all_labels:
            fh.write(_pad("uV" if lab != ANNOTATION_LABEL else "", 8))
        for lab in all_labels:
            fh.write(_pad(f"{pmin:g}" if lab != ANNOTATION_LABEL else "-1", 8))
        for lab in all_labels:
            fh.write(_pad(f"{pmax:g}" if lab != ANNOTATION_LABEL else "1", 8))
        for _ in all_labels:
            fh.write(_pad(str(dmin), 8))
        for _ in all_labels:
            fh.write(_pad(str(dmax), 8))
        for _ in all_labels:
            fh.write(_pad("", 80))
        for lab in all_labels:
            fh.write(_pad(str(ann_spr if lab == ANNOTATION_LABEL else spr), 8))
        for _ in all_labels:
            fh.write(_pad("", 32))

        for r in range(n_records):
            for lab in labels:
                chunk = signals[lab][r * spr : (r + 1) * spr]
                digital = np.clip(np.round((chunk - pmin) / scale + dmin), dmin, dmax)
                fh.write(digital.astype("<i2").tobytes())
            if ann_spr:
                fh.write(ann_bytes[r])
    return path
