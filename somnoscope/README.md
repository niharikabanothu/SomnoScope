# SomnoScope

**Sleep staging from single-channel EEG, with a robustness and spectral-explainability audit attached.**

A dual-stream 1-D CNN performs 5-stage AASM sleep staging from one EEG derivation
(Fpz-Cz, 100 Hz) on Sleep-EDF Expanded, evaluated subject-disjoint. The point of the
repo is not the classifier — it is everything wrapped around it:

- scoring against the **human inter-scorer ceiling** rather than against 100%,
- **wake cropping** so accuracy measures something,
- a **five-degradation robustness sweep** with an amplitude-scaling **shortcut probe**,
- **Grad-CAM projected onto frequency** and checked against the bands the AASM rules
  actually define each stage by.

```bash
pip install -e .
somnoscope selftest                     # full pipeline on synthetic data, ~1 min, no download
bash scripts/download_sleepedf.sh data/sleep-edf 20
somnoscope train --data data/sleep-edf --out runs/main
somnoscope robustness --checkpoint runs/main/cv/fold0.pt --data data/sleep-edf
somnoscope explain    --checkpoint runs/main/cv/fold0.pt --data data/sleep-edf
```

---

## 1. The model: two kernel scales, because sleep has two

The two AASM stages that are hardest to separate are defined by events at opposite
ends of the time scale:

| stage | defining event | timescale | frequency |
|-------|----------------|-----------|-----------|
| N2 | sleep spindles | 0.5–2 s bursts | 11–16 Hz (sigma) |
| N3 | slow-wave activity | 1–2 s per wave, ≥20% of the epoch | 0.5–2 Hz (delta) |

A single receptive field has to compromise between them, so the network runs both:

```
                    ┌─ fine stream    kernel 0.5 s (50 taps @ 100 Hz) ──┐
30 s epoch ─────────┤                                                  ├── concat ── linear ── 5 stages
(1 × 3000)          └─ coarse stream  kernel 4.0 s (400 taps) ─────────┘
```

Kernels are specified in **seconds**, not taps (`kernel_seconds` in
`ModelConfig`), so changing the sampling rate changes the tap count and leaves the
physiology fixed. Each stream ends in an adaptive pool, so the epoch length is not
baked into the weights.

This is the DeepSleepNet representation idea (Supratak et al., 2017). The sequential
BiLSTM stage is deliberately **omitted**: this repo audits what a single epoch
supports, and a temporal context model would let neighbouring epochs paper over
exactly the per-epoch failures the robustness sweep is looking for.

`somnoscope evaluate --ablation` zeroes one stream at a time to check the two scales
are not redundant — N2 should lean on the fine stream, N3 on the coarse one.

## 2. Evaluation: Cohen's kappa against the human ceiling

**Wake cropping.** Sleep Cassette recordings are ~20 h of ambulatory recording, so
raw class balance is **60–70% Wake**. A model that predicts "Wake" unconditionally
scores ~0.65 accuracy and κ ≈ 0.00. Following standard Sleep-EDF practice, only
30 minutes of Wake either side of the sleep period is kept. It is a labelling
decision made before any model sees the data, applied identically to train and test,
and `somnoscope train` prints the before/after class balance every run.

**Cohen's kappa, not accuracy.** Kappa corrects for chance agreement, which is the
only reason the majority-class predictor above scores 0.

**The 0.75–0.80 band.** Published inter-scorer reliability for 5-stage AASM scoring
is roughly **κ = 0.75–0.80** (Danker-Hopfe et al. 2009; Rosenberg & Van Hout 2013).
Ground truth here *is* a human opinion, so that band — not 1.0 — is the practical
ceiling, and every report in this repo is printed against it:

```
=== pooled out-of-fold (all held-out subjects) =========================
  accuracy     <your number>
  macro F1     <your number>
  Cohen kappa  <your number>   95% CI [<lo>, <hi>]

  human inter-scorer band: [0.75, 0.80]
  |########################################|  <%> of the band midpoint
```

(Fill `docs/RESULTS.md` with your actual run — the repo ships the harness, not a
results claim.)

Per-stage F1 is always reported alongside, because the aggregate hides N1 — the
stage where humans agree with each other least (often κ < 0.5 on N1 alone).

**Subject-disjoint everything.** Consecutive epochs from one night are enormously
correlated. Random epoch splits leak the test subject into training and inflate
kappa. `data/splits.py` splits at the *subject* level (both nights of a subject stay
together), and `verify_disjoint()` is called before every fold — it raises rather
than warns.

## 3. Robustness: five degradations, one of them a probe

Clinical PSG uses gelled electrodes on a shielded amplifier. A headband does not.
Each degradation is swept over severity `0.0 → 1.0` on held-out epochs, with **no
retraining**:

| degradation | models | severity 1.0 |
|---|---|---|
| `amplitude_scale` | gain / impedance mismatch | up to ×4 / ÷4 |
| `sensor_noise` | dry-electrode noise | 0 dB SNR |
| `motion_drift` | head-movement baseline wander | 0.05–0.5 Hz at 100% of signal RMS |
| `powerline_hum` | mains pickup | 50 Hz at 50% of RMS |
| `bandwidth_loss` | cheap analogue front end | 8 Hz low-pass |

### The amplitude shortcut probe

Multiplying an EEG epoch by a constant **does not change its sleep stage** — stage is
defined by the *relative* spectral content, and a gain change is exactly what a
different electrode impedance produces. So a correct model should be invariant here.
Any damage under gain change is not "the task got harder"; it is evidence the model
learned *"big numbers mean N3"* instead of *"slow waves mean N3"*.

```
shortcut index = mean κ drop under amplitude scaling
                 ─────────────────────────────────────
                 mean κ drop under sensor noise
```

- `< 0.15` — amplitude-invariant, no shortcut
- `0.5–1.0` — substantial amplitude dependence
- `≥ 1.0` — rescaling the signal is as damaging as burying it in noise

Two design decisions make this a real test rather than a formality:

1. **`amplitude_scale` is excluded from training augmentation** (see
   `TrainConfig.augment_kinds`). Augmenting with the probe would train the
   vulnerability away and leave the probe unable to detect anything.
2. **Normalisation is a flag, so the fix is falsifiable.** `--norm per_epoch`
   z-scores each 30 s window and should drive the index to ~0; `--norm global`
   preserves absolute amplitude and should not. Running both turns *"we normalised
   the data"* from a methods-section sentence into a measured number.

The sweep also reports per-stage F1 change, which yields directional predictions to
check rather than assume: `bandwidth_loss` strips sigma and should cost **N2** most,
while `motion_drift` contaminates delta and should cost **N3** most.

## 4. Explainability: Grad-CAM projected onto frequency

A saliency heatmap saying *"the model looked at seconds 12–14"* is not checkable
against anything. The AASM rules are **frequency** statements, so this repo moves the
explanation onto the axis where a pre-registered right answer exists:

| stage | clinically defining band |
|---|---|
| W | alpha 8–12 Hz |
| N1 | theta 4–8 Hz |
| N2 | **sigma 12–16 Hz** (spindles) |
| N3 | **delta 0.5–4 Hz** (slow waves) |
| REM | theta 4–8 Hz |

**Method.** Compute a 1-D Grad-CAM per stream → take the STFT of the epoch → weight
each time frame by its CAM value → sum. The result is *the spectrum of the parts of
the signal the model actually used*, and it is compared against `EXPECTED_BAND`,
fixed before training. The output is a **hit rate**, a number that can be wrong, not
a gallery of heatmaps.

**Contrast against the epoch's own spectrum.** If N3 epochs are 60% delta by raw
power, a model attending to delta proves nothing — there is nothing else to attend
to. `band_attribution(..., contrast=True)` divides the attributed spectrum by the
unweighted one, so the audit scores *over*-weighting, and prints the raw data profile
next to it so the comparison is visible rather than assumed.

**Honest caveat, reported by the tool itself:** W and REM are both marked by theta on
a frontal derivation and are separated clinically by EOG and chin EMG, which a
single-channel Fpz-Cz montage does not have. The headline hit rate therefore covers
N1/N2/N3 only.

The Grad-CAM implementation takes gradients via `torch.autograd.grad` against the
exact tensor each stream stored, not via a backward hook on the last `Conv1d` — those
are different tensors (post- vs pre-BatchNorm/GELU), and mismatching them is a quiet
way to produce saliency that looks plausible and means nothing.

---

## Repository layout

```
src/somnoscope/
  data/
    edf.py            dependency-free EDF/EDF+ reader (+ a writer for fixtures)
    sleepedf.py       AASM 5-stage mapping, epoching, wake cropping, synthetic data
    splits.py         subject-disjoint splits, with a leak assertion
    transforms.py     the five degradations + normalisation modes
  models/
    dual_stream_cnn.py  the 0.5 s / 4 s architecture, class-balanced loss
  explain/
    bandpower.py      Welch band power, EXPECTED_BAND (the clinical reference)
    gradcam.py        1-D Grad-CAM + CAM→frequency projection
    audit.py          the scored spectral audit
  metrics.py          Cohen's kappa vs the 0.75–0.80 human band, per-stage report
  train.py            subject-disjoint CV, kappa-selected checkpoints
  robustness.py       the severity sweep + shortcut index
  evaluate.py         checkpoint scoring, stream ablation, confusion plots
  cli.py              somnoscope {selftest,train,evaluate,robustness,explain}
tests/                pytest suite; torch tests skip cleanly if torch is absent
configs/default.yaml  every knob, annotated with why it is set that way
```

### Why the EDF reader is hand-written

Reading a well-specified 256-byte header does not need `mne` or `pyedflib`. Writing
it out keeps the install light and, more usefully, makes the physical-unit scaling
visible in the code — which matters here, because the amplitude shortcut probe is
only meaningful if you know the signal really is in microvolts before you rescale it.
It also comes with a small EDF *writer*, which is what lets `somnoscope selftest`
build a synthetic night and exercise the whole pipeline with no download.

## Data

Sleep-EDF Expanded (Kemp et al. 2000; PhysioNet, Goldberger et al. 2000) is **not**
redistributed here. `bash scripts/download_sleepedf.sh data/sleep-edf [n_subjects]`
fetches it; see [`docs/DATA.md`](docs/DATA.md) for the licence and citations. The
pipeline runs on 10 subjects; numbers are only comparable to published work on the
full Sleep Cassette set.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite covers EDF round-tripping, the R&K→AASM merge, wake cropping, split
disjointness (including a test that the leak assertion fires), each degradation's
severity-0 no-op and reproducibility, the CAM→frequency projection recovering the
band it was pointed at, and the fact that a majority-class predictor gets high
accuracy and κ = 0. Torch-dependent tests skip cleanly when torch is not installed.

## Not claimed

- No temporal/sequence model, so per-epoch numbers are below published
  DeepSleepNet-class results that use inter-epoch context.
- Sleep-EDF is healthy-ish adults on a specific amplifier; the degradations are
  *simulated* wearable conditions, not recordings from a wearable.
- The spectral audit shows attention is *consistent with* the AASM bands. Consistency
  is not causation, and Grad-CAM has known limitations as an attribution method.

## References

- Kemp et al. (2000), *Analysis of a sleep-dependent neuronal feedback loop*, IEEE TBME — Sleep-EDF.
- Goldberger et al. (2000), *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation.
- Supratak et al. (2017), *DeepSleepNet*, IEEE TNSRE — dual-kernel representation learning.
- Selvaraju et al. (2017), *Grad-CAM*, ICCV.
- Danker-Hopfe et al. (2009), *Interrater reliability... AASM vs Rechtschaffen & Kales*, J Sleep Res.
- Rosenberg & Van Hout (2013), *The AASM inter-scorer reliability program*, J Clin Sleep Med.
- Cui et al. (2019), *Class-balanced loss based on effective number of samples*, CVPR.
- Iber et al. (2007), *The AASM Manual for the Scoring of Sleep and Associated Events*.

## Licence

MIT for the code (see `LICENSE`). The Sleep-EDF data carries its own PhysioNet terms.
