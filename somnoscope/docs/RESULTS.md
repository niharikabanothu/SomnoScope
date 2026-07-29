# Results

> **Template.** This repository ships the harness, not a results claim. Run the
> commands below on your own download and paste the printed blocks in. Every number
> below is a placeholder.

## Setup

| | |
|---|---|
| data | Sleep-EDF Expanded, Sleep Cassette, `EEG Fpz-Cz` @ 100 Hz |
| subjects / nights | `<n>` / `<n>` |
| epochs after wake cropping | `<n>` (from `<n>` before) |
| split | subject-disjoint 5-fold CV |
| normalisation | `per_epoch` |
| hardware / runtime | `<gpu>`, `<mins>` per fold |

```bash
somnoscope train --data data/sleep-edf --out runs/main --folds 5
```

### Class balance, before → after wake cropping

| stage | before | after |
|---|---|---|
| W | `<%>` | `<%>` |
| N1 | `<%>` | `<%>` |
| N2 | `<%>` | `<%>` |
| N3 | `<%>` | `<%>` |
| REM | `<%>` | `<%>` |

## 1. Staging performance

| metric | value |
|---|---|
| Cohen's κ (pooled out-of-fold) | `<κ>` |
| κ across folds | `<mean> ± <std>` |
| accuracy | `<acc>` |
| macro F1 | `<f1>` |
| human inter-scorer band | 0.75 – 0.80 |

Per-stage F1:

| W | N1 | N2 | N3 | REM |
|---|---|---|---|---|
| `<f1>` | `<f1>` | `<f1>` | `<f1>` | `<f1>` |

Paste the `=== pooled out-of-fold ===` block from the run here, confusion matrix
included. N1 is expected to be the weakest stage; note whether the confusion is
mostly N1↔N2 and N1↔REM, which is also where humans disagree.

## 2. Stream ablation

```bash
somnoscope evaluate --checkpoint runs/main/cv/fold0.pt --data data/sleep-edf --ablation
```

| config | κ | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| both | | | | | | |
| fine only | | | | | | |
| coarse only | | | | | | |

**Prediction to check:** N2 F1 should fall most when the *fine* (0.5 s, spindle)
stream is removed; N3 F1 should fall most when the *coarse* (4 s, slow-wave) stream
is removed. If both ablations cost the same everywhere, the second scale is
redundant and the architecture is not doing what it claims.

## 3. Robustness sweep

```bash
somnoscope robustness --checkpoint runs/main/cv/fold0.pt --data data/sleep-edf
```

κ by severity:

| degradation | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | retained |
|---|---|---|---|---|---|---|
| amplitude_scale | | | | | | |
| sensor_noise | | | | | | |
| motion_drift | | | | | | |
| powerline_hum | | | | | | |
| bandwidth_loss | | | | | | |

### Amplitude shortcut probe

| | `--norm per_epoch` | `--norm global` |
|---|---|---|
| mean κ drop under amplitude scaling | | |
| mean κ drop under sensor noise | | |
| **shortcut index** | | |
| verdict | | |

Run the sweep under both normalisations. The expected story is that `per_epoch`
z-scoring drives the index to ~0 while `global` leaves the model amplitude-dependent
— which is what turns "we normalised the data" into a measurement. If `per_epoch`
does *not* close it, that is the more interesting result and worth chasing.

### Per-stage sensitivity at maximum severity

| degradation | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| bandwidth_loss | | | | | |
| motion_drift | | | | | |

**Predictions to check:** `bandwidth_loss` destroys sigma, so **N2** should suffer
most. `motion_drift` contaminates delta, so **N3** should suffer most. Record
whichever way it comes out, including if it contradicts the design story.

## 4. Spectral explainability audit

```bash
somnoscope explain --checkpoint runs/main/cv/fold0.pt --data data/sleep-edf
```

| stage | expected band | fine stream attended | coarse stream attended | hit |
|---|---|---|---|---|
| W | alpha | | | |
| N1 | theta | | | |
| N2 | sigma | | | |
| N3 | delta | | | |
| REM | theta | | | |

| | |
|---|---|
| hit rate, N1/N2/N3 | `<x/3>` |
| hit rate, all stages | `<x/5>` |

**Stream specialisation.** Does sigma attribution on N2 lead in the fine stream, and
delta attribution on N3 in the coarse stream? Quote the tool's line.

**Caveat carried through from the tool:** W and REM are both theta-marked on a
frontal derivation and are separated clinically by EOG and chin EMG, which this
montage lacks. Treat misses there as expected, not as evidence of a broken model.

## 5. What this does not show

- No inter-epoch context model, so these numbers sit below published
  DeepSleepNet-class results that use temporal context.
- Degradations are *simulated* wearable conditions, not wearable recordings.
- The audit shows attention is consistent with the AASM bands; consistency is not
  causation, and Grad-CAM has known attribution limitations.
