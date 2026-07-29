# Data: Sleep-EDF Expanded

This repository does **not** redistribute any recordings. Fetch them from PhysioNet.

```bash
bash scripts/download_sleepedf.sh data/sleep-edf        # full Sleep Cassette, ~8 GB
bash scripts/download_sleepedf.sh data/sleep-edf 20     # first 20 records, for a quick start
```

## What you get

The **Sleep Cassette (SC\*)** subset is 153 whole-night ambulatory PSGs from 78
healthy subjects aged 25–101, recorded in 1987–1991 in a study of age effects on
sleep. Most subjects contribute two consecutive nights.

Each night is two files:

```
SC4001E0-PSG.edf         signals
SC4001EC-Hypnogram.edf   expert scoring, as EDF+ annotations
```

Record ids encode the subject: `SC4` + two digits of subject + one digit of night.
`somnoscope` groups nights by subject (`subject_id()` in `data/sleepedf.py`) so both
of a subject's nights always land on the same side of a split.

| property | value |
|---|---|
| channel used here | `EEG Fpz-Cz` |
| sampling rate | 100 Hz |
| scoring epoch | 30 s (3000 samples) |
| scoring standard | Rechtschaffen & Kales (mapped to AASM 5-stage) |
| other channels present | `EEG Pz-Oz`, `EOG horizontal`, `Resp oro-nasal`, `EMG submental`, `Temp rectal`, `Event marker` |

There is also a **Sleep Telemetry (ST\*)** subset (22 subjects, temazepam study).
`find_pairs()` will pick it up if you download it into the same folder; it is a
different population and a different protocol, so mixing the two without saying so
is not recommended.

## Label mapping

Sleep-EDF is scored under R&K, which splits deep sleep into S3 and S4. AASM merges
them:

| annotation text | AASM stage | index |
|---|---|---|
| `Sleep stage W` | W | 0 |
| `Sleep stage 1` | N1 | 1 |
| `Sleep stage 2` | N2 | 2 |
| `Sleep stage 3` | N3 | 3 |
| `Sleep stage 4` | N3 | 3 |
| `Sleep stage R` | REM | 4 |
| `Sleep stage ?` | *dropped* | −1 |
| `Movement time` | *dropped* | −1 |

Unscorable epochs are **dropped, not folded into Wake**. Folding them into Wake is
common and inflates the majority class further.

## Wake cropping

The recordings run ~20 hours, including long periods of the subject awake and
walking around. Raw class balance is 60–70% Wake. `crop_margin_min` (default 30)
keeps 30 minutes of Wake either side of the first/last non-Wake epoch and discards
the rest.

This is standard Sleep-EDF practice and it is also the single biggest lever on the
headline numbers, so:

- it is applied identically to train, validation and test;
- it happens before any model sees the data;
- `somnoscope train` prints the before/after class balance on every run;
- `--crop-margin-min 10000` effectively disables it, if you want to see what the
  uncropped numbers look like.

## Licence and citation

Sleep-EDF Expanded is released under the **Open Data Commons Attribution License
v1.0**. If you publish anything using it, cite both the dataset and PhysioNet:

> Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. *Analysis of a
> sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.*
> IEEE Transactions on Biomedical Engineering, 47(9):1185–1194, 2000.

> Goldberger AL, Amaral LAN, Glass L, et al. *PhysioBank, PhysioToolkit, and
> PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.*
> Circulation, 101(23):e215–e220, 2000.

Dataset page: <https://physionet.org/content/sleep-edfx/1.0.0/>

## Working without the download

`somnoscope selftest` generates synthetic nights whose spectra follow the AASM stage
definitions (delta-dominant N3, discrete sigma bursts in N2, alpha in W, and so on),
then runs the whole pipeline on them. It verifies the plumbing in about a minute and
tells you nothing about real sleep staging — the printed numbers are not results.
