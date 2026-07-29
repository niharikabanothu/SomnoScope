#!/usr/bin/env bash
# Download the Sleep-EDF Expanded database (Sleep Cassette subset) from PhysioNet.
#
#   bash scripts/download_sleepedf.sh [target_dir] [n_subjects]
#
# The full Sleep Cassette set is 78 subjects / ~8 GB. Pass n_subjects to grab a
# smaller slice first -- the pipeline runs fine on 10 subjects and the numbers
# are only meaningfully comparable to published work at the full set.
#
# Data licence: Open Data Commons Attribution v1.0. Cite Kemp et al. (2000) and
# Goldberger et al. (2000); see docs/DATA.md.

set -euo pipefail

TARGET="${1:-data/sleep-edf}"
N_SUBJECTS="${2:-0}"
BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

mkdir -p "$TARGET"

if ! command -v wget >/dev/null 2>&1; then
  echo "error: wget is required (brew install wget / apt install wget)" >&2
  exit 1
fi

echo "downloading Sleep-EDF Expanded (sleep-cassette) into $TARGET"

if [[ "$N_SUBJECTS" -gt 0 ]]; then
  echo "fetching the file index ..."
  wget -q -O "$TARGET/.index.html" "$BASE"
  # Record ids look like SC4001E0-PSG.edf / SC4001EC-Hypnogram.edf.
  RECORDS=$(grep -o 'SC4[0-9]\{3\}E[0-9A-Z]-PSG\.edf' "$TARGET/.index.html" \
            | sort -u | head -n "$N_SUBJECTS")
  for psg in $RECORDS; do
    stem="${psg%%-PSG.edf}"
    prefix="${stem:0:7}"
    echo "  $prefix"
    wget -q -c -P "$TARGET" "${BASE}${psg}"
    hyp=$(grep -o "${prefix}[0-9A-Z]-Hypnogram\.edf" "$TARGET/.index.html" | head -n1)
    [[ -n "$hyp" ]] && wget -q -c -P "$TARGET" "${BASE}${hyp}"
  done
  rm -f "$TARGET/.index.html"
else
  wget -r -N -c -np -nH --cut-dirs=4 -R "index.html*" -P "$TARGET" "$BASE"
fi

echo
echo "done. files in $TARGET:"
find "$TARGET" -name '*.edf' | wc -l
echo
echo "next:  somnoscope train --data $TARGET"
