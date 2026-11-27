#!/bin/bash
#SBATCH -J gen-10x
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=5G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# Extra independent-goal training pool used by the decoder's CF-3e recipe (originally generated 2026-11-15; ported here
# 2026-11-27 with the same seeds and counts, paths now from the environment). New draws only; original pools untouched.
#   Allegro / Barrett: KITE's own sampler (scripts/kite_family/gen_kite_family.py), seed 303 (train used 101, val 202),
#                      18,000 each  -> + original 2,000 = 20,000 per family
#   LEAP:              kite_data kite_code regime, seed 1 (original files seed 0), the pilot's training hands x 1,000
# Environment (set before sbatch, e.g. in your shell profile):
#   KITE_ROOT     KITE checkout (assets/, checkpoints/)                     required
#   KITE_GEN_PY   python env able to import KITE's sampler                 required
#   KITE_DATA_PY  python env with kite_data installed (this repo)          default: python
#   OUT           output folder (the decoder reads it as $GETKITE_DATA/multifamily_10x)
#   HANDS_LIST    LEAP hand ids, one per line (the pilot's training hands)  default: $OUT/leap_train_hands.txt
# Run from this repo's root:  OUT=... KITE_ROOT=... KITE_GEN_PY=... sbatch scripts/kite_family/gen_10x.sbatch
set -euo pipefail
: "${KITE_ROOT:?set KITE_ROOT}" "${KITE_GEN_PY:?set KITE_GEN_PY}" "${OUT:?set OUT}"
KITE_DATA_PY=${KITE_DATA_PY:-python}
HANDS_LIST=${HANDS_LIST:-$OUT/leap_train_hands.txt}
mkdir -p "$OUT" logs
echo "host=$(hostname) start=$(date -Is)"
export KITE_ROOT
for R in allegro_right barrett; do
  "$KITE_GEN_PY" scripts/kite_family/gen_kite_family.py --robot $R --n 18000 --seed 303 --out "$OUT/${R}_train_extra_s303.npz" 2>&1 | tail -1 | cut -c1-200
done
time "$KITE_DATA_PY" -m kite_data.generate --regime kite_code --seed 1 --splits train \
  --hands-list "$HANDS_LIST" --per-hand train=1000 --out "$OUT/leap_kite_code_s1" --workers 4
echo "DONE $(date -Is)"
