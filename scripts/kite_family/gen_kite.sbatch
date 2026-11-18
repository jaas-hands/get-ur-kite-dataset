#!/bin/bash
#SBATCH -J kite-family-data
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o slurm/logs/%x-%j.out
# KITE-native records for KITE's own embodiments (Allegro, Barrett), with KITE's sampler, on CPU.
# Submit from this repo's root:  sbatch scripts/kite_family/gen_kite.sbatch [OUT_DIR] [N_TRAIN] [N_VAL]
#   OUT_DIR default: $GETKITE_DATA/multifamily_pilot (GETKITE_DATA default: ../getkite_data), outside the repo.
#   Seeds: train 101, val 202 (as used for the multi-family pilot).
# Needs KITE_PY: a python with torch + scripts/kite_family/requirements.txt, and KITE_ROOT (see gen_kite_family.py).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PY=${KITE_PY:-python}
OUT=${1:-${GETKITE_DATA:-$(cd .. && pwd)/getkite_data}/multifamily_pilot}; NTR=${2:-2000}; NVA=${3:-500}
mkdir -p "$OUT"
echo "host=$(hostname -s) start=$(date -Is) out=$OUT n_train=$NTR n_val=$NVA"
for R in allegro_right barrett; do
  "$PY" scripts/kite_family/gen_kite_family.py --robot $R --n "$NTR" --seed 101 --out "$OUT/${R}_train.npz" 2>&1 | tail -1
  "$PY" scripts/kite_family/gen_kite_family.py --robot $R --n "$NVA" --seed 202 --out "$OUT/${R}_val.npz" 2>&1 | tail -1
done
echo "end=$(date -Is)"
