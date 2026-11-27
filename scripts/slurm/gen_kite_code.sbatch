#!/bin/bash
#SBATCH -J gen-kite-code
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=5G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
# The kite_code example data the decoder trains and evaluates on (regime kite_code = KITE's released recipe, N_VALIDATION=15
# splits). This is the command that produced data/examples/kite_code (originally run 2026-11-13 as "T10"; ported here
# 2026-11-27 with paths from the environment). Run from this repo's root:
#   sbatch scripts/slurm/gen_kite_code.sbatch main-train --splits train                    (1,000 records per hand)
#   sbatch scripts/slurm/gen_kite_code.sbatch main-evals --splits val test_finger_subset test_combined test_joint_count test_geometry
#                                                                                            (val 1,000, test 2,000 per hand)
#   sbatch scripts/slurm/gen_kite_code.sbatch scale                                          (scaling study: 100 hands x 4k / 16k)
# Hands are built on demand (~590 MB peak per worker; 4 workers fit in 5 GB). index.csv / summary.json in an output folder are
# rewritten by each run, so each run's summary is also copied to summary.<tag>.<time>.json.
# Environment: KITE_DATA_PY = python with kite_data installed (default: python).
set -euo pipefail
PY=${KITE_DATA_PY:-python}
mkdir -p logs
TAG=$1; shift
echo "host=$(hostname) start=$(date -Is) restart_count=${SLURM_RESTART_COUNT:-0} tag=$TAG args=$*"
run() {  # run <out> <args...>
  local out=$1; shift
  time "$PY" -m kite_data.generate --regime kite_code --out "$out" --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
  cp "$out/summary.json" "$out/summary.$TAG.$(date +%s).json"; du -sh "$out"
}
if [ "$TAG" = scale ]; then
  run data/examples/kite_code_s4k  --splits train --hands-list scripts/slurm/scaling_hands_100.txt --per-hand train=4000
  run data/examples/kite_code_s16k --splits train --hands-list scripts/slurm/scaling_hands_100.txt --per-hand train=16000
else
  run data/examples/kite_code "$@"
fi
echo "end=$(date -Is)"
