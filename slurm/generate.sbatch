#!/bin/bash
#SBATCH -J kite-data
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=5G
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/%x-%j.out
# kite_data.generate on a CPU node. Submit from this repo's root:
#   sbatch slurm/generate.sbatch <tag> <generate args...>
# How the current kite_code data was made (about 8 min over 3 nodes, 4 workers each, ~590 MB peak per worker):
#   sbatch slurm/generate.sbatch main-train --splits train                                       (1000 per hand)
#   sbatch slurm/generate.sbatch main-evals --splits val test_geometry test_joint_count test_finger_subset test_combined
#   sbatch slurm/generate.sbatch scale      (train 4000 and 16000 per hand on data/scaling_hands_100.txt)
# Output: data/examples/kite_code[...] (data/ is gitignored; symlink it elsewhere for large runs). Each run's
# summary.json is copied to summary.<tag>.<time>.json because a run rewrites the split's summary.
# Set PY to a python with this package installed (default: python on PATH).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PY=${PY:-python}
TAG=$1; shift
echo "host=$(hostname) start=$(date -Is) restart_count=${SLURM_RESTART_COUNT:-0} tag=$TAG args=$*"
run() {  # run <out> <args...>
  local out=$1; shift
  time "$PY" -m kite_data.generate --regime kite_code --out "$out" --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
  cp "$out/summary.json" "$out/summary.$TAG.$(date +%s).json"; du -sh "$out"
}
if [ "$TAG" = scale ]; then
  run data/examples/kite_code_s4k  --splits train --hands-list data/scaling_hands_100.txt --per-hand train=4000
  run data/examples/kite_code_s16k --splits train --hands-list data/scaling_hands_100.txt --per-hand train=16000
else
  run data/examples/kite_code "$@"
fi
echo "end=$(date -Is)"
