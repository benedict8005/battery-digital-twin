#!/bin/sh
# Equal-budget baseline on the real NASA data: legacy 1.4 Ah capacity floor (19 training /
# 5 held-out batteries) versus the corrected dynamic floor (24 / 6), several seeds each, then
# docs/RESULTS.md via make_results_table.py.
#
# Usage: scripts/run_baseline.sh RAW_DIR [RUNS_DIR] [ITERATIONS] [LBFGS_ITERATIONS] [SEEDS...]
#   RAW_DIR    folder with metadata.csv + data/ (from `battery-pinn convert-mat`)
#   RUNS_DIR   where to put runs/{legacy,dynamic}/{data,artifacts} (default: runs)
#   ITERATIONS Adam iterations per seed (default: 1000 -- the reduced budget used for the
#              committed baseline; the notebook's full budget is 15000 + 2000 L-BFGS)
set -eu
RAW_DIR=${1:?raw dataset folder}
RUNS_DIR=${2:-runs}
ITERATIONS=${3:-1000}
LBFGS=${4:-0}
shift $(( $# < 4 ? $# : 4 ))
SEEDS=${*:-0 1 2}

cd "$(dirname "$0")/.."
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

python -m battery_pinn prepare-data --raw "$RAW_DIR" --out "$RUNS_DIR/legacy/data" \
    --min-capacity-fraction none --allow-missing-held-out
python -m battery_pinn prepare-data --raw "$RAW_DIR" --out "$RUNS_DIR/dynamic/data"

for cfg in legacy dynamic; do
  echo "=== $cfg: start $(date -u +%FT%TZ) ==="
  python -m battery_pinn train --data "$RUNS_DIR/$cfg/data/train.csv" \
      --out "$RUNS_DIR/$cfg/artifacts" --seeds $SEEDS --iterations "$ITERATIONS" \
      --lbfgs-iterations "$LBFGS" --num-domain 5000 --num-boundary 1000 --device auto
  echo "=== $cfg: done $(date -u +%FT%TZ) ==="
done

# docs/baseline_protocol.md describes the committed run; docs/baseline_notes.md interprets its
# numbers -- rewrite the notes after a rerun with a different budget.
NOTES=""
[ -f docs/baseline_notes.md ] && NOTES="--notes-file docs/baseline_notes.md"
python scripts/make_results_table.py \
    --run legacy "$RUNS_DIR/legacy/artifacts" "$RUNS_DIR/legacy/data" \
    --run dynamic "$RUNS_DIR/dynamic/artifacts" "$RUNS_DIR/dynamic/data" \
    --out docs/RESULTS.md --json docs/RESULTS.json $NOTES \
    --protocol "Adam ${ITERATIONS} iterations, L-BFGS ${LBFGS}, num_domain 5000, num_boundary 1000, seeds ${SEEDS}; all other settings as in battery_pinn/config.py. See docs/baseline_protocol.md for the committed baseline."
