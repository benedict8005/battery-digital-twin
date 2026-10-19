#!/bin/sh
# Validation of the local pipeline on the corrected 24/6 tables (CHANGELOG item 26):
#   seeds5     5 seeds at the converged budget (2000 Adam + 200 L-BFGS, --anchor-subset 2000)
#   fullanchor 1 seed, same protocol but anchors = all observation points (validates the subset)
#   fullbudget 1 seed, the notebook's full budget (15000 Adam + 2000 L-BFGS, 5000/1000 points)
#              with --anchor-subset 2000 (the closest feasible match to the Databricks protocol)
# then docs/RESULTS.md with the earlier baseline runs and explicit comparison tables.
#
# Usage: scripts/run_validation.sh [RUNS_DIR]   (expects RUNS_DIR/dynamic/data from run_baseline.sh)
set -eu
RUNS_DIR=${1:-runs}
cd "$(dirname "$0")/.."
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
DATA="$RUNS_DIR/dynamic/data/train.csv"
COMMON="--data $DATA --device auto --num-domain 1000 --num-boundary 200 --display-every 100"

python -m battery_pinn train $COMMON --out "$RUNS_DIR/seeds5/artifacts" --seeds 0 1 2 3 4 \
    --iterations 2000 --lbfgs-iterations 200 --anchor-subset 2000
python -m battery_pinn train $COMMON --out "$RUNS_DIR/fullanchor/artifacts" --seeds 0 \
    --iterations 2000 --lbfgs-iterations 200
python -m battery_pinn train --data "$DATA" --device auto --out "$RUNS_DIR/fullbudget/artifacts" --seeds 0 \
    --iterations 15000 --lbfgs-iterations 2000 --num-domain 5000 --num-boundary 1000 \
    --anchor-subset 2000 --display-every 500

python scripts/make_results_table.py \
    --run legacy "$RUNS_DIR/legacy/artifacts" "$RUNS_DIR/legacy/data" \
    --run dynamic "$RUNS_DIR/dynamic/artifacts" "$RUNS_DIR/dynamic/data" \
    --run dynamic-subset "$RUNS_DIR/dynamic_subset/artifacts" "$RUNS_DIR/dynamic/data" \
    --run seeds5 "$RUNS_DIR/seeds5/artifacts" "$RUNS_DIR/dynamic/data" \
    --run fullanchor "$RUNS_DIR/fullanchor/artifacts" "$RUNS_DIR/dynamic/data" \
    --run fullbudget "$RUNS_DIR/fullbudget/artifacts" "$RUNS_DIR/dynamic/data" \
    --compare seeds5 fullanchor --compare seeds5 fullbudget \
    --out docs/RESULTS.md --json docs/RESULTS.json \
    --protocol-file docs/baseline_protocol.md --notes-file docs/baseline_notes.md
