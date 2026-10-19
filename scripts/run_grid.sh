#!/bin/sh
# Physics-option grid, fixed-A_sei retrains and the ten-model abuse ensemble (CHANGELOG item 27).
# Expects RUNS_DIR/dynamic/data/train.csv (from run_baseline.sh) and RUNS_DIR/seeds5 (from
# run_validation.sh, seeds 0-4 of the baseline protocol). Every training run uses `train`'s
# defaults, i.e. the validated fast protocol.
#
# Usage: scripts/run_grid.sh [RUNS_DIR]
set -eu
RUNS_DIR=${1:-runs}
cd "$(dirname "$0")/.."
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
DATA="$RUNS_DIR/dynamic/data/train.csv"
run() { name=$1; shift; python -m battery_pinn train --data "$DATA" --out "$RUNS_DIR/$name/artifacts" "$@"; }

run grid_newton  --seeds 0 1 2 --newton-cooling
run grid_thermal --seeds 0 1 2 --learn-thermal
run grid_both    --seeds 0 1 2 --newton-cooling --learn-thermal
run seeds6_10    --seeds 5 6 7 8 9
run fixed_12     --seeds 0 1 2 --fix-log-a-sei 12
run fixed_18     --seeds 0 1 2 --fix-log-a-sei 18
run fixed_21     --seeds 0 1 2 --fix-log-a-sei 21

# Ten-model ensemble = seeds 0-4 (seeds5) + seeds 5-9 (seeds6_10), gathered in one folder.
mkdir -p "$RUNS_DIR/ensemble10/artifacts"
for k in 0 1 2 3 4; do ln -sf "$(pwd)/$RUNS_DIR/seeds5/artifacts/battery_pinn_model_seed$k.pt" "$RUNS_DIR/ensemble10/artifacts/"; done
for k in 5 6 7 8 9; do ln -sf "$(pwd)/$RUNS_DIR/seeds6_10/artifacts/battery_pinn_model_seed$k.pt" "$RUNS_DIR/ensemble10/artifacts/"; done
python -m battery_pinn abuse-test --checkpoints "$RUNS_DIR/ensemble10/artifacts" --data "$DATA" --min-models 10

python scripts/make_results_table.py \
    --run legacy "$RUNS_DIR/legacy/artifacts" "$RUNS_DIR/legacy/data" \
    --run dynamic "$RUNS_DIR/dynamic/artifacts" "$RUNS_DIR/dynamic/data" \
    --run dynamic-subset "$RUNS_DIR/dynamic_subset/artifacts" "$RUNS_DIR/dynamic/data" \
    --run seeds5 "$RUNS_DIR/seeds5/artifacts" "$RUNS_DIR/dynamic/data" \
    --run ensemble10 "$RUNS_DIR/ensemble10/artifacts" "$RUNS_DIR/dynamic/data" \
    --run fullanchor "$RUNS_DIR/fullanchor/artifacts" "$RUNS_DIR/dynamic/data" \
    --run fullbudget "$RUNS_DIR/fullbudget/artifacts" "$RUNS_DIR/dynamic/data" \
    --run grid_newton "$RUNS_DIR/grid_newton/artifacts" "$RUNS_DIR/dynamic/data" \
    --run grid_thermal "$RUNS_DIR/grid_thermal/artifacts" "$RUNS_DIR/dynamic/data" \
    --run grid_both "$RUNS_DIR/grid_both/artifacts" "$RUNS_DIR/dynamic/data" \
    --run fixed_12 "$RUNS_DIR/fixed_12/artifacts" "$RUNS_DIR/dynamic/data" \
    --run fixed_18 "$RUNS_DIR/fixed_18/artifacts" "$RUNS_DIR/dynamic/data" \
    --run fixed_21 "$RUNS_DIR/fixed_21/artifacts" "$RUNS_DIR/dynamic/data" \
    --grid seeds5 grid_newton grid_thermal grid_both \
    --compare seeds5 fullanchor --compare seeds5 fullbudget \
    --compare seeds5 fixed_12 --compare seeds5 fixed_18 --compare seeds5 fixed_21 \
    --out docs/RESULTS.md --json docs/RESULTS.json \
    --protocol-file docs/baseline_protocol.md --notes-file docs/baseline_notes.md
