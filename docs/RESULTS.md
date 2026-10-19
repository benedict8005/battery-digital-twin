# Baseline results on the NASA battery data set

Generated 2026-10-16 09:14 UTC by `scripts/make_results_table.py`. Temperature errors are the surface-temperature MAE / RMSE in °C against the measured values. Core temperature is **unvalidated** (inferred through the physics, never compared with an internal sensor). The verdict uses in-domain sweep points only, i.e. it **excludes ambient > 45 °C**; it is a heuristic, not a certified safety result.

## Protocol

Equal-budget comparison of the two capacity floors on NASA's battery data set (`.mat` release
converted with `battery-pinn convert-mat`), produced by `scripts/run_baseline.sh`:

- **legacy**: `prepare-data --min-capacity-fraction none --allow-missing-held-out` -- the original
  absolute 1.4 Ah floor, which leaves 19 training and 5 held-out batteries (B0056 has no data).
- **dynamic**: `prepare-data` defaults -- floor = min(1.4 Ah, 70 % of the battery's best cycle),
  24 training and 6 held-out batteries (CHANGELOG item 22).
- Training: Adam, 1000 iterations, learning rate 1e-4, **no L-BFGS**, `num_domain` 5000,
  `num_boundary` 1000, seeds 0 1 2, CPU (4 threads), all other settings as in
  `battery_pinn/config.py`. This is a **reduced budget** -- the notebook's protocol is 15000 Adam
  + 2000 L-BFGS iterations, about 10 h per seed on this machine because the PDE residual is
  evaluated at every observation anchor (54k / 86k rows). Both runs get the same budget, so the
  comparison is fair, but every model here is under-trained and the absolute errors are not the
  model's final quality. `scripts/run_baseline.sh RAW_DIR runs 15000 2000` reruns at full budget.

**dynamic-subset** (added after the two runs above, CHANGELOG item 25): the same 24 / 6 tables,
trained with the opt-in anchor subset so that it converges on a CPU:

    python -m battery_pinn train --data runs/dynamic/data/final_pinn_training_data.csv \
        --out runs/dynamic_subset/artifacts --seeds 0 1 2 --iterations 2000 --lbfgs-iterations 200 \
        --num-domain 1000 --num-boundary 200 --anchor-subset 2000 --a-sei-init-spread 1.0 \
        --display-every 100 --device cpu

i.e. 2000 observation points (of 85900) as PDE anchors and data-fit points, 2000 Adam + 200
L-BFGS iterations, and the three seeds' `log10(A_sei)` started at 21.885 / 22.885 / 23.885
(the recommended 22.885 +- 1 decade) for the identifiability check. About 6-15 min per seed on
the 4-core CPU. Evaluation uses every row of both tables, as for the other runs.

**Validation runs** (CHANGELOG item 26, `scripts/run_validation.sh`), all on the 24 / 6 tables:

- **seeds5**: five seeds (0-4) at the converged budget of `dynamic-subset` (2000 Adam + 200
  L-BFGS, `num_domain` 1000, `num_boundary` 200, `--anchor-subset 2000`), every seed starting
  from the default `log10(A_sei)` = 22.885 -- the seed-variance reference.
- **fullanchor**: seed 0 with exactly that protocol but *without* `--anchor-subset`, i.e. all
  85900 observation points as anchors and data-fit points (the notebook's choice). Its only
  difference from `seeds5` seed 0 is the anchor set, so the pair validates the subset option.
  Cost on this machine: ~2.5-4 s per iteration against ~0.15-0.3 s for the subset.
- **fullbudget**: seed 0 with the notebook's full budget and collocation counts -- 15000 Adam
  + 2000 L-BFGS iterations, `num_domain` 5000, `num_boundary` 1000 -- and `--anchor-subset
  2000` (the full-anchor version of this budget is ~10 h per seed here). This is the closest
  feasible match to the Databricks protocol.

## Runs

| run | train batteries | train cycles | train rows | held-out batteries | held-out cycles | held-out rows | seeds | artifacts |
|---|---|---|---|---|---|---|---|---|
| legacy | 19 | 1076 | 53800 | 5 | 330 | 16500 | 0, 1, 2 | `runs/legacy/artifacts` |
| dynamic | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/dynamic/artifacts` |
| dynamic-subset | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/dynamic_subset/artifacts` |
| seeds5 | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2, 3, 4 | `runs/seeds5/artifacts` |
| ensemble10 | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | `runs/ensemble10/artifacts` |
| fullanchor | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0 | `runs/fullanchor/artifacts` |
| fullbudget | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0 | `runs/fullbudget/artifacts` |
| grid_newton | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/grid_newton/artifacts` |
| grid_thermal | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/grid_thermal/artifacts` |
| grid_both | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/grid_both/artifacts` |
| fixed_12 | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/fixed_12/artifacts` |
| fixed_18 | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/fixed_18/artifacts` |
| fixed_21 | 24 | 1718 | 85900 | 6 | 477 | 23850 | 0, 1, 2 | `runs/fixed_21/artifacts` |

- **legacy**: held-out table lacks designated batteries ['B0056'].
- **legacy**: 59 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **dynamic**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **dynamic-subset**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **seeds5**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **ensemble10**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **fullanchor**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **fullbudget**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **grid_newton**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **grid_thermal**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **grid_both**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **fixed_12**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **fixed_18**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).
- **fixed_21**: 61 training and 1 held-out cycles are dropped on load (missing Re or other required values).

## Databricks reference result (v9) vs local runs

The v9 notebook run is the only Databricks run with recorded accuracy figures. It used the legacy 1.4 Ah capacity floor (19 training / 5 held-out batteries -- B0056 had no data), every observation point as anchor and the full 15000 + 2000 budget; the local runs below use the tables and protocols described under Protocol, so the per-battery numbers are the like-for-like comparison and the overall held-out mean is not (the local held-out set contains B0056).

| held-out MAE (°C) | Databricks v9 (19/5 split, full anchors, 15000 + 2000) | legacy | dynamic | dynamic-subset | seeds5 | ensemble10 | fullanchor | fullbudget | grid_newton | grid_thermal | grid_both | fixed_12 | fixed_18 | fixed_21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 7.65 | 8.70 ± 2.12 | 6.58 ± 1.28 | 4.73 ± 0.14 | 4.86 ± 0.11 | 4.76 ± 0.13 | 4.93 ± 0.00 | 5.76 ± 0.00 | 4.87 ± 0.22 | 4.82 ± 0.25 | 4.91 ± 0.42 | 4.97 ± 0.09 | 5.04 ± 0.04 | 5.04 ± 0.14 |
| train MAE (for reference) | 1.20 | 11.35 ± 2.00 | 9.59 ± 1.38 | 1.86 ± 0.02 | 1.82 ± 0.04 | 1.84 ± 0.05 | 1.85 ± 0.00 | 1.47 ± 0.00 | 1.85 ± 0.00 | 1.89 ± 0.02 | 1.81 ± 0.09 | 1.78 ± 0.04 | 1.80 ± 0.04 | 1.86 ± 0.10 |
| B0025 | 4.20 | 10.42 ± 3.02 | 9.97 ± 2.71 | 4.29 ± 0.36 | 4.56 ± 0.25 | 4.27 ± 0.39 | 4.17 ± 0.00 | 3.78 ± 0.00 | 4.36 ± 0.15 | 4.38 ± 0.12 | 4.38 ± 0.52 | 4.32 ± 0.17 | 4.26 ± 0.22 | 4.56 ± 0.13 |
| B0030 | 1.10 | 21.92 ± 0.87 | 22.00 ± 0.97 | 2.69 ± 0.10 | 2.54 ± 0.73 | 2.64 ± 0.60 | 2.51 ± 0.00 | 1.03 ± 0.00 | 2.59 ± 0.12 | 3.09 ± 0.55 | 2.79 ± 0.25 | 2.17 ± 0.14 | 2.20 ± 0.07 | 3.28 ± 1.26 |
| B0036 | 11.40 | 7.16 ± 2.50 | 6.86 ± 2.14 | 8.35 ± 0.45 | 8.65 ± 0.27 | 8.48 ± 0.30 | 8.86 ± 0.00 | 10.86 ± 0.00 | 8.68 ± 0.47 | 8.45 ± 0.34 | 8.83 ± 0.94 | 9.08 ± 0.21 | 9.07 ± 0.19 | 8.89 ± 0.39 |
| B0042 | 0.90 | 4.94 ± 2.71 | 4.68 ± 1.99 | 1.63 ± 0.15 | 1.57 ± 0.12 | 1.53 ± 0.16 | 1.70 ± 0.00 | 1.19 ± 0.00 | 1.61 ± 0.18 | 1.57 ± 0.16 | 1.36 ± 0.25 | 1.32 ± 0.21 | 1.48 ± 0.10 | 1.42 ± 0.24 |
| B0047 | 7.30 | 2.37 ± 0.92 | 3.44 ± 0.65 | 1.41 ± 0.24 | 1.42 ± 0.27 | 1.43 ± 0.28 | 1.16 ± 0.00 | 1.60 ± 0.00 | 1.49 ± 0.36 | 1.62 ± 0.25 | 1.32 ± 0.30 | 1.09 ± 0.15 | 1.40 ± 0.14 | 1.52 ± 0.16 |
| B0056 | -- (absent in v9) | -- (absent) | 1.66 ± 0.09 | 2.16 ± 0.20 | 2.21 ± 0.25 | 2.16 ± 0.30 | 2.31 ± 0.00 | 3.17 ± 0.00 | 2.21 ± 0.24 | 2.16 ± 0.24 | 2.24 ± 0.25 | 2.42 ± 0.27 | 2.57 ± 0.22 | 2.39 ± 0.21 |

## Per-battery held-out MAE (°C, mean ± std over seeds)

| battery | legacy | dynamic | dynamic-subset | seeds5 | ensemble10 | fullanchor | fullbudget | grid_newton | grid_thermal | grid_both | fixed_12 | fixed_18 | fixed_21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0025 | 10.42 ± 3.02 | 9.97 ± 2.71 | 4.29 ± 0.36 | 4.56 ± 0.25 | 4.27 ± 0.39 | 4.17 ± 0.00 | 3.78 ± 0.00 | 4.36 ± 0.15 | 4.38 ± 0.12 | 4.38 ± 0.52 | 4.32 ± 0.17 | 4.26 ± 0.22 | 4.56 ± 0.13 |
| B0030 | 21.92 ± 0.87 | 22.00 ± 0.97 | 2.69 ± 0.10 | 2.54 ± 0.73 | 2.64 ± 0.60 | 2.51 ± 0.00 | 1.03 ± 0.00 | 2.59 ± 0.12 | 3.09 ± 0.55 | 2.79 ± 0.25 | 2.17 ± 0.14 | 2.20 ± 0.07 | 3.28 ± 1.26 |
| B0036 | 7.16 ± 2.50 | 6.86 ± 2.14 | 8.35 ± 0.45 | 8.65 ± 0.27 | 8.48 ± 0.30 | 8.86 ± 0.00 | 10.86 ± 0.00 | 8.68 ± 0.47 | 8.45 ± 0.34 | 8.83 ± 0.94 | 9.08 ± 0.21 | 9.07 ± 0.19 | 8.89 ± 0.39 |
| B0042 | 4.94 ± 2.71 | 4.68 ± 1.99 | 1.63 ± 0.15 | 1.57 ± 0.12 | 1.53 ± 0.16 | 1.70 ± 0.00 | 1.19 ± 0.00 | 1.61 ± 0.18 | 1.57 ± 0.16 | 1.36 ± 0.25 | 1.32 ± 0.21 | 1.48 ± 0.10 | 1.42 ± 0.24 |
| B0047 | 2.37 ± 0.92 | 3.44 ± 0.65 | 1.41 ± 0.24 | 1.42 ± 0.27 | 1.43 ± 0.28 | 1.16 ± 0.00 | 1.60 ± 0.00 | 1.49 ± 0.36 | 1.62 ± 0.25 | 1.32 ± 0.30 | 1.09 ± 0.15 | 1.40 ± 0.14 | 1.52 ± 0.16 |
| B0056 | -- (absent) | 1.66 ± 0.09 | 2.16 ± 0.20 | 2.21 ± 0.25 | 2.16 ± 0.30 | 2.31 ± 0.00 | 3.17 ± 0.00 | 2.21 ± 0.24 | 2.16 ± 0.24 | 2.24 ± 0.25 | 2.42 ± 0.27 | 2.57 ± 0.22 | 2.39 ± 0.21 |

## Per-battery training MAE (°C, mean ± std over seeds)

| battery | legacy | dynamic | dynamic-subset | seeds5 | ensemble10 | fullanchor | fullbudget | grid_newton | grid_thermal | grid_both | fixed_12 | fixed_18 | fixed_21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0005 | 7.95 ± 2.75 | 7.76 ± 2.32 | 0.97 ± 0.22 | 1.02 ± 0.16 | 0.95 ± 0.16 | 1.01 ± 0.00 | 0.87 ± 0.00 | 1.04 ± 0.11 | 0.99 ± 0.06 | 0.96 ± 0.22 | 0.96 ± 0.03 | 0.99 ± 0.09 | 1.08 ± 0.16 |
| B0006 | 7.86 ± 2.64 | 7.52 ± 2.29 | 1.92 ± 0.21 | 1.94 ± 0.12 | 1.91 ± 0.21 | 1.92 ± 0.00 | 1.62 ± 0.00 | 2.02 ± 0.15 | 1.89 ± 0.10 | 1.89 ± 0.21 | 1.93 ± 0.07 | 1.93 ± 0.11 | 1.98 ± 0.13 |
| B0007 | 7.90 ± 2.69 | 7.54 ± 2.35 | 1.26 ± 0.23 | 1.35 ± 0.14 | 1.26 ± 0.20 | 1.23 ± 0.00 | 1.02 ± 0.00 | 1.36 ± 0.12 | 1.27 ± 0.06 | 1.21 ± 0.24 | 1.31 ± 0.03 | 1.32 ± 0.07 | 1.44 ± 0.13 |
| B0018 | 6.50 ± 2.73 | 6.21 ± 2.33 | 2.40 ± 0.10 | 2.36 ± 0.09 | 2.39 ± 0.20 | 2.34 ± 0.00 | 1.96 ± 0.00 | 2.46 ± 0.09 | 2.34 ± 0.08 | 2.32 ± 0.10 | 2.32 ± 0.05 | 2.31 ± 0.09 | 2.34 ± 0.04 |
| B0026 | 10.67 ± 3.22 | 10.22 ± 2.88 | 2.14 ± 0.25 | 1.96 ± 0.27 | 2.00 ± 0.28 | 1.59 ± 0.00 | 2.06 ± 0.00 | 1.83 ± 0.08 | 2.41 ± 0.26 | 1.75 ± 0.17 | 1.93 ± 0.17 | 2.06 ± 0.17 | 1.89 ± 0.06 |
| B0027 | 13.17 ± 3.25 | 12.74 ± 2.94 | 2.67 ± 0.13 | 2.40 ± 0.19 | 2.47 ± 0.18 | 2.25 ± 0.00 | 1.40 ± 0.00 | 2.29 ± 0.20 | 2.76 ± 0.04 | 2.47 ± 0.19 | 2.32 ± 0.13 | 2.38 ± 0.11 | 2.17 ± 0.08 |
| B0028 | 6.18 ± 3.16 | 5.75 ± 2.84 | 4.27 ± 0.17 | 4.19 ± 0.17 | 4.25 ± 0.33 | 3.84 ± 0.00 | 4.83 ± 0.00 | 4.20 ± 0.03 | 4.48 ± 0.31 | 3.82 ± 0.19 | 4.23 ± 0.19 | 4.34 ± 0.21 | 4.25 ± 0.03 |
| B0029 | 20.29 ± 1.05 | 20.30 ± 1.17 | 1.00 ± 0.58 | 0.90 ± 0.36 | 0.99 ± 0.40 | 1.37 ± 0.00 | 0.86 ± 0.00 | 0.80 ± 0.27 | 0.94 ± 0.41 | 1.10 ± 0.32 | 1.41 ± 0.04 | 1.42 ± 0.16 | 1.83 ± 0.68 |
| B0031 | 21.49 ± 0.97 | 21.51 ± 1.08 | 1.45 ± 0.38 | 1.07 ± 0.40 | 1.17 ± 0.56 | 1.06 ± 0.00 | 0.62 ± 0.00 | 1.16 ± 0.18 | 1.59 ± 0.37 | 1.23 ± 0.33 | 0.98 ± 0.21 | 1.08 ± 0.28 | 1.94 ± 1.63 |
| B0032 | 22.44 ± 0.84 | 22.53 ± 0.94 | 3.46 ± 0.08 | 3.41 ± 0.77 | 3.48 ± 0.59 | 3.41 ± 0.00 | 1.49 ± 0.00 | 3.41 ± 0.18 | 3.90 ± 0.69 | 3.61 ± 0.22 | 2.93 ± 0.09 | 2.95 ± 0.10 | 4.05 ± 1.14 |
| B0033 | 16.06 ± 2.77 | 16.03 ± 2.36 | 3.87 ± 0.38 | 3.65 ± 0.29 | 3.80 ± 0.35 | 3.84 ± 0.00 | 2.54 ± 0.00 | 3.66 ± 0.28 | 3.82 ± 0.26 | 3.77 ± 0.65 | 3.59 ± 0.09 | 3.56 ± 0.12 | 3.55 ± 0.15 |
| B0034 | 17.52 ± 2.89 | 17.31 ± 2.43 | 1.14 ± 0.03 | 1.23 ± 0.11 | 1.24 ± 0.14 | 1.40 ± 0.00 | 1.49 ± 0.00 | 1.23 ± 0.11 | 1.18 ± 0.05 | 1.23 ± 0.09 | 1.40 ± 0.10 | 1.33 ± 0.06 | 1.34 ± 0.06 |
| B0038 | 16.57 ± 0.55 | 16.80 ± 0.65 | 2.45 ± 0.35 | 1.99 ± 0.40 | 2.12 ± 0.32 | 2.13 ± 0.00 | 1.65 ± 0.00 | 1.91 ± 0.39 | 2.50 ± 0.27 | 2.29 ± 0.21 | 1.78 ± 0.22 | 1.74 ± 0.17 | 1.86 ± 0.28 |
| B0039 | 15.00 ± 0.76 | 15.61 ± 0.87 | 3.00 ± 0.53 | 2.67 ± 0.53 | 2.75 ± 0.46 | 2.84 ± 0.00 | 2.01 ± 0.00 | 2.81 ± 0.42 | 2.91 ± 0.30 | 2.81 ± 0.48 | 2.51 ± 0.08 | 2.51 ± 0.24 | 2.17 ± 0.44 |
| B0040 | 14.96 ± 0.65 | 15.14 ± 0.75 | 2.52 ± 0.66 | 1.91 ± 0.74 | 2.06 ± 0.62 | 2.05 ± 0.00 | 1.03 ± 0.00 | 2.07 ± 0.51 | 2.53 ± 0.42 | 2.35 ± 0.51 | 1.68 ± 0.21 | 1.70 ± 0.27 | 1.42 ± 0.25 |
| B0041 | -- (absent) | 2.92 ± 0.57 | 2.62 ± 0.58 | 2.70 ± 0.50 | 2.79 ± 0.43 | 2.29 ± 0.00 | 2.13 ± 0.00 | 2.81 ± 0.74 | 2.76 ± 0.18 | 2.33 ± 0.82 | 2.08 ± 0.32 | 2.57 ± 0.29 | 2.77 ± 0.43 |
| B0043 | 5.30 ± 2.66 | 4.97 ± 1.90 | 0.99 ± 0.06 | 0.98 ± 0.18 | 0.97 ± 0.15 | 1.04 ± 0.00 | 0.96 ± 0.00 | 1.00 ± 0.17 | 1.06 ± 0.10 | 0.95 ± 0.07 | 1.09 ± 0.07 | 1.07 ± 0.05 | 1.12 ± 0.13 |
| B0044 | 6.50 ± 2.89 | 5.63 ± 1.95 | 1.97 ± 0.12 | 1.93 ± 0.10 | 1.87 ± 0.14 | 2.08 ± 0.00 | 1.02 ± 0.00 | 1.99 ± 0.18 | 1.92 ± 0.16 | 1.74 ± 0.19 | 1.65 ± 0.18 | 1.81 ± 0.13 | 1.78 ± 0.26 |
| B0045 | -- (absent) | 2.01 ± 0.60 | 2.16 ± 0.02 | 2.18 ± 0.29 | 2.19 ± 0.32 | 2.34 ± 0.00 | 1.12 ± 0.00 | 2.31 ± 0.28 | 2.30 ± 0.37 | 2.16 ± 0.32 | 2.29 ± 0.13 | 2.56 ± 0.19 | 2.41 ± 0.07 |
| B0046 | 1.59 ± 0.70 | 2.45 ± 0.45 | 0.74 ± 0.11 | 0.82 ± 0.13 | 0.74 ± 0.14 | 0.73 ± 0.00 | 1.30 ± 0.00 | 0.78 ± 0.13 | 0.82 ± 0.11 | 0.80 ± 0.14 | 0.75 ± 0.02 | 0.73 ± 0.03 | 0.75 ± 0.08 |
| B0048 | 2.30 ± 0.76 | 3.41 ± 0.47 | 1.59 ± 0.08 | 1.64 ± 0.20 | 1.58 ± 0.19 | 1.43 ± 0.00 | 1.04 ± 0.00 | 1.59 ± 0.19 | 1.57 ± 0.21 | 1.50 ± 0.34 | 1.43 ± 0.12 | 1.35 ± 0.09 | 1.49 ± 0.17 |
| B0053 | -- (absent) | 2.29 ± 0.05 | 1.01 ± 0.13 | 1.09 ± 0.20 | 1.08 ± 0.19 | 1.12 ± 0.00 | 1.26 ± 0.00 | 1.12 ± 0.13 | 1.05 ± 0.12 | 1.01 ± 0.09 | 1.02 ± 0.10 | 1.09 ± 0.04 | 1.09 ± 0.18 |
| B0054 | -- (absent) | 2.82 ± 0.42 | 1.57 ± 0.14 | 1.63 ± 0.13 | 1.65 ± 0.15 | 1.53 ± 0.00 | 1.32 ± 0.00 | 1.62 ± 0.15 | 1.64 ± 0.06 | 1.59 ± 0.04 | 1.54 ± 0.08 | 1.62 ± 0.05 | 1.63 ± 0.15 |
| B0055 | -- (absent) | 1.29 ± 0.09 | 0.94 ± 0.08 | 1.00 ± 0.18 | 0.99 ± 0.18 | 1.15 ± 0.00 | 0.94 ± 0.00 | 1.06 ± 0.14 | 0.97 ± 0.14 | 0.91 ± 0.05 | 0.96 ± 0.06 | 1.03 ± 0.16 | 0.96 ± 0.08 |

## Per-ambient held-out MAE (°C, mean ± std over seeds)

| ambient | legacy | dynamic | dynamic-subset | seeds5 | ensemble10 | fullanchor | fullbudget | grid_newton | grid_thermal | grid_both | fixed_12 | fixed_18 | fixed_21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 °C | 2.09 ± 0.86 | 2.34 ± 0.23 | 1.81 ± 0.15 | 1.84 ± 0.12 | 1.81 ± 0.20 | 1.79 ± 0.00 | 2.43 ± 0.00 | 1.85 ± 0.06 | 1.87 ± 0.18 | 1.82 ± 0.05 | 1.86 ± 0.14 | 2.02 ± 0.17 | 1.97 ± 0.08 |
| 22 °C | 6.15 ± 3.47 | 5.74 ± 3.05 | 1.93 ± 0.21 | 1.81 ± 0.18 | 1.79 ± 0.21 | 2.22 ± 0.00 | 1.31 ± 0.00 | 1.92 ± 0.24 | 1.83 ± 0.13 | 1.52 ± 0.29 | 1.51 ± 0.29 | 1.79 ± 0.21 | 1.61 ± 0.46 |
| 24 °C | 7.57 ± 2.56 | 7.25 ± 2.21 | 7.85 ± 0.44 | 8.14 ± 0.26 | 7.95 ± 0.29 | 8.28 ± 0.00 | 9.98 ± 0.00 | 8.14 ± 0.43 | 7.94 ± 0.31 | 8.27 ± 0.88 | 8.49 ± 0.19 | 8.47 ± 0.18 | 8.35 ± 0.34 |
| 43 °C | 21.92 ± 0.87 | 22.00 ± 0.97 | 2.69 ± 0.10 | 2.54 ± 0.73 | 2.64 ± 0.60 | 2.51 ± 0.00 | 1.03 ± 0.00 | 2.59 ± 0.12 | 3.09 ± 0.55 | 2.79 ± 0.25 | 2.17 ± 0.14 | 2.20 ± 0.07 | 3.28 ± 1.26 |

## Per-ambient training MAE (°C, mean ± std over seeds)

| ambient | legacy | dynamic | dynamic-subset | seeds5 | ensemble10 | fullanchor | fullbudget | grid_newton | grid_thermal | grid_both | fixed_12 | fixed_18 | fixed_21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 °C | 2.06 ± 0.74 | 2.50 ± 0.09 | 1.39 ± 0.02 | 1.45 ± 0.08 | 1.43 ± 0.07 | 1.38 ± 0.00 | 1.19 ± 0.00 | 1.46 ± 0.05 | 1.43 ± 0.03 | 1.36 ± 0.10 | 1.33 ± 0.07 | 1.38 ± 0.09 | 1.41 ± 0.12 |
| 22 °C | 6.92 ± 3.35 | 6.52 ± 2.93 | 1.24 ± 0.09 | 1.19 ± 0.15 | 1.16 ± 0.11 | 1.53 ± 0.00 | 0.96 ± 0.00 | 1.27 ± 0.16 | 1.27 ± 0.09 | 1.06 ± 0.10 | 1.16 ± 0.07 | 1.29 ± 0.15 | 1.24 ± 0.25 |
| 24 °C | 10.26 ± 2.79 | 11.02 ± 2.40 | 2.04 ± 0.05 | 2.02 ± 0.02 | 2.03 ± 0.05 | 2.04 ± 0.00 | 1.70 ± 0.00 | 2.04 ± 0.01 | 2.05 ± 0.02 | 1.99 ± 0.08 | 2.02 ± 0.02 | 2.01 ± 0.04 | 2.04 ± 0.05 |
| 43 °C | 21.41 ± 0.95 | 21.45 ± 1.06 | 1.97 ± 0.08 | 1.79 ± 0.40 | 1.88 ± 0.34 | 1.95 ± 0.00 | 0.99 ± 0.00 | 1.79 ± 0.10 | 2.14 ± 0.33 | 1.98 ± 0.11 | 1.78 ± 0.10 | 1.82 ± 0.06 | 2.61 ± 1.14 |
| 44 °C | 15.53 ± 0.65 | 15.86 ± 0.75 | 2.66 ± 0.51 | 2.19 ± 0.54 | 2.32 ± 0.44 | 2.34 ± 0.00 | 1.57 ± 0.00 | 2.27 ± 0.44 | 2.65 ± 0.29 | 2.49 ± 0.40 | 2.00 ± 0.04 | 1.99 ± 0.11 | 1.82 ± 0.24 |

## Summary across seeds (mean ± std)

| run | seeds | log10 A_sei | train MAE (°C) | train RMSE (°C) | held-out MAE (°C) | held-out RMSE (°C) | gap (held-out − train) | core T max (°C, unvalidated) | SEI min | verdicts (ambient ≤ 45 °C only) |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 3 | 22.85 ± 0.03 | 11.35 ± 2.00 | 13.38 ± 1.47 | 8.70 ± 2.12 | 10.71 ± 1.44 | -2.65 ± 0.14 | 31.2 ± 1.4 | 0.838 ± 0.079 | 1 DANGER / 2 SAFE **(inconsistent)** |
| dynamic | 3 | 22.86 ± 0.02 | 9.59 ± 1.38 | 12.28 ± 1.21 | 6.58 ± 1.28 | 8.91 ± 0.97 | -3.01 ± 0.10 | 31.6 ± 1.2 | 0.841 ± 0.105 | 3 SAFE |
| dynamic-subset | 3 | 17.17 ± 2.33 | 1.86 ± 0.02 | 2.52 ± 0.04 | 4.73 ± 0.14 | 6.69 ± 0.32 | 2.86 ± 0.14 | 54.0 ± 5.6 | 0.272 ± 0.385 | 1 WARNING / 2 SAFE **(inconsistent)** |
| seeds5 | 5 | 14.47 ± 4.14 | 1.82 ± 0.04 | 2.45 ± 0.07 | 4.86 ± 0.11 | 6.83 ± 0.15 | 3.03 ± 0.12 | 54.5 ± 2.2 | 0.467 ± 0.443 | 1 DANGER / 4 SAFE **(inconsistent)** |
| ensemble10 | 10 | 16.34 ± 3.74 | 1.84 ± 0.05 | 2.48 ± 0.07 | 4.76 ± 0.13 | 6.71 ± 0.21 | 2.92 ± 0.15 | 54.9 ± 4.1 | 0.333 ± 0.442 | 1 DANGER / 1 WARNING / 8 SAFE **(inconsistent)** |
| fullanchor | 1 | 16.62 ± 0.00 | 1.85 ± 0.00 | 2.46 ± 0.00 | 4.93 ± 0.00 | 6.87 ± 0.00 | 3.08 ± 0.00 | 63.9 ± 0.0 | 0.968 ± 0.000 | 1 WARNING |
| fullbudget | 1 | 11.55 ± 0.00 | 1.47 ± 0.00 | 2.01 ± 0.00 | 5.76 ± 0.00 | 8.31 ± 0.00 | 4.29 ± 0.00 | 34.6 ± 0.0 | 0.423 ± 0.000 | 1 DANGER |
| grid_newton | 3 | 15.69 ± 4.32 | 1.85 ± 0.00 | 2.47 ± 0.05 | 4.87 ± 0.22 | 6.85 ± 0.20 | 3.03 ± 0.22 | 54.3 ± 1.7 | 0.333 ± 0.471 | 3 SAFE |
| grid_thermal | 3 | 14.09 ± 6.49 | 1.89 ± 0.02 | 2.53 ± 0.02 | 4.82 ± 0.25 | 6.72 ± 0.19 | 2.93 ± 0.23 | 50.9 ± 2.0 | 0.332 ± 0.470 | 3 SAFE |
| grid_both | 3 | 13.24 ± 8.57 | 1.81 ± 0.09 | 2.45 ± 0.14 | 4.91 ± 0.42 | 6.92 ± 0.58 | 3.10 ± 0.51 | 56.3 ± 6.1 | 0.333 ± 0.471 | 1 WARNING / 2 SAFE **(inconsistent)** |
| fixed_12 | 3 | 12.00 ± 0.00 | 1.78 ± 0.04 | 2.37 ± 0.03 | 4.97 ± 0.09 | 6.96 ± 0.07 | 3.19 ± 0.12 | 47.2 ± 5.2 | 0.949 ± 0.039 | 3 SAFE |
| fixed_18 | 3 | 18.00 ± 0.00 | 1.80 ± 0.04 | 2.39 ± 0.03 | 5.04 ± 0.04 | 7.01 ± 0.12 | 3.25 ± 0.06 | 41.8 ± 2.2 | 0.888 ± 0.009 | 1 DANGER / 2 SAFE **(inconsistent)** |
| fixed_21 | 3 | 21.00 ± 0.00 | 1.86 ± 0.10 | 2.49 ± 0.13 | 5.04 ± 0.14 | 6.93 ± 0.24 | 3.18 ± 0.09 | 34.6 ± 4.6 | 0.333 ± 0.471 | 3 SAFE |

## Per-seed checkpoint metrics

| run | seed | A_sei | train MAE | train RMSE | held-out MAE | held-out RMSE | core T max (°C, unvalidated) | SEI min | threshold | verdict | decided by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 0 | 7.65e+22 | 14.18 | 15.46 | 11.70 | 12.74 | 33.0 | 0.920 | 0.833 | SAFE | none |
| legacy | 1 | 7.23e+22 | 10.01 | 12.44 | 7.37 | 9.86 | 29.5 | 0.732 | 0.738 | DANGER | sei_below_threshold |
| legacy | 2 | 6.47e+22 | 9.87 | 12.25 | 7.04 | 9.54 | 31.1 | 0.862 | 0.809 | SAFE | none |
| dynamic | 0 | 7.65e+22 | 11.54 | 13.98 | 8.38 | 10.27 | 32.8 | 0.947 | 0.820 | SAFE | none |
| dynamic | 1 | 7.18e+22 | 8.47 | 11.27 | 5.54 | 8.14 | 29.9 | 0.698 | 0.692 | SAFE | none |
| dynamic | 2 | 6.94e+22 | 8.77 | 11.59 | 5.81 | 8.32 | 32.0 | 0.878 | 0.811 | SAFE | none |
| dynamic-subset | 0 | 5.53e+14 | 1.86 | 2.46 | 4.91 | 7.10 | 62.0 | 0.816 | 0.516 | WARNING | core_temp_warning |
| dynamic-subset | 1 | 2.10e+20 | 1.85 | 2.55 | 4.56 | 6.31 | 50.6 | 0.000 | 0.000 | SAFE | none |
| dynamic-subset | 2 | 2.83e+16 | 1.89 | 2.54 | 4.72 | 6.67 | 49.6 | 0.000 | 0.000 | SAFE | none |
| seeds5 | 0 | 3.55e+09 | 1.78 | 2.32 | 4.97 | 7.09 | 53.9 | 0.998 | 0.999 | DANGER | sei_below_threshold |
| seeds5 | 1 | 2.53e+20 | 1.83 | 2.48 | 4.65 | 6.66 | 52.7 | 0.000 | 0.000 | SAFE | none |
| seeds5 | 2 | 4.64e+16 | 1.86 | 2.52 | 4.84 | 6.81 | 57.2 | 0.000 | 0.000 | SAFE | none |
| seeds5 | 3 | 9.72e+09 | 1.87 | 2.50 | 4.92 | 6.88 | 56.8 | 0.968 | 0.889 | SAFE | none |
| seeds5 | 4 | 5.29e+15 | 1.78 | 2.42 | 4.90 | 6.70 | 51.8 | 0.367 | 0.000 | SAFE | none |
| ensemble10 | 0 | 3.55e+09 | 1.78 | 2.32 | 4.97 | 7.09 | 53.9 | 0.998 | 0.999 | DANGER | sei_below_threshold |
| ensemble10 | 1 | 2.53e+20 | 1.83 | 2.48 | 4.65 | 6.66 | 52.7 | 0.000 | 0.000 | SAFE | none |
| ensemble10 | 2 | 4.64e+16 | 1.86 | 2.52 | 4.84 | 6.81 | 57.2 | 0.000 | 0.000 | SAFE | none |
| ensemble10 | 3 | 9.72e+09 | 1.87 | 2.50 | 4.92 | 6.88 | 56.8 | 0.968 | 0.889 | SAFE | none |
| ensemble10 | 4 | 5.29e+15 | 1.78 | 2.42 | 4.90 | 6.70 | 51.8 | 0.367 | 0.000 | SAFE | none |
| ensemble10 | 5 | 3.17e+20 | 1.89 | 2.55 | 4.60 | 6.65 | 56.7 | 0.000 | 0.000 | SAFE | none |
| ensemble10 | 6 | 9.94e+16 | 1.76 | 2.44 | 4.60 | 6.40 | 62.3 | 1.000 | 0.891 | WARNING | core_temp_warning |
| ensemble10 | 7 | 1.96e+18 | 1.84 | 2.48 | 4.74 | 6.57 | 57.0 | 0.000 | 0.000 | SAFE | none |
| ensemble10 | 8 | 1.94e+15 | 1.89 | 2.52 | 4.79 | 6.91 | 46.0 | 0.000 | 0.000 | SAFE | none |
| ensemble10 | 9 | 1.01e+20 | 1.88 | 2.56 | 4.63 | 6.41 | 54.3 | 0.000 | 0.000 | SAFE | none |
| fullanchor | 0 | 4.13e+16 | 1.85 | 2.46 | 4.93 | 6.87 | 63.9 | 0.968 | 0.930 | WARNING | core_temp_warning |
| fullbudget | 0 | 3.53e+11 | 1.47 | 2.01 | 5.76 | 8.31 | 34.6 | 0.423 | 0.445 | DANGER | sei_below_threshold |
| grid_newton | 0 | 7.51e+09 | 1.85 | 2.40 | 5.18 | 7.12 | 52.3 | 1.000 | 0.999 | SAFE | none |
| grid_newton | 1 | 1.72e+20 | 1.85 | 2.51 | 4.68 | 6.66 | 54.2 | 0.000 | 0.000 | SAFE | none |
| grid_newton | 2 | 8.93e+16 | 1.84 | 2.49 | 4.76 | 6.76 | 56.4 | 0.000 | 0.000 | SAFE | none |
| grid_thermal | 0 | 1.19e+05 | 1.91 | 2.52 | 5.16 | 6.96 | 53.3 | 0.996 | 0.978 | SAFE | none |
| grid_thermal | 1 | 1.18e+20 | 1.86 | 2.51 | 4.58 | 6.50 | 48.5 | 0.000 | 0.000 | SAFE | none |
| grid_thermal | 2 | 1.33e+17 | 1.91 | 2.55 | 4.71 | 6.72 | 50.8 | 0.000 | 0.000 | SAFE | none |
| grid_both | 0 | 1.77e+01 | 1.68 | 2.26 | 5.45 | 7.65 | 62.8 | 1.000 | 0.999 | WARNING | core_temp_warning |
| grid_both | 1 | 5.59e+20 | 1.87 | 2.59 | 4.41 | 6.24 | 48.2 | 0.000 | 0.000 | SAFE | none |
| grid_both | 2 | 5.50e+17 | 1.87 | 2.50 | 4.86 | 6.87 | 58.1 | 0.000 | 0.000 | SAFE | none |
| fixed_12 | 0 | 1.00e+12 | 1.82 | 2.42 | 4.85 | 6.90 | 54.4 | 0.944 | 0.895 | SAFE | none |
| fixed_12 | 1 | 1.00e+12 | 1.73 | 2.34 | 5.02 | 6.91 | 42.4 | 1.000 | 0.961 | SAFE | none |
| fixed_12 | 2 | 1.00e+12 | 1.78 | 2.36 | 5.04 | 7.06 | 44.8 | 0.905 | 0.624 | SAFE | none |
| fixed_18 | 0 | 1.00e+18 | 1.86 | 2.44 | 5.02 | 7.06 | 41.0 | 0.894 | 0.789 | SAFE | none |
| fixed_18 | 1 | 1.00e+18 | 1.76 | 2.36 | 5.01 | 6.83 | 39.6 | 0.875 | 0.801 | SAFE | none |
| fixed_18 | 2 | 1.00e+18 | 1.78 | 2.38 | 5.10 | 7.12 | 44.8 | 0.895 | 0.897 | DANGER | sei_below_threshold |
| fixed_21 | 0 | 1.00e+21 | 1.88 | 2.44 | 5.19 | 7.27 | 41.1 | 0.000 | 0.000 | SAFE | none |
| fixed_21 | 1 | 1.00e+21 | 1.73 | 2.36 | 4.86 | 6.73 | 31.2 | 0.000 | 0.000 | SAFE | none |
| fixed_21 | 2 | 1.00e+21 | 1.98 | 2.67 | 5.08 | 6.79 | 31.6 | 1.000 | 0.997 | SAFE | none |

## A_sei identifiability across seeds

| run | seeds | initial log10 A_sei (min..max) | final log10 A_sei (mean ± std) | final range | status | reason |
|---|---|---|---|---|---|---|
| legacy | 3 | 22.88..22.88 | 22.85 ± 0.03 | 0.07 | **not identifiable** | every seed stayed within 0.1 decades of its initial value (max shift 0.074); the loss is insensitive to A_sei (initial value not recorded; assumed the recommended default) |
| dynamic | 3 | 22.88..22.88 | 22.86 ± 0.02 | 0.04 | **not identifiable** | every seed stayed within 0.1 decades of its initial value (max shift 0.043); the loss is insensitive to A_sei (initial value not recorded; assumed the recommended default) |
| dynamic-subset | 3 | 21.88..23.88 | 17.17 ± 2.33 | 5.58 | **not identifiable** | seeds disagree: final log10(A_sei) spans 5.58 decades (> 0.5); the value depends on the seed |
| seeds5 | 5 | 22.88..22.88 | 14.47 ± 4.14 | 10.85 | **not identifiable** | seeds disagree: final log10(A_sei) spans 10.85 decades (> 0.5); the value depends on the seed |
| ensemble10 | 10 | 22.88..22.88 | 16.34 ± 3.74 | 10.95 | **not identifiable** | seeds disagree: final log10(A_sei) spans 10.95 decades (> 0.5); the value depends on the seed (initial value not recorded; assumed the recommended default) |
| fullanchor | 1 | 22.88..22.88 | 16.62 ± 0.00 | 0.00 | **inconclusive** | fewer than two seeds |
| fullbudget | 1 | 22.88..22.88 | 11.55 ± 0.00 | 0.00 | **inconclusive** | fewer than two seeds |
| grid_newton | 3 | 22.88..22.88 | 15.69 ± 4.32 | 10.36 | **not identifiable** | seeds disagree: final log10(A_sei) spans 10.36 decades (> 0.5); the value depends on the seed |
| grid_thermal | 3 | 22.88..22.88 | 14.09 ± 6.49 | 15.00 | **not identifiable** | seeds disagree: final log10(A_sei) spans 15.00 decades (> 0.5); the value depends on the seed |
| grid_both | 3 | 22.88..22.88 | 13.24 ± 8.57 | 19.50 | **not identifiable** | seeds disagree: final log10(A_sei) spans 19.50 decades (> 0.5); the value depends on the seed |
| fixed_12 | 3 | -- | -- | -- | **fixed (not trained)** | log10(A_sei) held at 12 with --fix-log-a-sei |
| fixed_18 | 3 | -- | -- | -- | **fixed (not trained)** | log10(A_sei) held at 18 with --fix-log-a-sei |
| fixed_21 | 3 | -- | -- | -- | **fixed (not trained)** | log10(A_sei) held at 21 with --fix-log-a-sei |

Thresholds: seeds must end within 0.5 decades of each other and have moved at least 0.1 decades from their start; only seeds that *started* apart (`train --a-sei-init-spread`) can show identifiability rather than consistency. Runs trained with `--fix-log-a-sei` hold A_sei constant by construction.

## Final weighted loss terms (mean over seeds; share of total in brackets)

| run | pde_thermal | pde_sei | ic_T | ic_c | observe_T | dominant |
|---|---|---|---|---|---|---|
| dynamic-subset | 2.04e-04 (0%) | 4.68e-05 (0%) | 1.58e-03 (3%) | 1.15e-03 (2%) | 4.92e-02 (94%) | observe_T |
| seeds5 | 2.20e-04 (0%) | 1.39e-05 (0%) | 1.46e-03 (3%) | 8.34e-04 (2%) | 4.71e-02 (95%) | observe_T |
| fullanchor | 8.79e-05 (0%) | 4.11e-05 (0%) | 2.79e-03 (5%) | 4.96e-04 (1%) | 4.83e-02 (93%) | observe_T |
| fullbudget | 8.02e-05 (0%) | 2.60e-08 (0%) | 7.96e-04 (2%) | 4.81e-04 (1%) | 3.15e-02 (96%) | observe_T |
| grid_newton | 1.48e-04 (0%) | 2.68e-05 (0%) | 1.70e-03 (3%) | 6.83e-04 (1%) | 4.76e-02 (95%) | observe_T |
| grid_thermal | 1.80e-04 (0%) | 2.09e-05 (0%) | 1.94e-03 (4%) | 5.98e-04 (1%) | 4.99e-02 (95%) | observe_T |
| grid_both | 1.91e-04 (0%) | 7.51e-05 (0%) | 1.96e-03 (4%) | 8.62e-04 (2%) | 4.69e-02 (94%) | observe_T |
| fixed_12 | 1.73e-04 (0%) | 4.18e-09 (0%) | 8.58e-04 (2%) | 2.19e-04 (0%) | 4.46e-02 (97%) | observe_T |
| fixed_18 | 1.31e-04 (0%) | 2.91e-05 (0%) | 1.11e-03 (2%) | 3.50e-04 (1%) | 4.50e-02 (97%) | observe_T |
| fixed_21 | 1.73e-04 (0%) | 7.82e-03 (13%) | 1.07e-03 (2%) | 4.87e-04 (1%) | 4.87e-02 (84%) | observe_T |

Weights pde_thermal=10, pde_sei=10, ic_T=30, ic_c=30, observe_T=20; per-step histories are in each run's `loss_history_seed*.csv`.

## Physics options grid (opt-in; decision rule: adopt only if the worst held-out batteries improve by more than the seed spread and no other battery degrades beyond its spread)

### grid_newton vs seeds5: **REJECTED (negative result, left off)** -- does not lower B0025, B0036 beyond the seed spread

| held-out battery | seeds5 | grid_newton | delta | seed spread | status |
|---|---|---|---|---|---|
| B0025 (worst) | 4.56 | 4.36 | -0.20 | 0.25 | not improved beyond spread |
| B0030 | 2.54 | 2.59 | +0.05 | 0.73 | unchanged within spread |
| B0036 (worst) | 8.65 | 8.68 | +0.03 | 0.47 | not improved beyond spread |
| B0042 | 1.57 | 1.61 | +0.04 | 0.18 | unchanged within spread |
| B0047 | 1.42 | 1.49 | +0.07 | 0.36 | unchanged within spread |
| B0056 | 2.21 | 2.21 | +0.00 | 0.25 | unchanged within spread |

### grid_thermal vs seeds5: **REJECTED (negative result, left off)** -- does not lower B0025, B0036 beyond the seed spread

| held-out battery | seeds5 | grid_thermal | delta | seed spread | status |
|---|---|---|---|---|---|
| B0025 (worst) | 4.56 | 4.38 | -0.18 | 0.25 | not improved beyond spread |
| B0030 | 2.54 | 3.09 | +0.55 | 0.73 | unchanged within spread |
| B0036 (worst) | 8.65 | 8.45 | -0.20 | 0.34 | not improved beyond spread |
| B0042 | 1.57 | 1.57 | +0.01 | 0.16 | unchanged within spread |
| B0047 | 1.42 | 1.62 | +0.20 | 0.27 | unchanged within spread |
| B0056 | 2.21 | 2.16 | -0.05 | 0.25 | unchanged within spread |

### grid_both vs seeds5: **REJECTED (negative result, left off)** -- does not lower B0025, B0036 beyond the seed spread

| held-out battery | seeds5 | grid_both | delta | seed spread | status |
|---|---|---|---|---|---|
| B0025 (worst) | 4.56 | 4.38 | -0.17 | 0.52 | not improved beyond spread |
| B0030 | 2.54 | 2.79 | +0.25 | 0.73 | unchanged within spread |
| B0036 (worst) | 8.65 | 8.83 | +0.17 | 0.94 | not improved beyond spread |
| B0042 | 1.57 | 1.36 | -0.20 | 0.25 | unchanged within spread |
| B0047 | 1.42 | 1.32 | -0.10 | 0.30 | unchanged within spread |
| B0056 | 2.21 | 2.24 | +0.03 | 0.25 | unchanged within spread |

## Abuse-test verdict spread (ensemble per run)

Core temperature is **unvalidated** (inferred through the physics, never compared with an internal sensor). The verdict uses in-domain sweep points only, i.e. it **excludes ambient > 45 °C**; it is a heuristic, not a certified safety result.

| run | models | DANGER | WARNING | SAFE | consistent | decided by (count x criterion) | SEI threshold min..max (mean ± std) | core T max (°C, unvalidated) min..max |
|---|---|---|---|---|---|---|---|---|
| legacy | 3 | 1 | 0 | 2 | **no** | 2 x none, 1 x sei_below_threshold | 0.738..0.833 (0.793 ± 0.040) | 29.5..33.0 |
| dynamic | 3 | 0 | 0 | 3 | yes | 3 x none | 0.692..0.820 (0.774 ± 0.058) | 29.9..32.8 |
| dynamic-subset | 3 | 0 | 1 | 2 | **no** | 1 x core_temp_warning, 2 x none | 0.000..0.516 (0.172 ± 0.243) | 49.6..62.0 |
| seeds5 | 5 | 1 | 0 | 4 | **no** | 1 x sei_below_threshold, 4 x none | 0.000..0.999 (0.378 ± 0.464) | 51.8..57.2 |
| ensemble10 | 10 | 1 | 1 | 8 | **no** | 1 x sei_below_threshold, 8 x none, 1 x core_temp_warning | 0.000..0.999 (0.278 ± 0.425) | 46.0..62.3 |
| fullanchor | 1 | 0 | 1 | 0 | yes | 1 x core_temp_warning | 0.930..0.930 (0.930 ± 0.000) | 63.9..63.9 |
| fullbudget | 1 | 1 | 0 | 0 | yes | 1 x sei_below_threshold | 0.445..0.445 (0.445 ± 0.000) | 34.6..34.6 |
| grid_newton | 3 | 0 | 0 | 3 | yes | 3 x none | 0.000..0.999 (0.333 ± 0.471) | 52.3..56.4 |
| grid_thermal | 3 | 0 | 0 | 3 | yes | 3 x none | 0.000..0.978 (0.326 ± 0.461) | 48.5..53.3 |
| grid_both | 3 | 0 | 1 | 2 | **no** | 1 x core_temp_warning, 2 x none | 0.000..0.999 (0.333 ± 0.471) | 48.2..62.8 |
| fixed_12 | 3 | 0 | 0 | 3 | yes | 3 x none | 0.624..0.961 (0.827 ± 0.146) | 42.4..54.4 |
| fixed_18 | 3 | 1 | 0 | 2 | **no** | 2 x none, 1 x sei_below_threshold | 0.789..0.897 (0.829 ± 0.048) | 39.6..44.8 |
| fixed_21 | 3 | 0 | 0 | 3 | yes | 3 x none | 0.000..0.997 (0.332 ± 0.470) | 31.2..41.1 |

## Run-to-run comparisons

### seeds5 vs fullanchor (seed 0 of each run)

| metric | seeds5 | fullanchor | delta (fullanchor − seeds5) |
|---|---|---|---|
| train MAE (°C) | 1.78 | 1.85 | +0.07 |
| train RMSE (°C) | 2.32 | 2.46 | +0.14 |
| held-out MAE (°C) | 4.97 | 4.93 | -0.04 |
| held-out RMSE (°C) | 7.09 | 6.87 | -0.22 |
| core T max (°C, unvalidated) | 53.91 | 63.87 | +9.96 |
| SEI min | 0.998 | 0.968 | -0.030 |
| log10 A_sei | 9.550 | 16.616 | +7.066 |

| held-out battery | seeds5 | fullanchor | delta |
|---|---|---|---|
| B0025 | 5.05 | 4.17 | -0.88 |
| B0030 | 1.21 | 2.51 | +1.30 |
| B0036 | 9.05 | 8.86 | -0.18 |
| B0042 | 1.42 | 1.70 | +0.28 |
| B0047 | 1.07 | 1.16 | +0.09 |
| B0056 | 2.64 | 2.31 | -0.33 |

### seeds5 vs fullbudget (seed 0 of each run)

| metric | seeds5 | fullbudget | delta (fullbudget − seeds5) |
|---|---|---|---|
| train MAE (°C) | 1.78 | 1.47 | -0.31 |
| train RMSE (°C) | 2.32 | 2.01 | -0.31 |
| held-out MAE (°C) | 4.97 | 5.76 | +0.79 |
| held-out RMSE (°C) | 7.09 | 8.31 | +1.21 |
| core T max (°C, unvalidated) | 53.91 | 34.60 | -19.30 |
| SEI min | 0.998 | 0.423 | -0.575 |
| log10 A_sei | 9.550 | 11.547 | +1.997 |

| held-out battery | seeds5 | fullbudget | delta |
|---|---|---|---|
| B0025 | 5.05 | 3.78 | -1.27 |
| B0030 | 1.21 | 1.03 | -0.18 |
| B0036 | 9.05 | 10.86 | +1.82 |
| B0042 | 1.42 | 1.19 | -0.23 |
| B0047 | 1.07 | 1.60 | +0.53 |
| B0056 | 2.64 | 3.17 | +0.53 |

### seeds5 vs fixed_12 (means over 5 and 3 seeds)

| metric | seeds5 | fixed_12 | delta (fixed_12 − seeds5) |
|---|---|---|---|
| train MAE (°C) | 1.82 | 1.78 | -0.05 |
| train RMSE (°C) | 2.45 | 2.37 | -0.08 |
| held-out MAE (°C) | 4.86 | 4.97 | +0.11 |
| held-out RMSE (°C) | 6.83 | 6.96 | +0.13 |
| core T max (°C, unvalidated) | 54.48 | 47.23 | -7.25 |
| SEI min | 0.467 | 0.949 | +0.483 |
| log10 A_sei | 14.466 | 12.000 | -2.466 |

| held-out battery | seeds5 | fixed_12 | delta |
|---|---|---|---|
| B0025 | 4.56 | 4.32 | -0.24 |
| B0030 | 2.54 | 2.17 | -0.37 |
| B0036 | 8.65 | 9.08 | +0.43 |
| B0042 | 1.57 | 1.32 | -0.25 |
| B0047 | 1.42 | 1.09 | -0.32 |
| B0056 | 2.21 | 2.42 | +0.21 |

### seeds5 vs fixed_18 (means over 5 and 3 seeds)

| metric | seeds5 | fixed_18 | delta (fixed_18 − seeds5) |
|---|---|---|---|
| train MAE (°C) | 1.82 | 1.80 | -0.03 |
| train RMSE (°C) | 2.45 | 2.39 | -0.06 |
| held-out MAE (°C) | 4.86 | 5.04 | +0.19 |
| held-out RMSE (°C) | 6.83 | 7.01 | +0.18 |
| core T max (°C, unvalidated) | 54.48 | 41.76 | -12.72 |
| SEI min | 0.467 | 0.888 | +0.421 |
| log10 A_sei | 14.466 | 18.000 | +3.534 |

| held-out battery | seeds5 | fixed_18 | delta |
|---|---|---|---|
| B0025 | 4.56 | 4.26 | -0.30 |
| B0030 | 2.54 | 2.20 | -0.34 |
| B0036 | 8.65 | 9.07 | +0.41 |
| B0042 | 1.57 | 1.48 | -0.09 |
| B0047 | 1.42 | 1.40 | -0.01 |
| B0056 | 2.21 | 2.57 | +0.37 |

### seeds5 vs fixed_21 (means over 5 and 3 seeds)

| metric | seeds5 | fixed_21 | delta (fixed_21 − seeds5) |
|---|---|---|---|
| train MAE (°C) | 1.82 | 1.86 | +0.04 |
| train RMSE (°C) | 2.45 | 2.49 | +0.04 |
| held-out MAE (°C) | 4.86 | 5.04 | +0.19 |
| held-out RMSE (°C) | 6.83 | 6.93 | +0.10 |
| core T max (°C, unvalidated) | 54.48 | 34.62 | -19.86 |
| SEI min | 0.467 | 0.333 | -0.133 |
| log10 A_sei | 14.466 | 21.000 | +6.534 |

| held-out battery | seeds5 | fixed_21 | delta |
|---|---|---|---|
| B0025 | 4.56 | 4.56 | +0.00 |
| B0030 | 2.54 | 3.28 | +0.74 |
| B0036 | 8.65 | 8.89 | +0.23 |
| B0042 | 1.57 | 1.42 | -0.15 |
| B0047 | 1.42 | 1.52 | +0.10 |
| B0056 | 2.21 | 2.39 | +0.19 |

## Notes

How to read these numbers (reduced-budget baseline; see Protocol):

- **Same budget, more data, no penalty.** With identical settings the dynamic floor trains on
  1718 instead of 1076 cycles and still reaches a lower training MAE (9.6 vs 11.4 °C, seed
  spread ±1.4 / ±2.0) and a lower held-out MAE (6.6 vs 8.7 °C). The held-out figures are not
  like-for-like, though: the dynamic held-out set contains B0056 (101 cycles at 4 °C, the
  easiest battery at 1.7 °C MAE) and 48 rather than 11 cycles of B0047. On the five batteries
  both runs hold out, the per-battery errors are within the seed spread of each other (B0025
  10.4 vs 10.0, B0030 21.9 vs 22.0, B0036 7.2 vs 6.9, B0042 4.9 vs 4.7 °C); B0047 is higher in
  the dynamic run (3.4 vs 2.4 °C) because that run also evaluates its 37 later, lower-capacity
  cycles. The corrected floor therefore changes *coverage* -- all six designated held-out
  batteries, including the 4 °C / 2.7 V cell that motivated the fix -- without costing accuracy
  at equal budget.
- **Both runs are far from converged.** 1000 Adam iterations leave a training MAE around
  10 °C; the calibrated `log10 A_sei` (22.85-22.86) is still at its initial value (22.88), i.e.
  the SEI rate parameter has not started to move. Treat every absolute number here as a
  pipeline baseline, not as the model's accuracy. CHANGELOG items 17-20 were obtained at the
  full budget on the legacy 19/5 tables and are not comparable with this table.
- **B0030 dominates the held-out error in both runs** (22 °C MAE). It is the only 43 °C
  battery held out, and only three 43 °C batteries (120 cycles) remain in training, so the
  warm-ambient regime is the least constrained part of the input domain; this is the same
  uneven generalisation noted in CHANGELOG items 17-20 and is independent of the floor.
- **Held-out MAE is below training MAE** in both runs (gap about -3 °C). The held-out set is
  not harder than the training set for an under-trained model; do not read the negative gap as
  evidence of generalisation.
- **Verdict spread.** Legacy: 1 DANGER / 2 SAFE; dynamic: 3 SAFE. With three seeds and
  under-trained models this says little about the physics -- the data-driven SEI threshold
  moves with each seed (0.69-0.83) and the DANGER call in legacy seed 1 comes from `SEI min`
  (0.732) falling just under that seed's threshold (0.738). It does reproduce the instability
  that motivated the ensemble check (CHANGELOG item 20).
- **Rows dropped on load**: 59-61 training cycles and 1 held-out cycle in each run have no
  impedance measurement before them (no `Re`) and are excluded by the loader; the per-battery
  coverage report printed by `train` / `evaluate` lists them.

### The converged run (`dynamic-subset`) and what it says about A_sei

- **It converges.** With the anchor subset the same 24/6 tables train to a 1.86 ± 0.02 °C
  training MAE and 4.73 ± 0.14 °C held-out MAE in 2000 Adam + 200 L-BFGS iterations, against
  9.6 / 6.6 °C for the under-trained full-anchor run. The seed spread collapses (± 0.02 °C on
  train), B0030 goes from 22 °C to 2.7 °C, and a real generalisation gap appears (+2.9 °C:
  held-out is now harder than training, as it should be). B0036 (8.4 °C) is the weakest
  held-out battery; it is the one with the outlier 2.44 Ah best cycle that compresses its SoH scale.
- **Almost all of the converged loss is the data term** (`observe_T` 94 %); the SEI residual is
  5e-5 (0 %). During the Adam phase it was the opposite picture: `pde_sei` held ~25-30 % of the
  total while `log10(A_sei)` crept from 21.88 to 21.85 at learning rate 1e-4 (see
  `loss_history_seed*.csv`); L-BFGS then moved it by decades in a few hundred evaluations.
- **A_sei is not identifiable from this data.** The three seeds started at 21.9 / 22.9 / 23.9
  and finished at 14.7 / 20.3 / 16.5 -- a 5.6-decade spread -- with indistinguishable
  temperature fits (1.85-1.89 °C). Every seed moved *down*: making `A_sei` small makes the
  SEI rate, and with it the SEI residual, vanish for any temperature field, so the optimiser
  switches the reaction off rather than calibrating it. Surface temperature during normal
  cycling simply carries no information about the SEI frequency factor (the SEI heat
  `H_sei * rate` is negligible next to Joule heating at these temperatures). The "calibrated
  A_sei" printed after training is therefore an artefact of the initial value and the optimiser
  path, and the abuse-test `SEI min` inherits that arbitrariness: 0.27 ± 0.39 across seeds,
  verdicts 1 WARNING / 2 SAFE. The core-temperature part of the verdict (54 ± 6 °C) is the
  only component grounded in the data.
- The earlier `legacy` and `dynamic` runs show the other failure mode of the same fact: at
  lr 1e-4 without L-BFGS, `A_sei` never left its initial value (shift 0.04 decades), which
  looks like agreement across seeds but is just the initialisation.

### Validation against the Databricks baseline (seeds5, fullanchor, fullbudget)

**What "the Databricks figures" are.** The Databricks runs left no numeric accuracy record in
this repository: MLflow metrics were never exported, the notebook is stored without outputs,
and `docs/CHANGELOG.md` reports them qualitatively. The recorded, comparable facts are: (i) the
weighted `ic_T` loss component plateaued at 0.849 before the Celsius/Kelvin fix and reached
**0.00235** after it (item 13); (ii) `A_sei` was "frozen dead flat" at its initial value for
15000 Adam steps before the gradient fix (item 9) and moved afterwards; (iii) held-out error
was "uneven across batteries" with at least one outlier battery (items 17, 20); (iv) the
abuse-test verdict flipped between DANGER and SAFE across otherwise identical trainings
(item 20). The comparisons below are therefore against those facts, not against MAE values.

**Seed variance (seeds5).** Five seeds from the same initial `A_sei`: train MAE
1.82 ± 0.04 °C (range 1.78-1.87), held-out MAE 4.86 ± 0.11 °C (4.65-4.97), generalisation gap
+3.0 ± 0.1 °C; per-battery held-out σ between 0.12 °C (B0042) and 0.73 °C (B0030). The
temperature model is reproducible to a few hundredths of a degree on training data and about
a tenth of a degree held out. Everything that depends on `A_sei` is not: final `log10(A_sei)`
ranges from 9.6 to 20.4 (10.85 decades) for the same fit quality, `SEI min` is 0.47 ± 0.44,
and the verdict is 1 DANGER / 4 SAFE. The single DANGER (seed 0, `A_sei` ≈ 4e9) is a pure
artefact: with the reaction switched off the SEI output is 1.000 everywhere, the data-driven
threshold (mean − 3σ) becomes 0.999, and a rounding-level dip trips it. This reproduces the
Databricks fact (iv) and explains it.

**The `ic_T` figure.** The converged runs' weighted `ic_T` term is 1.1-1.6e-3 (seeds5 and
dynamic-subset, see the loss-term table), the same order as the Databricks 0.00235 -- and far
below the 0.849 plateau of the pre-fix notebook -- so the local pipeline sits on the fixed
(v6+) side of that history.

**Anchor subset vs full anchors (fullanchor).** Seed 0 trained with all 85900 observation
points as anchors and data-fit points (the notebook's choice) against the same seed with the
2000-point subset: train MAE 1.85 vs 1.78 °C (+0.07), held-out MAE 4.93 vs 4.97 °C (−0.04),
held-out RMSE 6.87 vs 7.09 °C. Both differences are within the five-seed spread of the subset
protocol (σ 0.04 °C train, 0.11 °C held-out), and the per-battery deltas (−0.9 to +1.3 °C,
largest on B0030 whose seed σ is 0.73 °C) are of the same size as seed-to-seed variation. The
Adam-phase loss trajectories of the two runs track each other step for step (`val_*.log`,
e.g. step 1300: observe_T 1.48 vs 1.25, ic_T 0.30 vs 0.30, pde_sei 0.66 vs 0.61). The subset
therefore reproduces the full-anchor result to within seed noise -- at ~7 min instead of
3 h 40 min per seed on this machine (the full-anchor run had to share the CPU for part of
that time; alone it is ~2.5 s per Adam step, and its 200 L-BFGS iterations took 75 min).
What the subset does *not* reproduce is `A_sei` (9.6 vs 16.6) and hence the SEI-dependent
abuse-test outputs (core T max 54 vs 64 °C at the abuse conditions, SEI min 0.998 vs 0.968);
that is the non-identifiability again, not the anchor set -- two subset seeds differ by as much.

**Bottom line.** Every quantity the data determine -- temperature fit, held-out error, its
per-battery structure, the `ic_T` loss level, the overfitting with longer training, the
verdict instability -- is reproduced by the local pipeline with the subset protocol, and the
five-seed run bounds its variance at about ±0.1 °C held out. The quantities it cannot
reproduce, `A_sei` and the SEI half of the verdict, are the ones no pipeline can reproduce
from this data, as item 25 established.

**Full budget (fullbudget).** The notebook's 15000 Adam + 2000 L-BFGS with 5000/1000
collocation points (and the 2000-point subset; ~1.5 h on one CPU core) takes seed 0 from a
train MAE of 1.78 °C (2000 + 200 budget) to **1.47 °C**, but the held-out MAE goes the other
way, 4.97 → **5.76 °C**: B0025, B0030 and B0042 improve (−1.3, −0.2, −0.2 °C) while B0036
(+1.8 °C, to 10.9), B0047 and B0056 (+0.5 °C each) get worse. Seven times more optimisation
buys a better description of the training batteries and a worse one of unseen batteries --
the generalisation gap grows from +3.2 to +4.3 °C. The converged short budget is therefore not
a compromise forced by the CPU; for held-out accuracy it is at least as good as the notebook's
budget, and the seed variance of 0.11 °C (seeds5) is small against that +0.8 °C shift. The
Databricks pipeline reported exactly this pattern qualitatively (item 17: "a meaningfully
larger held-out error than train error is a direct sign of overfitting to the training
batteries"); B0036 is the outlier battery both here and there. `A_sei` again ran away
(22.9 → 11.5) and the verdict is DANGER on the SEI criterion (SEI min 0.423 < threshold 0.445)
with a core temperature of only 34.6 °C -- one more instance of the verdict's SEI half being
decided by a non-identifiable parameter. The end-of-Adam `ic_T` term is 0.024 and the final
one 1.3e-3, against the Databricks 0.00235.

### Databricks v9 reference, per battery

The v9 figures (train 1.2 °C, held-out 7.65 °C; B0025 4.2, B0030 1.1, B0036 11.4, B0042 0.9,
B0047 7.3 °C) are the only numeric record of the Databricks pipeline. They were obtained on the
legacy 19/5 tables with every observation point as anchor and the full 15000 + 2000 budget, so
the honest comparison is per battery and against the local runs of the same kind:

- **B0036 and B0025, the two hard batteries, are hard everywhere**: v9 11.4 / 4.2 °C, local
  converged baseline 8.7 / 4.6 °C, local full budget 10.9 / 3.8 °C. B0036 is the battery with
  the outlier 2.44 Ah best cycle; the longer the training, the worse it gets (8.7 → 10.9 °C),
  which is the v9 picture.
- **B0030 and B0042 are easy everywhere** (v9 1.1 / 0.9 °C; local 2.5 / 1.6 °C at the short
  budget, 1.0 / 1.2 °C at the full budget).
- **B0047 is where the tables differ**: v9 7.3 °C on 11 early cycles (legacy floor), local
  1.4–1.6 °C on 48 cycles (dynamic floor). The v9 held-out set also lacked B0056 entirely.
- **Train 1.2 °C vs local 1.82 ± 0.04 °C (short budget) / 1.47 °C (full budget)**: the gap is
  the budget, and closing it costs held-out accuracy (see *Full budget*). The v9 overall
  held-out 7.65 °C is not comparable with the local 4.86 °C: different held-out sets (5 vs 6
  batteries, different B0047 cycles), and the v9 mean is dominated by B0036 and B0047.

### Per-ambient errors

See the per-ambient tables: the 4 °C group (B0047, B0056 held out) and the 43 °C group (B0030)
are fitted to 1.5–2.6 °C; the 24 °C group carries the error (B0025, B0036, the two worst
batteries, are both 24 °C cells), so the residual held-out error is not an ambient-coverage
problem but a battery-specific one.

### Physics options grid (negative result)

Three seeds per option against the five-seed baseline, judged by the fixed rule (worst
held-out batteries must improve by more than the seed spread, no other battery may degrade
beyond its spread). **All three are rejected and stay off**:

- *Newton cooling*: every held-out battery within ±0.2 °C of the baseline (spreads 0.16–0.73).
- *Learnable thermal constants*: `k_cond` 1.0 → 0.65–0.71, `ρ·Cp` 2.0e6 → 2.5–3.8e6 (consistent
  across seeds, so these are identified by the data, unlike `A_sei`), but B0025 and B0036
  improve by only 0.18 and 0.20 °C against spreads of 0.25 and 0.34 °C.
- *Both*: one seed fits training better (1.68 °C) and held-out worse (B0025 5.45 °C); means
  within spread.

### Fixed A_sei (log10 = 12, 18, 21; three seeds each)

Freezing `log10(A_sei)` changes the held-out error by less than the seed spread: 4.97 ± 0.09,
5.04 ± 0.04 and 5.04 ± 0.14 °C against 4.86 ± 0.11 °C with `A_sei` free (train 1.78–1.86 vs
1.82 °C; per battery every delta is within 0.3 °C). Nine decades of `A_sei` are invisible in
the temperature fit -- the direct confirmation of the non-identifiability found in item 25.
What *does* change is the abuse test: the unvalidated core-temperature maximum at the abuse
conditions slides from 47 ± 5 °C (12) to 42 ± 2 °C (18) to 35 ± 5 °C (21), and the verdict
flips from 3 SAFE to 1 DANGER / 2 SAFE (seed 2 at 18: SEI min 0.895 against a threshold of
0.897) and back to 3 SAFE. A parameter the data cannot see therefore decides part of the
verdict, which is why both caveats -- core temperature unvalidated, verdict excluded above
45 °C ambient -- are attached to every verdict in this repository.

### Ten-model abuse-test ensemble (`ensemble10`, seeds 0–9 of the baseline protocol)

1 DANGER / 1 WARNING / 8 SAFE. The DANGER is decided by the SEI criterion in its degenerate
form (seed 0: `A_sei` ≈ 4e9, SEI output ≡ 1, threshold `mean − 3σ` = 0.999, SEI min 0.998); the
WARNING by a core temperature of 62.3 °C (seed 6; unvalidated). SEI thresholds span
0.000–0.999 (mean 0.28 ± 0.43): for the models whose `A_sei` ran to ~1e20 the clipped SEI output
is 0 everywhere and the threshold is 0, for those at ~1e10 it is 1 everywhere and the threshold
is 0.999 -- there is no calibrated middle. Core-temperature maxima range 46–62 °C. Eight of ten
models agree on SAFE; the two exceptions are artefacts of `A_sei`, not evidence about the cell.

