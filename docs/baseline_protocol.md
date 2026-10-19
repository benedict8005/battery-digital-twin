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
