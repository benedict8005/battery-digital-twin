# Battery PINN Digital Twin

[![tests](https://github.com/benedict8005/battery-digital-twin/actions/workflows/tests.yml/badge.svg)](https://github.com/benedict8005/battery-digital-twin/actions/workflows/tests.yml)

A physics-informed neural network (PINN, built with DeepXDE + PyTorch) that models a lithium-ion
cell's temperature rise and SEI (solid electrolyte interphase) chemical stability during
discharge, trained on the NASA battery-cycling dataset. It is aimed at students and engineers
who want a runnable, tested example of physics-informed modelling of battery degradation.

> **Status: research prototype.** Internal/core temperature is **unvalidated**: it is inferred
> through the physics and has never been compared with a real internal sensor. The "abuse test"
> verdict is a heuristic derived from the model's own outputs, is **excluded above 45 °C
> ambient** (outside the trained domain), and its SEI half depends on a parameter the data do
> not identify (see below). It is not a certified safety limit. Do not use it for real safety
> decisions.

## Requirements

- **Python 3.10 or newer** (3.12 recommended).
- PyTorch, DeepXDE, NumPy, pandas, SciPy, Matplotlib -- all installed by `pip install -e ".[dev]"`
  (or `pip install -r requirements.txt`).
- For real data: the cleaned NASA battery dataset on local disk (see [Data layout](#data-layout)).
  The synthetic demo needs no data at all.

Everything runs locally on CPU (a GPU is used automatically if PyTorch finds one). There is no
Spark, MLflow, Delta or cloud dependency anywhere in `src/`, and the test suite enforces that.

## Quick start: local demo

Run all commands from the folder that contains `pyproject.toml`, not from inside `.venv`.

macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
python --version        # must show 3.10 or newer
pip install -e ".[dev]"
python -m battery_pinn demo --workdir .
```

Windows (PowerShell):

```powershell
py -0p                          # list installed Pythons; install 3.12 from python.org if none is 3.10+
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                # must show 3.10 or newer
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m battery_pinn demo --workdir .
```

If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope Process Bypass` and try again.

The demo generates synthetic data, trains 2 seeds briefly, evaluates on held-out batteries,
and runs the abuse test (output goes to `data/` and `artifacts/`, both git-ignored). It is
deliberately short, so the model is under-trained and the numbers are only a smoke test.
It takes about 5 minutes on a CPU-only machine; times vary. **Synthetic data says
nothing about real cells.**

## Quick start: real NASA data

1. Download the battery data set from NASA (six zip files of `B00xx.mat`) and unpack the
   `.mat` files into one folder, e.g. `data/nasa_mat/`.
2. Build the processed tables -- `prepare-data` accepts the `.mat` folder directly (it converts
   it into `data/processed/raw_csv/` first) or the per-cycle CSV layout:

   ```bash
   python -m battery_pinn prepare-data --raw data/nasa_mat --out data/processed
   ```

   This is the local port of the former Spark ETL. It writes `data/processed/train.csv`
   (24 training batteries) and `data/processed/held_out.csv` (6 held-out batteries), plus the
   silver and gold tables for inspection. It prints the cycle-processing summary and, for
   **every** designated battery, how many cycles were kept and how many were dropped for which
   reason (file missing, too short, low capacity), together with that battery's capacity floor
   and best cycle. A designated battery that is absent from the metadata is reported as such. If
   a **held-out** battery is missing or ends up with zero surviving cycles the command stops with
   a `BatteryCoverageError` and writes nothing; a missing training battery is only a warning.
3. Train, evaluate and run the abuse test. The defaults are the processed tables above and the
   validated fast protocol (2000 Adam + 200 L-BFGS iterations, `--anchor-subset 2000`,
   `num_domain` 1000, `num_boundary` 200, ten seeds -- about 5 minutes per seed on a 4-core CPU):

   ```bash
   python -m battery_pinn train                 # ten seeds -> artifacts/battery_pinn_model_seed*.pt
   python -m battery_pinn evaluate              # train vs held-out error, per battery
   python -m battery_pinn abuse-test            # ten-model ensemble verdict
   ```

   The notebook's protocol is `--iterations 15000 --lbfgs-iterations 2000 --num-domain 5000
   --num-boundary 1000 --anchor-subset all`; it is ~10 h per seed on a CPU and, on the held-out
   batteries, no better (see `docs/RESULTS.md`).

`train`, `evaluate` and `abuse-test` print a per-battery report when they load a table (cycles
and rows kept after dropping rows with missing values, e.g. cycles with no impedance
measurement). `evaluate` additionally requires every designated held-out battery
(`HELD_OUT_BATTERIES`, or `--held-out-batteries ...`) to be present with surviving cycles and
fails otherwise; pass `--no-battery-check` for data whose battery IDs are not designated, such
as the synthetic set.

The synthetic equivalent of step 2 is `python -m battery_pinn generate-data --out data/synthetic`,
which writes `train.csv` / `held_out.csv` with the same columns (point the other commands at
them with `--data` / `--held-out-data` and evaluate with `--no-battery-check`).

### The abuse test: a ten-model ensemble

`abuse-test` restores every checkpoint in `--checkpoints`, sweeps the ambient input from 25 to
150 °C at SoH 0.8 / Re 0.5 / full exposure for each model, and reports per model the verdict
**and the criterion that decided it** (core temperature > 80 °C, SEI stability below the
model's own data-driven threshold, both, core temperature > 60 °C for WARNING, or none), then
for the ensemble the DANGER / WARNING / SAFE counts, the deciding-criterion counts and the
spread of the SEI threshold (`mean − 3σ` of each model's in-domain SEI output; it moves with
the seed). Everything is written to `abuse_test_summary.json` and `abuse_test.png`. Two caveats
are printed with every result and stated wherever the numbers appear: **core temperature is
unvalidated** (inferred through the physics, never compared with an internal sensor) and the
verdict **excludes ambient > 45 °C** (extrapolation beyond the trained domain; the plot shows it
dashed). The intended ensemble is ten seeds (`config.ABUSE_ENSEMBLE_SIZE`; `--min-models`
enforces a minimum).

### Opt-in physics variants

Both are off by default, and a test asserts that the default residual is exactly the original
PDE (`test_default_options_reproduce_the_original_residual_exactly`):

- `train --newton-cooling` adds a Newton surface heat-loss sink `−h (T − T_ambient)` to the heat
  sources (`NEWTON_COOLING_COEFF`);
- `train --learn-thermal` makes `k_cond`, `ρ·Cp` (and `h`) trainable log10 variables, reported
  in `train_summary.json` and stored in the checkpoint's `physics` entry;
- `train --fix-log-a-sei X` freezes `log10(A_sei)` at `X` instead of training it.

`docs/RESULTS.md` (section *Physics options grid*) records a 3-seed grid of these options
against the baseline with the decision rule used: an option is adopted only if it lowers the
held-out MAE of the worst batteries by more than the seed spread without degrading any other
battery beyond its spread; otherwise the negative result is recorded and the option stays off.
Outcome on the NASA data (3 seeds each against the 5-seed baseline): **all three rejected**.
Newton cooling changes nothing (every held-out battery within ±0.2 °C, inside the spread);
learnable constants move `k_cond` from 1.0 to 0.65–0.71 and `ρ·Cp` from 2.0e6 to 2.5–3.8e6 but
lower the two worst batteries by only 0.18–0.20 °C against spreads of 0.25–0.34 °C; the
combination helps one seed's training fit (1.68 °C) and hurts held-out (B0025 5.45 °C on that
seed). None of them touches the B0036 problem (8.5–8.7 °C). The defaults stay off.

#### Making training converge on a CPU: `--anchor-subset`

DeepXDE evaluates the PDE residual -- with its Jacobian and Hessian -- on *every* input
point, including the temperature observation points that only feed the data-fit term. With
the notebook's `anchors = all observation points` that is ~86k rows twice per iteration,
about 2-3 s per Adam step on a 4-core CPU, i.e. ~10 h for the notebook's 15000 + 2000
iterations per seed; that is why the first baseline (`docs/RESULTS.md`, runs `legacy` and
`dynamic`) had to stop at 1000 iterations and was under-trained.

`train --anchor-subset N` (now the default, with N = 2000; `--anchor-subset all` restores the
notebook's behaviour) uses a seeded random subset of `N` observation points as the PDE anchors
*and* the data-fit points. A few thousand points cover
every battery and operating condition of the 86k-row table, the per-iteration cost falls to a
few tenths of a second, and evaluation still uses every row. The run `dynamic-subset` in
`docs/RESULTS.md` used `--anchor-subset 2000 --num-domain 1000 --num-boundary 200` with
2000 Adam + 200 L-BFGS iterations (under 20 min per seed here). `--num-initial N` adds explicit
`t = 0` points for the initial conditions (the notebook relied on the ~1 % of random points
with `t <= 0.01`, so tiny point budgets can end up with none). A failing L-BFGS line search (a
PyTorch error on non-finite losses) no longer aborts the run: the Adam result is kept and the
error is recorded in `train_summary.json`.

#### Loss-term logging

Every `--display-every` steps the five weighted loss terms (`pde_thermal`, `pde_sei`, `ic_T`,
`ic_c`, `observe_T`) and the current `log10(A_sei)` are appended to
`artifacts/loss_history_seed<k>.csv`; the final terms, their shares of the total and the
dominant term are printed and stored in `train_summary.json`, and `make_results_table.py`
tabulates them per run. This is what tells you whether the SEI residual dominates the loss.

#### Is `A_sei` identifiable?

`log10(A_sei)` is a trainable variable, so an ensemble can test whether the data actually pin
it down. `train --a-sei-init-spread D` starts the seeds evenly across `init ± D` decades (the
default init, 22.88, makes the SEI rate O(1) at 35 °C); `train_ensemble` then writes
`identifiability.json` and prints the verdict from `battery_pinn.identifiability`: **not
identifiable (stuck)** if no seed moved more than 0.1 decades from its start, **not
identifiable (seed-dependent)** if the final values span more than 0.5 decades,
**identifiable** if seeds that started apart converged to within 0.5 decades, and merely
**consistent across seeds** if they agree but all started from the same value.

**Finding on the NASA data (`docs/RESULTS.md`, run `dynamic-subset`): `A_sei` is not
identifiable.** Three seeds started at `log10(A_sei)` = 21.9 / 22.9 / 23.9 and converged to
14.7 / 20.3 / 16.5 -- 5.6 decades apart -- with indistinguishable temperature fits
(train MAE 1.85-1.89 °C, held-out 4.6-4.9 °C). Every seed drifted *downwards*: a small
`A_sei` zeroes the SEI rate and with it the SEI residual for any temperature field, so the
optimiser switches the reaction off instead of calibrating it. Surface temperature during
normal cycling carries no information about the SEI frequency factor (the SEI heat term is
negligible next to Joule heating there). Consequences: the `A_sei` printed after training is
an artefact of the initial value and optimiser path, the SEI-stability half of the abuse-test
verdict inherits that arbitrariness (SEI min 0.27 ± 0.39 across seeds, verdicts
1 WARNING / 2 SAFE), and only the core-temperature part of the verdict is data-grounded.
Identifying `A_sei` would need data where the SEI reaction is thermally visible (abuse /
elevated-temperature tests) or an independent prior on it. Freezing it (`--fix-log-a-sei`
12 / 18 / 21, three seeds each) leaves the held-out error within the seed spread (4.97 / 5.04 /
5.04 vs 4.86 ± 0.11 °C) but moves the abuse outputs: the unvalidated core-temperature maximum
goes 47 → 42 → 35 °C and the verdict flips 3 SAFE → 1 DANGER / 2 SAFE → 3 SAFE. The ten-model
ensemble on the baseline protocol gives 1 DANGER / 1 WARNING / 8 SAFE with SEI thresholds
spanning 0.000–0.999 (`docs/RESULTS.md`). In the two earlier under-trained
runs `A_sei` never left its initial value (shift 0.04 decades) -- which looks like
cross-seed agreement but is the other face of the same insensitivity.

`train` (and `demo`) take `--device auto|cpu|cuda`. `auto` (the default) uses CUDA when PyTorch
reports it *and* a probe tensor op succeeds, otherwise the CPU; `cuda` does the same but prints
a warning when it has to fall back; `cpu` forces the CPU. The choice is applied as PyTorch's
default device so that everything DeepXDE builds lands on it, and checkpoints are always saved
as CPU tensors so they load anywhere. Apple MPS is never selected automatically.

`train`, `evaluate` and `abuse-test` print the same kind of per-battery report when they load a
table (cycles and rows kept after dropping rows with missing values, e.g. cycles with no
impedance measurement). `evaluate` additionally requires every designated held-out battery
(`HELD_OUT_BATTERIES`, or `--held-out-batteries ...`) to be present with surviving cycles and
fails otherwise; pass `--no-battery-check` for data whose battery IDs are not designated, such
as the synthetic set.

The synthetic equivalent of step 2 is `python -m battery_pinn generate-data --out data/synthetic`,
which writes `train.csv` / `held_out.csv` with the same columns (evaluate them with
`--no-battery-check`).

### What `prepare-data` does

The same steps as the original pipeline, in the same order, with pandas and NumPy:

| Layer | Step |
|---|---|
| bronze | read `metadata.csv` (column names matched case-insensitively) |
| silver | drop `EXCLUDED_BATTERIES`; parse the numpy-array-style `start_time` (plain or scientific notation) into a timestamp; number each battery's discharge (and impedance) rows chronologically |
| gold | attach to each discharge the impedance measurement (`Re`, `Rct`) taken at or before it; `capacity`/`Re`/`Rct` rounded to float32 as the `FLOAT` columns were |
| per cycle | truncate at the battery's cutoff voltage (`CUTOFF_VOLTAGE_MAP`, default 2.7 V); skip cycles shorter than `MIN_SAMPLES_PER_CYCLE` or at/below the capacity floor (see below); coulomb-counting SoC; SoH = capacity / battery max capacity; average into `BINS_PER_CYCLE` bins; `Time_normalized` |
| split | by battery: `HELD_OUT_BATTERIES` -> held-out table, everything else -> training table |

Every name in capitals is a value in `src/battery_pinn/config.py`. The test suite runs the
original feature-engineering code (with Spark stubbed out) next to the port and asserts
bit-identical tables. One thing the Spark version left undefined -- which impedance row wins
when several precede a discharge -- is now fixed as "the most recent one".

### The capacity floor and the 4 °C batteries (why B0056 used to vanish)

The original pipeline skipped any cycle whose measured capacity was at or below
`MIN_VALID_CAPACITY_AH = 1.4` Ah. That number is NASA's end-of-life criterion -- 30 % fade of
the 2 Ah *nominal* capacity -- not a data-quality threshold. Batteries B0053-B0056 (and B0041,
B0045) were cycled at 4 °C, where a healthy cell reaches its cutoff voltage after delivering
only 1.1-1.35 Ah; B0056's *best* cycle is 1.344 Ah. So the floor removed every cycle of six
batteries, including held-out B0056, and most of B0046-B0048 (held-out B0047 kept 11 of 72).
It was also stricter than its own intent for the room-temperature cells, whose best cycles are
1.8-1.9 Ah rather than 2 Ah.

The floor is now `config.capacity_floor(best_cycle)`:
`min(MIN_VALID_CAPACITY_AH, MIN_CAPACITY_FRACTION_OF_MAX x best cycle of that battery)`, i.e.
a cycle is kept if it clears the nominal end-of-life capacity **or** has faded less than 30 %
from what that battery actually delivered. It never drops a cycle the old rule kept (B0036's
outlier best cycle of 2.44 Ah would otherwise have *tightened* its floor), the aberrant
near-zero-capacity runs NASA mentions are still dropped, and setting
`MIN_CAPACITY_FRACTION_OF_MAX = None` restores the old rule exactly. On the real dataset this
keeps 2257 of 2694 cycles instead of 1466, gives every designated battery data (B0056:
101/102) and makes the split a genuine 24/6 for the first time. Full before/after counts are
in `docs/CHANGELOG.md` item 22.

### Baseline results and the results table

`docs/RESULTS.md` compares the legacy 1.4 Ah capacity floor (19 training / 5 held-out
batteries) with the corrected dynamic floor (24 / 6) on the real NASA data across several
seeds: checkpoint metrics (calibrated `A_sei`, train / held-out MAE and RMSE), per-battery
errors and the abuse-test verdict spread. It is produced by

```bash
scripts/run_baseline.sh data/raw            # prepare both splits, train 3 seeds each, write docs/RESULTS.md
python scripts/make_results_table.py --run legacy runs/legacy/artifacts runs/legacy/data \
                                     --run dynamic runs/dynamic/artifacts runs/dynamic/data \
                                     --out docs/RESULTS.md --json docs/RESULTS.json
```

`make_results_table.py` restores every `battery_pinn_model_seed*.pt` in each run folder and
evaluates it on that run's processed tables; any number of runs can be compared side by side,
and `--compare A B` adds a metric-by-metric delta table for a pair of runs.
`scripts/run_validation.sh` adds the validation runs described in `docs/RESULTS.md`: five
seeds at the converged budget (seed variance), the same protocol with all observation points
as anchors (validates `--anchor-subset`), and the notebook's full 15000 + 2000 budget.
Outcome (details in `docs/RESULTS.md`): the subset reproduces the full-anchor result within
seed noise (train MAE 1.78 vs 1.85 °C, held-out 4.97 vs 4.93 °C, same seed) at ~1/30 of the
cost; five seeds give train 1.82 ± 0.04 °C and held-out 4.86 ± 0.11 °C; the notebook's full
15000 + 2000 budget improves the training fit (1.47 °C) but worsens held-out error (5.76 °C),
i.e. it overfits the training batteries, so the short budget is the better choice for
generalisation. The Databricks runs left no numeric accuracy record; the recorded facts they
do provide (the `ic_T` loss level, `A_sei` moving, uneven held-out error, verdict instability)
are all reproduced.
The legacy split is reproduced with `prepare-data --min-capacity-fraction none
--allow-missing-held-out` (the second flag downgrades the missing-B0056 hard error to a
warning; it exists only for such comparison runs).

## Configuration

`src/battery_pinn/config.py` is the single source of truth for physical constants, network and
loss settings, abuse-test thresholds, the battery lists (`TARGET_BATTERIES`,
`HELD_OUT_BATTERIES`, `EXCLUDED_BATTERIES`), the per-battery cutoff voltages, the ETL
thresholds, table names, default paths and the default training protocol
(`TRAIN_DEFAULTS`). It imports only the standard library.

## Troubleshooting

- **`No matching distribution found for torch` (`from versions: none`):** your Python is too old
  (PyTorch has no wheels for Python 3.8 or earlier). Check `python --version`; if it is below 3.10,
  create the environment with a newer interpreter (`py -3.12 -m venv .venv` on Windows).
- **`does not appear to be a Python project`:** you are not in the folder that contains
  `pyproject.toml`. If you unzipped into a folder with the same name, there may be a nested
  `battery-pinn-digital-twin\battery-pinn-digital-twin`; `cd` into the inner one.
- **Old `pip` (20.x):** run `python -m pip install --upgrade pip` first.

## Tests

```bash
python -m pytest
```

Unit tests cover the local ETL (every medallion layer against hand-computed expectations, plus
an exact comparison against the original feature-engineering loop), the `.mat` converter, the
per-battery coverage reports and hard errors (`-k held_out_missing`), interpolation including
near-duplicate breakpoints (`-k near_duplicate`), device selection and fallback, a very short
real training run, the OCV
fit, the Arrhenius numerics, the PDE residuals, checkpoint round-trips, data handling, the
abuse-test verdict logic, the CLI and the configuration. A dedicated test module proves that
nothing under `src/` imports Spark, MLflow or Databricks libraries: it scans every module
statically and then imports the whole package and runs `prepare-data` with those packages made
unimportable. Tests run on CPU in a few seconds; those that need PyTorch/DeepXDE are skipped if
they are not installed.

GitHub Actions (`.github/workflows/tests.yml`) runs the suite on Ubuntu, Windows and macOS with
Python 3.12 (plus Python 3.10 on Ubuntu), installing CPU-only PyTorch, and once more on Ubuntu
without PyTorch/DeepXDE to keep the skip path working. The test session pins PyTorch's default
device to the CPU so results do not depend on the runner's GPU.

### Coverage

```bash
python -m pytest --cov=battery_pinn -q
```

Measured on 2026-10-31 with PyTorch and DeepXDE installed (190 tests, statement coverage, no
branch coverage): **92 % overall** (1145 statements, 95 missed). The seven core modules
(`config`, `data`, `etl`, `nasa_mat`, `interpolation`, `ocv`, `physics`) are at 96-100 %; the
lowest are `cli.py` (69 %) and `inference.py` (78 %), whose remaining lines are mostly the
training and evaluation entry points that the test suite deliberately does not run in full.
Without PyTorch the torch-dependent tests are skipped and the figure is lower.

## Layout

```
src/battery_pinn/   config, nasa_mat (.mat -> CSV), etl (raw -> train/held_out), data, synthetic,
                    interpolation, ocv, physics, training, inference, identifiability, checkpoint,
                    device, cli
scripts/            run_baseline.sh, run_validation.sh, make_results_table.py (-> docs/RESULTS.md)
test/               pytest suite
docs/               CHANGELOG.md, RESULTS.md / RESULTS.json, baseline_protocol.md, baseline_notes.md
data/               placeholder; raw, processed and generated data are not committed
```

The original Databricks notebook and its stage files were removed once the local pipeline had
been validated against them (`docs/RESULTS.md`, CHANGELOG items 21-27); they remain in the git
history. Nothing in `src/` imports Spark or MLflow, and the test suite enforces that.

## Data layout

```
data/nasa_mat/B00xx.mat      NASA's MATLAB files (input to prepare-data or convert-mat)
data/raw/metadata.csv        the per-cycle CSV layout (alternative input)
data/raw/data/<filename>
data/processed/train.csv     written by prepare-data (plus held_out.csv, the silver/gold tables
                             and raw_csv/ when converted from .mat)
data/synthetic/              written by generate-data
```

All folders are arguments (`--raw`, `--out`, `--mat-dir`); the defaults are `DEFAULT_*` in the
config module.

## Known limitations

- Held-out generalisation is uneven across batteries (see `docs/CHANGELOG.md`, items 17-20).
- The abuse-test verdict can vary between training seeds; the ensemble step measures this. The
  SEI-stability component depends on `A_sei`, which the data do not identify (see above), so
  that part of the verdict is not trustworthy; the core-temperature component is.
- Models only normal-cycling thermal behaviour; no internal-short, overcharge or mechanical-damage mechanisms.
- Predictions outside the trained ambient range (0-45 C) are extrapolation and are excluded from the verdict.
- The local demo shows that the pipeline executes, not that the model fits well. In the longest
  test run (6000 Adam iterations, one seed, synthetic data) the temperature fit stayed loose
  (train MAE about 7 C, held-out about 3 C), with the SEI residual loss still dominating.
- The CHANGELOG items 17-20 results came from a training set the old capacity floor had
  silently reduced to 19 batteries (and 5 held-out) and are not directly comparable with
  `docs/RESULTS.md`, whose converged run (`dynamic-subset`) uses the corrected 24/6 tables
  with `--anchor-subset 2000` rather than the notebook's full-anchor protocol.
- Cycles recorded before a battery's first impedance measurement have no `Re` and are dropped
  when a table is loaded (e.g. B0005's first 19 cycles); the loader report shows this per
  battery. Carrying the next measurement backwards would be a modelling decision, not made here.
- SoH is relative to each battery's single best cycle, so an outlier best cycle (B0036:
  2.44 Ah from a 2 Ah cell) compresses that battery's SoH scale.
- The representative-profile interpolation treats pooled normalised-time breakpoints that are
  closer than `MIN_RELATIVE_SPACING` of the range (or a few float32 ulps) as one point: inside
  such a gap it steps between the two values, and the edge slopes use the nearest well-separated
  breakpoint. Near-duplicate times from different cycles therefore no longer blow up the
  extrapolation at `t=0` / `t=1`, but the step means the profile is not smooth there.

## Roadmap

Planned work is in [ROADMAP.md](ROADMAP.md): code quality, an independent physics check,
data tooling, a stable API and command line, and packaging and documentation.

## Licence

This project is released under the [MIT License](LICENSE).

## Third-party licences

Checked in October 2026 against the metadata shipped with each package (PyPI releases of the
versions shown) and against NASA's data page. Re-check when you change versions.

| Dependency | Version checked | Licence |
|---|---|---|
| PyTorch | 2.14.1 | Apache-2.0, BSD-2/3-Clause, MIT, BSL-1.0 (combined; see its `LICENSE`) |
| DeepXDE | 1.15.0 | LGPL-2.1 (fine to import as a library; modified copies of DeepXDE itself must be shared under LGPL) |
| NumPy | 2.5.3 | BSD-3-Clause (plus bundled 0BSD, MIT, Zlib, CC0 components) |
| pandas | 3.0.5 | BSD-3-Clause |
| SciPy | 1.18.1 | BSD |
| Matplotlib | 3.11.2 | Matplotlib licence (PSF-based) |
| pytest | 9.1.1 | MIT |


## Data

The model uses NASA's Li-ion battery data (Data Set 5, "Batteries") from the NASA Prognostics
Center of Excellence data repository. NASA's page names no licence; it states that users employ the
data at their own risk and asks that publications acknowledge the repository and the data donors.
The data is not redistributed here.

> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository,
> NASA Ames Research Center, Moffett Field, CA.
> https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
