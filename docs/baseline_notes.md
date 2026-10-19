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

