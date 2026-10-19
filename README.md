# Battery Digital Twin (Physics-Informed Neural Network)

A digital twin of a lithium-ion cell, built as a physics-informed neural network (PINN). It predicts how a cell's internal temperature and the chemical stability of its solid electrolyte interphase (SEI) layer evolve during charge and discharge, and uses that to run a simulated thermal-abuse ("point of no return") test.

The model is trained on the public NASA battery degradation dataset with [DeepXDE](https://github.com/lululxvi/deepxde) on PyTorch. The pipeline currently runs as a Databricks notebook.

## What it does

- **Learns battery physics, not just curves.** Coupled thermal and SEI-degradation partial differential equations are enforced as loss terms, alongside the measured cell temperature.
- **Estimates a hidden parameter.** The Arrhenius pre-exponential factor of SEI growth, `A_sei`, is a learnable value (trained as `log10(A_sei)`).
- **Takes the operating condition as input.** The network input is `[space, SoH, Re, Ambient, time]`: position through the cell, state of health, internal resistance, ambient temperature and time.
- **Simulates abuse.** An oven sweep raises the ambient temperature and reports whether the cell stays within a data-driven stability limit.
- **Checks generalisation.** Six batteries are held out of training and used only for evaluation.

## Pipeline

The notebook is organised in nine sections:

1. Setup
2. Data engineering (Bronze, Silver and Gold Delta tables). This is a one-off step, gated by `RUN_MEDALLION_ETL`.
3. Feature engineering: state of charge, state of health, and each cycle downsampled to 50 points. The 30 target batteries are split into 24 for training and 6 held out.
4. Physics definition: the PDE system, an empirical open-circuit-voltage curve and GPU-native interpolation.
5. Geometry, boundary and initial conditions.
6. Model architecture and training: Adam followed by L-BFGS, tracked with MLflow.
7. Digital twin restoration and the abuse test.
8. Held-out battery validation.
9. Ensemble abuse-test verdict stability across several seeds.

## Repository layout

```
notebooks/
  Battery_PINN_Digital_Twin_Revised_v10.ipynb   current notebook
  archive/                                      earlier revisions (v2 to v9) and the first training notebook
LICENSE
```

The header of the v10 notebook lists all twenty fixes made since v2.

## Getting started

The notebook is written for Databricks (Delta tables, MLflow and `dbutils`).

1. Obtain the NASA battery dataset from its provider and upload it to your workspace. The data is not included in this repository.
2. Import `notebooks/Battery_PINN_Digital_Twin_Revised_v10.ipynb` into Databricks.
3. On the first run, set `RUN_MEDALLION_ETL = True` in section 2 to build the tables. Set it back to `False` afterwards.
4. Run sections 1 to 6 to train.
5. In section 7, set `run_id` to the MLflow run that trained your model, then run sections 7 to 9.

Dependencies are installed by the first cell: `torch`, `torchvision`, `torchaudio`, `deepxde` and `scipy`.

## Results and limitations

- On the six held-out batteries the model shows a real generalisation gap compared with the batteries it trained on.
- The abuse-test verdict is not yet stable. Removing six batteries from training moved the maximum in-domain internal temperature from 82.4 °C to 53.8 °C and flipped the verdict from DANGER to SAFE. Do not use a single-run verdict for any safety decision.
- The model's `Ambient` input was only trained on 0 to 45 °C, so the oven sweep up to 150 °C is extrapolation.
- Core temperature has not been validated against measured core temperatures.
- `A_sei` is not expected to be well constrained by surface temperature alone.

## Planned work

- A portable local version that runs without Databricks, so results can be reproduced from the public data.
- A surface heat-loss boundary condition, and thermal constants (density, heat capacity, conductivity) as learnable parameters.
- A 10-seed ensemble verdict, with the improvements judged on the worst-performing batteries without degrading the others.

## Licence

Released under the [MIT Licence](LICENSE).
