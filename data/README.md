# data/

Battery data is **not** committed to this repository (see `.gitignore`); only this file and
`.gitkeep` are tracked.

## NASA's .mat files

NASA distributes the battery data set as zip files of MATLAB files (`B0005.mat`, ...). Unpack
them into one folder (default `data/nasa_mat/`) and convert them with

    python -m battery_pinn convert-mat --mat-dir data/nasa_mat --out data/raw

which writes the raw layout below. Only the `.mat` files are needed; the READMEs are optional.

## Raw dataset

The pipeline expects the per-cycle CSV layout (the "cleaned" form of the dataset) in
`DEFAULT_RAW_DATA_DIR` from `src/battery_pinn/config.py`; any folder can be passed with `--raw`:

    data/raw/metadata.csv        one row per test: type, start_time, ambient_temperature,
                                 battery_id, filename, Capacity, Re, Rct (plus test_id, uid)
    data/raw/data/<filename>     one CSV per cycle: Voltage_measured, Current_measured,
                                 Temperature_measured, Time (plus Current_load, Voltage_load)

Download the dataset from the NASA Prognostics Center of Excellence data repository and unpack
it there. The dataset has no formal licence and is used at your own risk; NASA asks that
publications acknowledge the repository and the data donors. Cite:

> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository,
> NASA Ames Research Center, Moffett Field, CA.

## Processed tables

`python -m battery_pinn prepare-data --raw data/nasa_mat --out data/processed` (a `.mat` folder
or the CSV layout) writes:

    data/processed/train.csv                           24 training batteries
    data/processed/held_out.csv                        6 held-out batteries
    data/processed/raw_csv/                            the converted per-cycle CSVs (from .mat)
    data/processed/gold_battery_pinn_database.csv      per-cycle metadata with Re/Rct attached
    data/processed/silver_discharge_metadata.csv       discharge rows with parsed timestamps
    data/processed/silver_impedance_metadata.csv       impedance rows with parsed timestamps

`train.csv` / `held_out.csv` are the defaults of `train`, `evaluate` and `abuse-test`
(`--data` / `--held-out-data`). The battery split, cutoff voltages and thresholds are defined in
`src/battery_pinn/config.py`. `prepare-data` reports kept/dropped cycles for every designated
battery and refuses to write anything if a held-out battery would have no data; see the README
section on the capacity floor for why this used to happen to B0056.

## Synthetic data

`python -m battery_pinn generate-data` writes `data/synthetic/train.csv` and `held_out.csv`
with the same columns as the processed tables, so the rest of the pipeline can run without the
NASA files. It is not battery data and says nothing about real cells.
