"""Single source of truth for the battery PINN.

Everything that used to be hardcoded across the original notebook lives here: physical
constants, network/loss settings, abuse-test thresholds, the battery lists and per-battery
cutoff voltages, the ETL thresholds, the table names, the default paths and the default
training protocol, so a value only ever has to change in one place.

This module deliberately imports nothing but the standard library: it must stay importable
without PyTorch, DeepXDE, Spark or MLflow.
"""

from __future__ import annotations

import os
from typing import Any

# ----------------------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------------------
R_GAS = 8.314  # gas constant, J/(mol*K)
EA_SEI = 1.35e5  # activation energy, J/mol
H_SEI = 2.57e5  # enthalpy of reaction, J/kg
RHO = 2000.0  # density, kg/m3
CP = 1000.0  # heat capacity, J/(kg*K)
K_COND = 1.0  # thermal conductivity, W/(m*K)

# Opt-in physics variants (off by default; the defaults reproduce the original PDE exactly).
# Newton cooling: a volumetric surface heat-loss sink  -NEWTON_COOLING_COEFF * (T - T_ambient)
# added to the heat sources, W/(m3*K).
NEWTON_COOLING_COEFF = 5.0

EXPONENT_CLAMP_MIN = -70
EXPONENT_CLAMP_MAX = 50

TEMP_SCALE_C = 50.0

REFERENCE_TEMP_K = 308.15

# ----------------------------------------------------------------------------------------
# Network and training
# ----------------------------------------------------------------------------------------
# Inputs: [space, SoH, Re, ambient, time]; outputs: [T_delta, c].
NETWORK_LAYERS = [5] + [64] * 4 + [2]
# Order: [pde_thermal, pde_sei, ic_T, ic_c, observe_T].
LOSS_WEIGHTS = [10, 10, 30, 30, 20]
LOSS_TERM_NAMES = ["pde_thermal", "pde_sei", "ic_T", "ic_c", "observe_T"]

# A_sei identifiability across seeds (log10 decades): seeds must end within this spread of each
# other, and must have moved at least this far from where they started, for A_sei to count as
# identified by the data.
IDENTIFIABLE_MAX_SPREAD_DECADES = 0.5
IDENTIFIABLE_MIN_MOVEMENT_DECADES = 0.1
# If every seed drifts this far in the same direction, A_sei is running away (towards zero,
# where the SEI residual vanishes for any data) rather than being calibrated.
IDENTIFIABLE_MAX_RUNAWAY_DECADES = 3.0

# Input domain: [space, SoH, Re, ambient_C]; time is [0, 1].
PARAM_BOUNDS_MIN = [0, 0.5, 0.0, 0.0]
PARAM_BOUNDS_MAX = [1, 1.2, 12.0, 45.0]

# Default ensemble: ten seeds, so the abuse-test verdict spread is reported over ten models.
ENSEMBLE_SEEDS = list(range(10))
ABUSE_ENSEMBLE_SIZE = 10

# Default ("fast") training protocol, validated against the full-anchor protocol in
# docs/RESULTS.md: it reproduces it within seed noise at ~1/30 of the cost.
TRAIN_DEFAULTS = {
    "iterations": 2000,
    "lbfgs_iterations": 200,
    "num_domain": 1000,
    "num_boundary": 200,
    "anchor_subset": 2000,
    "display_every": 100,
}

# ----------------------------------------------------------------------------------------
# Abuse-test thresholds
# ----------------------------------------------------------------------------------------
AMBIENT_TRAINED_MAX_C = PARAM_BOUNDS_MAX[3]
CORE_TEMP_DANGER_C = 80.0
CORE_TEMP_WARNING_C = 60.0

# ----------------------------------------------------------------------------------------
# Dataset definition (NASA PCoE battery data set, cleaned CSV release)
# ----------------------------------------------------------------------------------------
# Batteries dropped at the bronze -> silver step; they never reach the gold table.
EXCLUDED_BATTERIES = ["B0049", "B0050", "B0051", "B0052"]

# Per-battery discharge cutoff voltage (V). A cycle is truncated at the first sample whose
# measured voltage drops below this value.
CUTOFF_VOLTAGE_MAP = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5,
    "B0025": 2.0,
    "B0026": 2.2,
    "B0027": 2.5,
    "B0028": 2.7,
    "B0029": 2.0,
    "B0030": 2.2,
    "B0031": 2.5,
    "B0032": 2.7,
    "B0033": 2.0,
    "B0034": 2.2,
    "B0036": 2.7,
    "B0038": 2.2,
    "B0039": 2.5,
    "B0040": 2.7,
    "B0041": 2.0,
    "B0042": 2.2,
    "B0043": 2.5,
    "B0044": 2.7,
    "B0045": 2.0,
    "B0046": 2.2,
    "B0047": 2.5,
    "B0048": 2.7,
    "B0053": 2.0,
    "B0054": 2.2,
    "B0055": 2.5,
    "B0056": 2.7,
}
# Used for any battery that is not in CUTOFF_VOLTAGE_MAP.
DEFAULT_CUTOFF_VOLTAGE = 2.7

# The 30 batteries the model is built from, split by battery (never by cycle) into a
# 24-battery training set and a 6-battery held-out set.
TARGET_BATTERIES = [
    "B0005",
    "B0006",
    "B0007",
    "B0018",
    "B0025",
    "B0026",
    "B0027",
    "B0028",
    "B0029",
    "B0030",
    "B0031",
    "B0032",
    "B0033",
    "B0034",
    "B0036",
    "B0038",
    "B0039",
    "B0040",
    "B0041",
    "B0042",
    "B0043",
    "B0044",
    "B0045",
    "B0046",
    "B0047",
    "B0048",
    "B0053",
    "B0054",
    "B0055",
    "B0056",
]
HELD_OUT_BATTERIES = ["B0025", "B0030", "B0036", "B0042", "B0047", "B0056"]
TRAIN_BATTERIES = [b for b in TARGET_BATTERIES if b not in HELD_OUT_BATTERIES]

# Ambient test groups present in the dataset (degrees C); used by the synthetic generator.
AMBIENT_GROUPS_C = (4.0, 24.0, 43.0)

# ----------------------------------------------------------------------------------------
# ETL thresholds (feature engineering)
# ----------------------------------------------------------------------------------------
# A discharge cycle is skipped when its measured capacity is at or below the capacity floor.
# The floor is the lower of two expressions of NASA's end-of-life criterion (30 % fade):
#   * MIN_VALID_CAPACITY_AH: 30 % fade of the 2 Ah *nominal* capacity, the original absolute
#     floor. On its own it removed every cycle of the 4 C batteries (B0041, B0045, B0053-B0056,
#     whose best cycles deliver only 1.1-1.35 Ah at that temperature) and most of B0046-B0048.
#   * MIN_CAPACITY_FRACTION_OF_MAX x the battery's own best cycle: the same 30 % fade measured
#     against what that battery actually delivered. Set to None to restore the absolute floor.
# See capacity_floor() and docs/CHANGELOG.md item 22.
MIN_VALID_CAPACITY_AH = 1.4
MIN_CAPACITY_FRACTION_OF_MAX = 0.7
MIN_SAMPLES_PER_CYCLE = 50  # cycles with fewer samples after truncation are skipped
BINS_PER_CYCLE = 50  # each kept cycle is downsampled to this many points

# ----------------------------------------------------------------------------------------
# Column names
# ----------------------------------------------------------------------------------------
# Columns the raw metadata.csv must provide (matched case-insensitively when read).
RAW_METADATA_COLUMNS = [
    "battery_id",
    "type",
    "filename",
    "ambient_temperature",
    "start_time",
    "capacity",
    "Re",
    "Rct",
]
# Columns each raw per-cycle discharge CSV must provide.
RAW_CYCLE_COLUMNS = ["Time", "Voltage_measured", "Current_measured", "Temperature_measured"]
# Column order of the processed tables (final_pinn_training_data / held_out_pinn_validation_data).
PROCESSED_COLUMNS = [
    "bin",
    "Time",
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "SoC",
    "battery_id",
    "cycle_number",
    "SoH",
    "Re",
    "Rct",
    "ambient_temperature",
    "Time_normalized",
]
# Columns the training / inference code needs; rows missing any of them are dropped.
REQUIRED_COLUMNS = [
    "Time",
    "Time_normalized",
    "Current_measured",
    "Voltage_measured",
    "SoC",
    "SoH",
    "Re",
    "Temperature_measured",
    "ambient_temperature",
    "battery_id",
]

# ----------------------------------------------------------------------------------------
# Table names (medallion layers, kept from the original pipeline) and their local file names.
# ----------------------------------------------------------------------------------------
BRONZE_TABLE = "bronze_battery_metadata"
SILVER_DISCHARGE_TABLE = "silver_discharge_metadata"
SILVER_IMPEDANCE_TABLE = "silver_impedance_metadata"
GOLD_TABLE = "gold_battery_pinn_database"
TRAIN_TABLE = "final_pinn_training_data"
HELD_OUT_TABLE = "held_out_pinn_validation_data"
# The two tables the model consumes are written under short names; the others as <table>.csv.
TABLE_FILES = {TRAIN_TABLE: "train.csv", HELD_OUT_TABLE: "held_out.csv"}
TRAIN_FILE = TABLE_FILES[TRAIN_TABLE]
HELD_OUT_FILE = TABLE_FILES[HELD_OUT_TABLE]

# ----------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------
# Layout of the raw dataset folder: <raw_dir>/metadata.csv and <raw_dir>/data/<filename>.
# `prepare-data` also accepts a folder of NASA's B00xx.mat files and converts it first
# (into <out>/raw_csv/); `convert-mat` does that step on its own.
METADATA_FILENAME = "metadata.csv"
CYCLE_DATA_SUBDIR = "data"
MAT_CONVERSION_SUBDIR = "raw_csv"

# Local defaults (relative to the working directory; all git-ignored).
DEFAULT_MAT_DIR = os.path.join("data", "nasa_mat")
DEFAULT_RAW_DATA_DIR = os.path.join("data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
DEFAULT_SYNTHETIC_DIR = os.path.join("data", "synthetic")
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_TRAIN_CSV = os.path.join(DEFAULT_PROCESSED_DIR, TRAIN_FILE)
DEFAULT_HELD_OUT_CSV = os.path.join(DEFAULT_PROCESSED_DIR, HELD_OUT_FILE)


UNSET: Any = object()  # "use the module constant" sentinel for keyword overrides


def capacity_floor(
    max_capacity_ah: float | None,
    min_capacity_ah: Any = UNSET,
    fraction_of_max: Any = UNSET,
) -> float:
    """Capacity (Ah) at or below which a discharge cycle is skipped.

    ``min(MIN_VALID_CAPACITY_AH, MIN_CAPACITY_FRACTION_OF_MAX * max_capacity_ah)``: a cycle
    survives if it clears the nominal end-of-life capacity *or* has faded less than 30 % from
    the battery's own best cycle. Never stricter than the absolute floor alone. With the
    fraction disabled (``None``) or an unknown battery maximum, the absolute floor is used.
    The two keyword arguments override the module constants.
    """
    if min_capacity_ah is UNSET:
        min_capacity_ah = MIN_VALID_CAPACITY_AH
    if fraction_of_max is UNSET:
        fraction_of_max = MIN_CAPACITY_FRACTION_OF_MAX
    if fraction_of_max is None or max_capacity_ah is None or max_capacity_ah != max_capacity_ah:
        return float(min_capacity_ah)
    return float(min(min_capacity_ah, fraction_of_max * float(max_capacity_ah)))


def metadata_path(raw_dir: str) -> str:
    """Path of the raw metadata file inside a raw dataset folder."""
    return os.path.join(raw_dir, METADATA_FILENAME)


def cycle_file_path(raw_dir: str, filename: str) -> str:
    """Path of one raw per-cycle CSV inside a raw dataset folder."""
    return os.path.join(raw_dir, CYCLE_DATA_SUBDIR, filename)


def table_path(out_dir: str, table: str) -> str:
    """Local CSV path standing in for a named table (``train.csv`` / ``held_out.csv`` for the
    two model tables, ``<table>.csv`` otherwise)."""
    return os.path.join(out_dir, TABLE_FILES.get(table, f"{table}.csv"))


def find_table(data_dir: str, table: str) -> str:
    """``table_path`` if it exists, else the pre-rename ``<table>.csv`` if that exists, else
    the new path (so callers get a clear FileNotFoundError)."""
    new = table_path(data_dir, table)
    legacy = os.path.join(data_dir, f"{table}.csv")
    if not os.path.exists(new) and os.path.exists(legacy):
        return legacy
    return new


def is_mat_folder(path: str) -> bool:
    """True if ``path`` holds NASA ``B*.mat`` files and no ``metadata.csv``."""
    import glob

    return (
        os.path.isdir(path)
        and not os.path.exists(metadata_path(path))
        and bool(glob.glob(os.path.join(path, "B*.mat")))
    )
