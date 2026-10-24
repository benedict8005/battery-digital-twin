"""Consistency of the dataset definition in ``battery_pinn.config`` (the single source of truth)."""

import numpy as np
import pytest

from battery_pinn import config


def test_no_duplicate_target_batteries():
    assert len(config.TARGET_BATTERIES) == len(set(config.TARGET_BATTERIES)) == 30


def test_held_out_is_a_subset_of_targets_and_train_is_the_rest():
    assert len(config.HELD_OUT_BATTERIES) == 6
    assert set(config.HELD_OUT_BATTERIES) <= set(config.TARGET_BATTERIES)
    assert len(config.TRAIN_BATTERIES) == 24
    assert set(config.TRAIN_BATTERIES).isdisjoint(config.HELD_OUT_BATTERIES)
    assert sorted(config.TRAIN_BATTERIES + config.HELD_OUT_BATTERIES) == sorted(config.TARGET_BATTERIES)


def test_cutoff_map_covers_exactly_the_targets():
    assert set(config.CUTOFF_VOLTAGE_MAP) == set(config.TARGET_BATTERIES)
    assert set(config.CUTOFF_VOLTAGE_MAP.values()) == {2.0, 2.2, 2.5, 2.7}
    assert config.DEFAULT_CUTOFF_VOLTAGE in config.CUTOFF_VOLTAGE_MAP.values()


def test_excluded_batteries_are_not_targets():
    assert set(config.EXCLUDED_BATTERIES) == {"B0049", "B0050", "B0051", "B0052"}
    assert not set(config.TARGET_BATTERIES) & set(config.EXCLUDED_BATTERIES)


def test_etl_thresholds():
    assert config.MIN_VALID_CAPACITY_AH == 1.4
    assert config.MIN_SAMPLES_PER_CYCLE == config.BINS_PER_CYCLE == 50


def test_processed_schema_contains_everything_training_needs():
    assert set(config.REQUIRED_COLUMNS) <= set(config.PROCESSED_COLUMNS)
    assert len(config.PROCESSED_COLUMNS) == len(set(config.PROCESSED_COLUMNS))


def test_table_and_path_helpers(tmp_path):
    assert config.table_path(str(tmp_path), config.TRAIN_TABLE).endswith("train.csv")
    assert config.table_path(str(tmp_path), config.HELD_OUT_TABLE).endswith("held_out.csv")
    assert config.table_path(str(tmp_path), config.GOLD_TABLE).endswith("gold_battery_pinn_database.csv")
    assert config.metadata_path("raw").endswith("metadata.csv")
    assert config.cycle_file_path("raw", "00001.csv").replace("\\", "/").endswith("raw/data/00001.csv")
    # find_table falls back to the pre-rename file name, and otherwise returns the new path.
    assert config.find_table(str(tmp_path), config.TRAIN_TABLE) == str(tmp_path / "train.csv")
    (tmp_path / "final_pinn_training_data.csv").write_text("x")
    assert config.find_table(str(tmp_path), config.TRAIN_TABLE) == str(tmp_path / "final_pinn_training_data.csv")
    (tmp_path / "train.csv").write_text("x")
    assert config.find_table(str(tmp_path), config.TRAIN_TABLE) == str(tmp_path / "train.csv")


def test_is_mat_folder(tmp_path):
    assert not config.is_mat_folder(str(tmp_path))
    (tmp_path / "B0005.mat").write_bytes(b"")
    assert config.is_mat_folder(str(tmp_path))
    (tmp_path / "metadata.csv").write_text("x")
    assert not config.is_mat_folder(str(tmp_path))
    assert not config.is_mat_folder(str(tmp_path / "missing"))


def test_train_defaults_are_the_validated_fast_protocol():
    assert config.TRAIN_DEFAULTS == {
        "iterations": 2000,
        "lbfgs_iterations": 200,
        "num_domain": 1000,
        "num_boundary": 200,
        "anchor_subset": 2000,
        "display_every": 100,
    }
    assert config.ENSEMBLE_SEEDS == list(range(10)) and config.ABUSE_ENSEMBLE_SIZE == 10


def test_config_imports_only_the_standard_library():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(config))
    modules = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    modules |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert modules <= {"os", "glob"}  # standard library only


def test_capacity_floor_is_min_of_nominal_and_relative():
    nominal, frac = config.MIN_VALID_CAPACITY_AH, config.MIN_CAPACITY_FRACTION_OF_MAX
    assert (nominal, frac) == (1.4, 0.7)
    assert config.capacity_floor(2.444) == nominal  # B0036: outlier best cycle, nominal wins
    assert config.capacity_floor(1.856) == pytest.approx(1.2992)  # B0005: 30 % fade of its own best
    assert config.capacity_floor(1.344) == pytest.approx(0.9408)  # B0056: never reached 1.4 Ah
    assert config.capacity_floor(float("nan")) == nominal  # unknown best cycle -> nominal
    assert config.capacity_floor(None) == nominal
    assert config.capacity_floor(1.0, fraction_of_max=None) == nominal  # relative rule disabled
    assert config.capacity_floor(2.0, min_capacity_ah=1.0, fraction_of_max=0.5) == 1.0
    # Never stricter than the absolute floor, whatever the battery's best cycle.
    assert all(config.capacity_floor(m) <= nominal for m in np.linspace(0.1, 3.0, 50))
