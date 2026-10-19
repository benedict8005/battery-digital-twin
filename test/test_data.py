import numpy as np
import pandas as pd
import pytest

from battery_pinn import config, synthetic
from battery_pinn.data import (
    BatteryCoverageError, battery_report, build_points_and_targets, coverage_problems,
    format_battery_report, load_dataset, representative_profile,
)


def test_synthetic_split_never_shares_batteries():
    train, held_out = synthetic.generate_train_and_held_out(n_batteries=6, n_held_out=2, cycles_per_battery=4)
    assert set(train["battery_id"]).isdisjoint(held_out["battery_id"])
    assert held_out["battery_id"].nunique() == 2


def test_synthetic_matches_required_schema(tmp_path):
    train, _ = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=1, cycles_per_battery=3)
    path = tmp_path / "t.csv"
    train.to_csv(path, index=False)
    assert len(load_dataset(path)) == len(train)


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Time,SoC\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_target_is_celsius_difference_scaled_by_50():
    df, _ = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=1, cycles_per_battery=3)
    points, targets = build_points_and_targets(df)
    expected = (df["Temperature_measured"].values - df["ambient_temperature"].values) / 50.0
    np.testing.assert_allclose(targets, expected, atol=1e-6)
    assert points.shape == (len(df), 5)
    assert np.all(points[:, 0] == 1.0)  # surface


def test_representative_profile_is_sorted_and_unique():
    df, _ = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=1, cycles_per_battery=3)
    prof = representative_profile(df)
    assert prof["Time_normalized"].is_monotonic_increasing
    assert prof["Time_normalized"].is_unique


# ----------------------------------------------------------------------------------------
# Per-battery coverage report and the held-out hard error
# ----------------------------------------------------------------------------------------
def _write(tmp_path, df, name="held_out.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_load_dataset_reports_kept_and_dropped_cycles_per_battery(tmp_path):
    df, _ = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=0, cycles_per_battery=3)
    # S0002 loses one whole cycle (no Re) and S0003 loses a single row.
    df.loc[(df["battery_id"] == "S0002") & (df["cycle_number"] == 2), "Re"] = np.nan
    df.loc[df[df["battery_id"] == "S0003"].index[0], "Temperature_measured"] = np.inf
    messages = []
    clean = load_dataset(_write(tmp_path, df), log=messages.append)

    assert len(clean) == len(df) - 50 - 1
    text = "\n".join(messages)
    assert "Per-battery coverage of" in text
    assert "S0001: 3/3 cycles kept, 150/150 rows" in text
    assert "S0002: 2/3 cycles kept, 100/150 rows  (1 cycles dropped: missing/non-finite values)" in text
    assert "S0003: 3/3 cycles kept, 149/150 rows" in text


def test_held_out_missing_battery_raises(tmp_path):
    """A designated held-out battery that is absent from the file is a hard error."""
    df, _ = synthetic.generate_train_and_held_out(n_batteries=2, n_held_out=0, cycles_per_battery=2)
    path = _write(tmp_path, df)
    with pytest.raises(BatteryCoverageError, match=r"S0003: not present in the file") as info:
        load_dataset(path, expected_batteries=["S0001", "S0002", "S0003"], log=None)
    assert "S0001" not in str(info.value)  # only the problem batteries are listed
    # The same file is fine when its batteries are the designated ones.
    assert len(load_dataset(path, expected_batteries=["S0001", "S0002"], log=None)) == len(df)


def test_held_out_missing_battery_uses_config_list_for_real_data(tmp_path):
    """The real failure mode: a held-out table written without B0056."""
    df, _ = synthetic.generate_train_and_held_out(n_batteries=5, n_held_out=0, cycles_per_battery=1)
    present = [b for b in config.HELD_OUT_BATTERIES if b != "B0056"]
    df["battery_id"] = df["battery_id"].map(dict(zip(sorted(df["battery_id"].unique()), present)))
    messages = []
    with pytest.raises(BatteryCoverageError, match="B0056: not present in the file"):
        load_dataset(_write(tmp_path, df), expected_batteries=config.HELD_OUT_BATTERIES, log=messages.append)
    assert "B0056 [HELD-OUT]: MISSING from file" in "\n".join(messages)


def test_held_out_missing_battery_is_a_warning_when_not_required(tmp_path):
    df, _ = synthetic.generate_train_and_held_out(n_batteries=2, n_held_out=0, cycles_per_battery=2)
    messages = []
    clean = load_dataset(_write(tmp_path, df), expected_batteries=["S0001", "S0009"], required=False,
                         log=messages.append)
    assert len(clean) == len(df)
    assert any(m.startswith("WARNING:") and "S0009: not present" in m for m in messages)


def test_held_out_battery_with_zero_surviving_cycles_raises(tmp_path):
    df, _ = synthetic.generate_train_and_held_out(n_batteries=2, n_held_out=0, cycles_per_battery=2)
    df.loc[df["battery_id"] == "S0002", "Re"] = np.nan   # every S0002 row is dropped by cleaning
    path = _write(tmp_path, df)
    with pytest.raises(BatteryCoverageError, match=r"S0002: 0 of 2 cycles survived cleaning"):
        load_dataset(path, expected_batteries=["S0001", "S0002"], log=None)
    # Without an expected list every battery in the file is expected, so it still fails.
    with pytest.raises(BatteryCoverageError, match="S0002"):
        load_dataset(path, log=None)


def test_battery_report_covers_expected_batteries_and_extras():
    raw = pd.DataFrame({"battery_id": ["A"] * 4 + ["B"] * 2 + ["X"] * 2,
                        "cycle_number": [1, 1, 2, 2, 1, 1, 1, 2], "Re": [1, 1, np.nan, np.nan, 1, 1, 1, 1]})
    clean = raw.dropna()
    report = battery_report(raw, clean, expected_batteries=["A", "B", "C"])
    assert list(report) == ["A", "B", "C", "X"]
    assert report["A"] == {"present": True, "expected": True, "cycles_in_file": 2, "cycles_kept": 1,
                           "cycles_dropped": 1, "rows_in_file": 4, "rows_kept": 2}
    assert report["C"]["present"] is False and report["X"]["expected"] is False
    assert coverage_problems(report) == ["C: not present in the file"]
    text = format_battery_report(report, held_out_batteries=["B"])
    assert "B [HELD-OUT]: 1/1 cycles kept, 2/2 rows" in text
    assert "C: MISSING from file" in text
    assert "X: 2/2 cycles kept, 2/2 rows  (not a designated battery)" in text
