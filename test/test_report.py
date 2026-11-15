"""The data report against numbers worked out by hand on a tiny synthetic table."""

import numpy as np
import pandas as pd
import pytest

from battery_pinn import report
from battery_pinn.report import column_report, dropped_cycles, format_report, inspect_file, inspect_table
from battery_pinn.validation import DataValidationError


def make_table() -> pd.DataFrame:
    """B1: cycles 1 and 2 (3 rows each, all finite). B2: cycle 1 (2 rows, SoH missing in one),
    cycle 2 (2 rows, Re infinite in one). B3: cycle 1 (1 row, all finite)."""
    rows = []
    for bid, cycle, n in [("B1", 1, 3), ("B1", 2, 3), ("B2", 1, 2), ("B2", 2, 2), ("B3", 1, 1)]:
        for i in range(n):
            rows.append(
                {
                    "Time": float(i),
                    "Time_normalized": i / 10,
                    "Current_measured": -2.0,
                    "Voltage_measured": 3.5 + i,
                    "Temperature_measured": 25.0 + i,
                    "SoC": 1.0 - i / 10,
                    "SoH": 90.0,
                    "Re": 0.1,
                    "ambient_temperature": 24.0,
                    "battery_id": bid,
                    "cycle_number": cycle,
                }
            )
    df = pd.DataFrame(rows)
    df.loc[(df.battery_id == "B2") & (df.cycle_number == 1) & (df.Time == 1.0), "SoH"] = np.nan
    df.loc[(df.battery_id == "B2") & (df.cycle_number == 2) & (df.Time == 0.0), "Re"] = np.inf
    return df


def test_column_report_counts_and_ranges():
    cols = column_report(make_table())
    assert cols["Voltage_measured"] == {"rows": 11, "missing": 0, "min": 3.5, "max": 5.5}
    assert cols["SoH"] == {"rows": 11, "missing": 1, "min": 90.0, "max": 90.0}
    assert cols["Re"]["missing"] == 1 and cols["Re"]["max"] == 0.1
    assert "battery_id" not in cols


def test_column_report_all_missing_column_has_no_range():
    df = make_table()
    df["Rct"] = np.nan
    assert column_report(df)["Rct"] == {"rows": 11, "missing": 11, "min": None, "max": None}


def test_dropped_cycles_name_the_column():
    assert dropped_cycles(make_table()) == {"B2": {"SoH": [1], "Re": [2]}}


def test_inspect_table_matches_hand_counts():
    report = inspect_table(make_table())
    assert report["rows"] == 11
    b = report["batteries"]
    assert (b["B1"]["cycles_kept"], b["B1"]["rows_kept"]) == (2, 6)
    assert (b["B2"]["cycles_in_file"], b["B2"]["cycles_kept"], b["B2"]["cycles_dropped"]) == (2, 2, 0)
    assert (b["B2"]["rows_in_file"], b["B2"]["rows_kept"]) == (4, 2)
    assert (b["B3"]["cycles_kept"], b["B3"]["rows_kept"]) == (1, 1)


def test_format_report_mentions_dropped_cycles_and_batteries():
    text = format_report(inspect_table(make_table()), title="demo")
    assert text.startswith("--- demo: 11 rows ---")
    assert "B1: 2/2 cycles kept, 6/6 rows" in text
    assert "B2: cycles [1] (column 'SoH')" in text
    assert "SoH: 90 .. 90, 1 missing" in text


def test_format_report_without_drops():
    df = make_table().dropna()
    df = df[np.isfinite(df["Re"])]
    assert format_report(inspect_table(df)).endswith("Cycles dropped: none")


def test_inspect_file_checks_schema(tmp_path):
    good = tmp_path / "train.csv"
    make_table().to_csv(good, index=False)
    assert inspect_file(good)["rows"] == 11
    bad = tmp_path / "bad.csv"
    make_table().drop(columns=["SoH"]).to_csv(bad, index=False)
    with pytest.raises(DataValidationError, match="SoH"):
        inspect_file(bad)


def test_inspect_raw_reports_skip_reasons(raw_dataset):
    summary = report.inspect_raw(raw_dataset)
    # Counts worked out by hand from the synthetic raw dataset (see test_etl).
    assert summary["total_candidates"] == 19
    assert summary["total_kept"] == 13
    assert summary["skip_counts"] == {"file_not_found": 1, "too_short": 2, "low_capacity": 3}
    assert summary["per_battery"]["B0005"]["dropped"] == {"file_not_found": 1, "too_short": 1, "low_capacity": 1}


def test_inspect_raw_rejects_mat_folder(tmp_path):
    from conftest import write_fake_mat

    write_fake_mat(str(tmp_path / "B0005.mat"), "B0005")
    with pytest.raises(DataValidationError, match="convert-mat"):
        report.inspect_raw(tmp_path)


def test_inspect_data_raw_prints_skip_reasons(raw_dataset, capsys):
    from battery_pinn import cli

    cli.main(["inspect-data", "--raw", raw_dataset])
    out = capsys.readouterr().out
    assert "Skipped (too short):      2" in out
    assert "B0005: 2/5 cycles kept, dropped: 1 file_not_found, 1 too_short, 1 low_capacity" in out
