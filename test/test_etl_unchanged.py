"""Pin the ETL output on the shared synthetic raw dataset.

The data-validation work added checks around the readers; it must not change what the ETL
produces. These golden values were taken from the pipeline before those checks existed
(and the old and new pipelines were confirmed to write byte-identical CSVs), so a later
change that alters the processed tables fails here. Row counts, columns and column sums are
used rather than file hashes so the test does not depend on the platform's line endings.
"""

import pandas as pd
import pytest

from battery_pinn import etl

COLUMNS = [
    "bin", "Time", "Voltage_measured", "Current_measured", "Temperature_measured", "SoC",
    "battery_id", "cycle_number", "SoH", "Re", "Rct", "ambient_temperature", "Time_normalized",
]  # fmt: skip

EXPECTED = {
    "train": (
        {"B0005": 100, "B0006": 100},
        {
            "bin": 4900.0, "Time": 106827.272, "Voltage_measured": 740.218,
            "Current_measured": -400.008, "Temperature_measured": 7647.334, "SoC": 16720.631,
            "cycle_number": 450.0, "SoH": 19344.595, "Re": 10.5, "Rct": 21.0,
            "ambient_temperature": 6700.0, "Time_normalized": 101.029,
        },
    ),
    "held_out": (
        {"B0025": 100, "B0030": 50, "B0036": 50, "B0042": 50, "B0047": 100, "B0056": 100},
        {
            "bin": 11025.0, "Time": 196364.607, "Voltage_measured": 1649.81,
            "Current_measured": -899.973, "Temperature_measured": 7876.228, "SoC": 37651.922,
            "cycle_number": 600.0, "SoH": 43245.193, "Re": 61.5, "Rct": 123.0,
            "ambient_temperature": 5650.0, "Time_normalized": 226.838,
        },
    ),
}  # fmt: skip


@pytest.fixture(scope="module")
def processed(raw_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("etl_unchanged")
    etl.run_pipeline(str(raw_dataset), str(out), log=lambda _msg: None)
    return out


@pytest.mark.parametrize("table", sorted(EXPECTED))
def test_processed_table_matches_golden_values(processed, table):
    df = pd.read_csv(processed / f"{table}.csv")
    rows_per_battery, column_sums = EXPECTED[table]
    assert list(df.columns) == COLUMNS
    assert df["battery_id"].value_counts().sort_index().to_dict() == rows_per_battery
    for column, expected in column_sums.items():
        assert float(df[column].sum()) == pytest.approx(expected, abs=2e-3), column
