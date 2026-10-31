"""Conversion of NASA's .mat files into the raw CSV layout the pipeline reads."""

import numpy as np
import pandas as pd
import pytest

from battery_pinn import etl
from battery_pinn.nasa_mat import _scalar, convert_mat_dir, read_cycles
from conftest import write_fake_mat


@pytest.fixture(scope="module")
def mat_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("mat")
    write_fake_mat(root / "B0005.mat", "B0005")
    write_fake_mat(root / "B0056.mat", "B0056", ambient=4.0)
    return root


def test_read_cycles_yields_type_ambient_time_and_fields(mat_dir):
    cycles = list(read_cycles(str(mat_dir / "B0005.mat")))
    assert [c[0] for c in cycles] == ["charge", "impedance", "discharge", "discharge"]
    ctype, ambient, when, data = cycles[2]
    assert ambient == 24.0
    assert when.tolist() == [2008, 4, 2, 15, 25, 41.0]
    assert data["Capacity"] == 1.85 and len(data["Time"]) == 120
    assert cycles[1][3]["Re"] == 0.05 and cycles[1][3]["Rct"] == 0.1


def test_scalar_handles_empty_and_complex_values():
    assert np.isnan(_scalar(np.zeros((1, 0))))
    assert _scalar(np.array([[1.5]])) == 1.5
    assert _scalar(np.array([[0.05 + 0j]])) == 0.05  # real-valued complex -> float
    assert _scalar(np.array([[0.05 + 0.01j]])) == "(0.05+0.01j)"  # truly complex -> text (ETL -> NaN)


def test_convert_writes_metadata_and_cycle_files(mat_dir, tmp_path):
    out = tmp_path / "raw"
    meta = convert_mat_dir(str(mat_dir), str(out), log=lambda *_: None)
    assert len(meta) == 8 and meta["uid"].tolist() == list(range(1, 9))
    assert meta["filename"].tolist() == [f"{i:05d}.csv" for i in range(1, 9)]
    assert meta["test_id"].tolist() == [0, 1, 2, 3] * 2
    assert meta.loc[meta["type"] == "discharge", "Capacity"].tolist() == [1.85, 1.80] * 2
    assert meta.loc[meta["type"] == "impedance", "Re"].tolist() == [0.05, 0.05]
    assert meta.loc[meta["type"] == "charge", ["Capacity", "Re", "Rct"]].isna().all().all()
    for filename in meta["filename"]:
        assert (out / "data" / filename).is_file()
    impedance = pd.read_csv(out / "data" / "00002.csv")
    assert len(impedance) == 5 and impedance["Rectified_Impedance"].isna().sum() == 2  # padded


def test_convert_output_parses_through_the_etl(mat_dir, tmp_path):
    out = tmp_path / "raw"
    convert_mat_dir(str(mat_dir), str(out), log=lambda *_: None)
    bronze = etl.load_raw_metadata(str(out))
    discharge = etl.build_silver_discharge(bronze)
    assert discharge["start_datetime"].notna().all()
    assert discharge[discharge["battery_id"] == "B0005"]["cycle_number"].tolist() == [1, 2]
    impedance = etl.build_silver_impedance(bronze)
    assert impedance["start_datetime"].iloc[0] == pd.Timestamp("2008-04-02 13:00:17.234")
    gold = etl.build_gold(discharge, impedance)
    assert (gold["Re"] == np.float32(0.05)).all()


def test_convert_can_restrict_batteries_and_reports_missing_dir(mat_dir, tmp_path):
    meta = convert_mat_dir(str(mat_dir), str(tmp_path / "one"), battery_ids=["B0056"], log=lambda *_: None)
    assert set(meta["battery_id"]) == {"B0056"}
    with pytest.raises(FileNotFoundError, match="No B\\*.mat files"):
        convert_mat_dir(str(tmp_path / "empty"), str(tmp_path / "out"), log=lambda *_: None)
