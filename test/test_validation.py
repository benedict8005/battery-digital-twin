"""Malformed-input cases for the .mat reader: each must raise a readable error."""

import numpy as np
import pytest
from scipy.io import savemat

from battery_pinn.nasa_mat import convert_mat_dir, read_cycles
from battery_pinn.validation import DataValidationError
from conftest import write_fake_mat


def _struct(**fields):
    return np.array([[tuple(fields.values())]], dtype=[(k, "O") for k in fields])


def _write(path, battery_id, cycles):
    """Write a .mat holding ``cycles`` = list of (type, time, ambient, data_struct)."""
    names = ["type", "ambient_temperature", "time", "data"]
    arr = np.empty((1, len(cycles)), dtype=[(n, "O") for n in names])
    for i, (ctype, time, ambient, data) in enumerate(cycles):
        arr[0, i] = (ctype, ambient, time, data)
    savemat(str(path), {battery_id: _struct(cycle=arr)})


GOOD_TIME = np.array([2008, 4, 2, 12, 0, 0.0])
GOOD_DATA = _struct(Voltage_measured=np.ones(3), Time=np.arange(3.0))


def test_valid_file_still_reads(tmp_path):
    write_fake_mat(tmp_path / "B0005.mat", "B0005")
    assert len(list(read_cycles(str(tmp_path / "B0005.mat")))) == 4


def test_truncated_file_names_the_file(tmp_path):
    path = tmp_path / "B0005.mat"
    write_fake_mat(path, "B0005")
    path.write_bytes(path.read_bytes()[:200])
    with pytest.raises(DataValidationError, match=r"B0005\.mat.*cannot read"):
        list(read_cycles(str(path)))


def test_not_a_mat_file(tmp_path):
    path = tmp_path / "B0005.mat"
    path.write_text("this is not matlab")
    with pytest.raises(DataValidationError, match="cannot read"):
        list(read_cycles(str(path)))


def test_wrong_variable_name(tmp_path):
    write_fake_mat(tmp_path / "B0005.mat", "B0006")
    with pytest.raises(DataValidationError, match="'B0005'"):
        list(read_cycles(str(tmp_path / "B0005.mat")))


def test_missing_cycle_struct(tmp_path):
    savemat(str(tmp_path / "B0005.mat"), {"B0005": _struct(other=np.ones(2))})
    with pytest.raises(DataValidationError, match="no 'cycle'"):
        list(read_cycles(str(tmp_path / "B0005.mat")))


def test_missing_cycle_field_names_cycle_and_field(tmp_path):
    arr = np.empty((1, 1), dtype=[("type", "O"), ("time", "O"), ("data", "O")])
    arr[0, 0] = ("discharge", GOOD_TIME, GOOD_DATA)
    savemat(str(tmp_path / "B0005.mat"), {"B0005": _struct(cycle=arr)})
    with pytest.raises(DataValidationError, match=r"cycle 0: missing field\(s\) ambient_temperature"):
        list(read_cycles(str(tmp_path / "B0005.mat")))


def test_empty_data_struct(tmp_path):
    _write(tmp_path / "B0005.mat", "B0005", [("discharge", GOOD_TIME, 24.0, np.zeros((0, 0)))])
    with pytest.raises(DataValidationError, match="cycle 0: empty 'data'"):
        list(read_cycles(str(tmp_path / "B0005.mat")))


def test_bad_time_vector_names_cycle(tmp_path):
    _write(
        tmp_path / "B0005.mat",
        "B0005",
        [("discharge", GOOD_TIME, 24.0, GOOD_DATA), ("discharge", np.array([2008, 4.0]), 24.0, GOOD_DATA)],
    )
    with pytest.raises(DataValidationError, match="cycle 1: 'time' must be a 6-element"):
        list(read_cycles(str(tmp_path / "B0005.mat")))


def test_empty_ambient_temperature(tmp_path):
    _write(tmp_path / "B0005.mat", "B0005", [("discharge", GOOD_TIME, np.zeros((0, 0)), GOOD_DATA)])
    with pytest.raises(DataValidationError, match="cycle 0"):
        list(read_cycles(str(tmp_path / "B0005.mat")))


def test_convert_surfaces_the_error(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "B0005.mat").write_text("junk")
    with pytest.raises(DataValidationError, match=r"B0005\.mat"):
        convert_mat_dir(str(tmp_path / "in"), str(tmp_path / "out"), log=lambda *_: None)


def test_validation_error_is_a_value_error():
    assert issubclass(DataValidationError, ValueError)


# ----------------------------------------------------------------------------------------
# Per-cycle CSVs read by the ETL
# ----------------------------------------------------------------------------------------
from battery_pinn import etl  # noqa: E402


def test_cycle_csv_missing_columns_is_still_a_value_error(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("Time,Voltage_measured\n0,4.0\n")
    with pytest.raises(ValueError, match="missing required columns"):  # old contract
        etl.load_cycle_file(str(path))
    with pytest.raises(DataValidationError, match=r"c\.csv.*Current_measured"):
        etl.load_cycle_file(str(path))


def test_cycle_csv_zero_bytes_names_the_file(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("")
    with pytest.raises(DataValidationError, match=r"c\.csv.*empty"):
        etl.load_cycle_file(str(path))


def test_cycle_csv_header_only_is_rejected(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("Time,Voltage_measured,Current_measured,Temperature_measured\n")
    with pytest.raises(DataValidationError, match=r"c\.csv.*no data rows"):
        etl.load_cycle_file(str(path))


def test_cycle_csv_out_of_order_time_names_the_row(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(
        "Time,Voltage_measured,Current_measured,Temperature_measured\n0,4.0,-1,25\n10,3.9,-1,25\n5,3.8,-1,25\n"
    )
    with pytest.raises(DataValidationError, match=r"c\.csv.*'Time'.*row 2"):
        etl.load_cycle_file(str(path))


def test_cycle_csv_valid_file_loads(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("Time,Voltage_measured,Current_measured,Temperature_measured\n0,4.0,-1,25\n10,3.9,-1,25\n")
    assert len(etl.load_cycle_file(str(path))) == 2
