"""Property-based tests (Hypothesis) for the data readers and the schema check."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from battery_pinn import config, etl
from battery_pinn.nasa_mat import read_cycles
from battery_pinn.validation import DataValidationError, check_processed_table

# tmp_path is reused across examples, so every example overwrites the same file.
FILES = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


def _cycle_frame(time):
    n = len(time)
    return pd.DataFrame(
        {
            "Time": time,
            "Voltage_measured": np.full(n, 3.7),
            "Current_measured": np.full(n, -1.0),
            "Temperature_measured": np.full(n, 25.0),
        }
    )


@FILES
@given(steps=st.lists(st.floats(min_value=0, max_value=50, allow_nan=False), min_size=1, max_size=30))
def test_sorted_time_is_accepted_unchanged(tmp_path, steps):
    time = np.cumsum(steps)
    path = tmp_path / "cycle.csv"
    _cycle_frame(time).to_csv(path, index=False)
    out = etl.load_cycle_file(str(path))
    assert len(out) == len(time)
    assert list(out.columns) == list(config.RAW_CYCLE_COLUMNS)
    np.testing.assert_allclose(out["Time"].to_numpy(), time)


@FILES
@given(
    steps=st.lists(st.floats(min_value=0, max_value=50, allow_nan=False), min_size=1, max_size=20),
    drop=st.floats(min_value=0.5, max_value=100),
)
def test_a_backwards_step_names_its_row(tmp_path, steps, drop):
    time = list(np.cumsum(steps))
    time.append(time[-1] - drop)  # last row goes backwards
    path = tmp_path / "cycle.csv"
    _cycle_frame(time).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match=rf"out of order at row {len(time) - 1}\b"):
        etl.load_cycle_file(str(path))


@FILES
@given(dropped=st.sets(st.sampled_from(config.RAW_CYCLE_COLUMNS), min_size=1, max_size=3))
def test_any_missing_column_is_named(tmp_path, dropped):
    path = tmp_path / "cycle.csv"
    _cycle_frame([0.0, 1.0]).drop(columns=sorted(dropped)).to_csv(path, index=False)
    with pytest.raises(DataValidationError) as info:
        etl.load_cycle_file(str(path))
    assert all(name in str(info.value) for name in dropped)


@FILES
@given(
    values=st.lists(st.one_of(finite, st.just(float("nan")), st.just(float("inf"))), min_size=1, max_size=20),
)
def test_missing_and_infinite_values_pass_the_schema(values):
    df = pd.DataFrame({"a": values, "b": values})
    check_processed_table(df, "t.csv", ["a", "b"])


@FILES
@given(
    values=st.lists(finite, min_size=1, max_size=20),
    where=st.data(),
    junk=st.text(alphabet="xyz", min_size=1, max_size=5),
)
def test_a_non_numeric_cell_is_located_by_csv_line(values, where, junk):
    row = where.draw(st.integers(0, len(values) - 1))
    cells = [str(v) for v in values]
    cells[row] = junk
    df = pd.DataFrame({"a": cells, "b": values})
    with pytest.raises(DataValidationError, match=rf"t\.csv.*column 'a'.*row {row + 2}\b"):
        check_processed_table(df, "t.csv", ["a", "b"])


@FILES
@given(blob=st.binary(max_size=300))
def test_random_bytes_never_escape_as_other_errors(tmp_path, blob):
    path = tmp_path / "B0005.mat"
    path.write_bytes(blob)
    try:
        list(read_cycles(str(path)))
    except DataValidationError:
        pass
