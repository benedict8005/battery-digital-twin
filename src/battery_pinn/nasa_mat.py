"""Convert NASA's MATLAB battery files into the cleaned CSV layout the pipeline reads.

NASA distributes each battery as ``B00xx.mat`` holding a ``cycle`` struct array (fields
``type``, ``ambient_temperature``, ``time``, ``data``). ``prepare-data`` expects the
"cleaned" layout instead -- ``metadata.csv`` with one row per cycle plus ``data/<uid>.csv``
with that cycle's measurements -- so this module produces exactly that:

* ``metadata.csv`` columns: ``type, start_time, ambient_temperature, battery_id, test_id,
  uid, filename, Capacity, Re, Rct``. ``start_time`` is the MATLAB date vector rendered as
  a printed numpy array (the form the ETL parses). ``Capacity`` is filled for discharge
  rows, ``Re``/``Rct`` for impedance rows; the handful of impedance fits that came out
  complex are written as complex literals, which the ETL treats as missing (as Spark's
  ``CAST(... AS FLOAT)`` did).
* ``data/<uid>.csv``: the ``data`` struct's fields as columns (discharge: ``Voltage_measured,
  Current_measured, Temperature_measured, Current_load, Voltage_load, Time``).
"""

import glob
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from battery_pinn import config

METADATA_COLUMNS = [
    "type",
    "start_time",
    "ambient_temperature",
    "battery_id",
    "test_id",
    "uid",
    "filename",
    "Capacity",
    "Re",
    "Rct",
]
# Fields that are per-cycle scalars rather than measurement vectors.
SCALAR_FIELDS = ("Capacity", "Re", "Rct")


def _scalar(value: Any) -> float | str:
    """A struct field holding one number (possibly complex), or NaN when it is empty."""
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return float("nan")
    v = arr[0]
    if np.iscomplexobj(v):
        # A genuinely complex fit is kept as text so the column stays numeric for the rest.
        return str(complex(v)) if v.imag != 0 else float(v.real)
    return float(v)


def read_cycles(
    mat_path: str,
) -> Iterator[tuple[str, float, npt.NDArray[np.float64], dict[str, Any]]]:
    """Yield ``(type, ambient_temperature, time_vector, data)`` for every cycle in a .mat file.

    ``data`` maps each measurement field to a 1-D array and each scalar field to a number.
    """
    from scipy.io import loadmat

    battery_id = os.path.splitext(os.path.basename(mat_path))[0]
    mat = loadmat(mat_path)
    if battery_id not in mat:
        raise ValueError(f"{mat_path}: expected a top-level variable named {battery_id!r}")
    cycles = mat[battery_id]["cycle"][0, 0]
    for i in range(cycles.shape[1]):
        cycle = cycles[0, i]
        struct = cycle["data"][0, 0]
        data: dict[str, Any] = {}
        for name in struct.dtype.names:
            field = struct[name]
            data[name] = _scalar(field) if name in SCALAR_FIELDS else np.asarray(field).ravel()
        yield (
            str(cycle["type"].ravel()[0]),
            float(np.asarray(cycle["ambient_temperature"]).ravel()[0]),
            np.asarray(cycle["time"], dtype=float).ravel(),
            data,
        )


def _measurements_frame(data: dict[str, Any]) -> pd.DataFrame:
    """Measurement fields of one cycle as columns, NaN-padded to the longest vector."""
    vectors = {k: v for k, v in data.items() if k not in SCALAR_FIELDS}
    if not vectors:
        return cast("pd.DataFrame", pd.DataFrame())
    n = max(len(v) for v in vectors.values())
    padded: dict[str, Any] = {}
    for name, values in vectors.items():
        if len(values) < n:  # impedance spectra have fields of different lengths
            values = np.concatenate(
                [values, np.full(n - len(values), np.nan, dtype=values.dtype if values.dtype.kind == "c" else float)]
            )
        padded[name] = values
    return cast("pd.DataFrame", pd.DataFrame(padded))


def convert_mat_dir(
    mat_dir: str,
    out_dir: str,
    battery_ids: Sequence[str] | None = None,
    log: Callable[[str], object] = print,
) -> pd.DataFrame:
    """Write ``metadata.csv`` and ``data/*.csv`` for every ``B*.mat`` in ``mat_dir``.

    Returns the metadata frame. ``battery_ids`` restricts the conversion to those batteries.
    """
    paths = sorted(glob.glob(os.path.join(mat_dir, "B*.mat")))
    if battery_ids is not None:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in set(battery_ids)]
    if not paths:
        raise FileNotFoundError(f"No B*.mat files found in {mat_dir}")

    data_dir = os.path.join(out_dir, config.CYCLE_DATA_SUBDIR)
    os.makedirs(data_dir, exist_ok=True)
    records: list[dict[str, Any]] = []
    uid = 0
    for path in paths:
        battery_id = os.path.splitext(os.path.basename(path))[0]
        n_before = len(records)
        for test_id, (ctype, ambient, time_vec, data) in enumerate(read_cycles(path)):
            uid += 1
            filename = f"{uid:05d}.csv"
            _measurements_frame(data).to_csv(os.path.join(data_dir, filename), index=False)
            records.append(
                {
                    "type": ctype,
                    "start_time": str(time_vec),
                    "ambient_temperature": ambient,
                    "battery_id": battery_id,
                    "test_id": test_id,
                    "uid": uid,
                    "filename": filename,
                    "Capacity": data.get("Capacity", np.nan),
                    "Re": data.get("Re", np.nan),
                    "Rct": data.get("Rct", np.nan),
                }
            )
        log(f"{battery_id}: {len(records) - n_before} cycles")
    meta = pd.DataFrame(records, columns=METADATA_COLUMNS)
    meta.to_csv(config.metadata_path(out_dir), index=False)
    log(f"Wrote {len(meta)} metadata rows for {len(paths)} batteries to {config.metadata_path(out_dir)}")
    return cast("pd.DataFrame", meta)
