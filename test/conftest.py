"""Shared fixtures: a small synthetic *raw* dataset in the layout of the cleaned NASA release.

The raw layout is ``<root>/metadata.csv`` plus ``<root>/data/<filename>`` per cycle. The
fixture deliberately covers every branch of the ETL: an excluded battery, a non-target
battery, metadata rows out of chronological order, ``start_time`` strings in plain,
scientific and fractional-second form, an impedance row with a malformed ``start_time``,
a discharge with no earlier impedance measurement, an impedance measurement at exactly the
discharge start time, one cycle for each skip reason (too short, low capacity, file
missing), all six designated held-out batteries, and two 4 C batteries that exercise the
relative capacity floor (B0047: best cycle above 1.4 Ah; B0056: never reaches 1.4 Ah).

Expected survival with the default config (see test_etl.py):
  B0005 2/5, B0006 2/2, B0025 2/2, B0030 1/1, B0036 1/1, B0042 1/2, B0047 2/3, B0056 2/3
  -> 13 of 19 candidate cycles; train = B0005 + B0006 (4 cycles), held-out = 9 cycles.
"""
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def _tests_run_on_cpu():
    """Pin PyTorch's default device to CPU for the whole session.

    DeepXDE makes CUDA -- or Apple MPS -- the default device at import time when it detects
    one. Production code chooses its device explicitly (``battery_pinn.device``); the tests
    must behave the same on every CI runner, so they always use the CPU.
    """
    try:
        import torch
    except ImportError:
        return
    torch.set_default_device("cpu")


RAW_CYCLE_COLUMNS = [
    "Voltage_measured", "Current_measured", "Temperature_measured",
    "Current_load", "Voltage_load", "Time",
]


def np_style(y, m, d, hr, mi, sec, sci=False):
    """Render a start time the way the cleaned dataset stores it: a printed numpy array."""
    arr = np.array([y, m, d, hr, mi, sec], dtype=float)
    if sci:
        return np.array2string(arr, formatter={"float_kind": lambda v: f"{v:.8e}"})
    return str(arr)


def make_cycle_frame(n, ambient, drop_at=None, v_high=4.1, v_mid=3.2, v_low=1.9, seed=0):
    """A discharge curve of ``n`` samples whose voltage falls below ``v_mid`` at ``drop_at``."""
    rng = np.random.default_rng(seed)
    voltage = np.linspace(v_high, v_mid, n)
    if drop_at is not None:
        voltage[drop_at:] = np.linspace(v_low, v_low - 0.3, n - drop_at)
    time = np.cumsum(rng.uniform(9.0, 11.0, n)) - 9.0 + rng.uniform(0.0, 2.0)
    current = -2.0 + rng.normal(0.0, 0.01, n)
    temperature = ambient + 8.0 * (1 - np.exp(-np.arange(n) / (0.4 * n))) + rng.normal(0, 0.05, n)
    return pd.DataFrame({
        "Voltage_measured": voltage,
        "Current_measured": current,
        "Temperature_measured": temperature,
        "Current_load": np.abs(current),
        "Voltage_load": voltage - 0.05,
        "Time": time,
    })[RAW_CYCLE_COLUMNS]


# Each battery: ambient, list of (type, start_time args, extra fields, cycle spec or None).
# Times are (y, m, d, hr, mi, sec). "sci" renders the start_time in scientific notation.
# Spec keys: n, drop_at, write (default True).
RAW_SPEC = {
    "B0005": {  # training battery, cutoff 2.7 V, ambient 24
        "ambient": 24,
        "rows": [
            ("impedance", (2008, 4, 1, 9, 0, 0), {"Re": 0.05, "Rct": 0.10}, None),
            ("charge", (2008, 4, 2, 12, 0, 0), {}, None),
            ("discharge", (2008, 4, 2, 15, 25, 41), {"Capacity": 1.85}, {"n": 100, "drop_at": 80}),
            ("discharge", (2008, 4, 3, 10, 0, 0), {"Capacity": 1.80}, {"n": 100, "drop_at": 30}),
            ("impedance", (2008, 4, 3, 12, 0, 0), {"Re": 0.06, "Rct": 0.12}, None),
            ("discharge", (2008, 4, 4, 10, 0, 0), {"Capacity": 1.20}, {"n": 100, "drop_at": 80}),  # low capacity
            ("impedance", "[2008.   4.]", {"Re": 0.99, "Rct": 0.99}, None),  # malformed -> dropped
            ("discharge", (2008, 4, 5, 10, 0, 0), {"Capacity": 1.75}, {"n": 100, "drop_at": 80, "write": False}),
            ("impedance", (2008, 4, 6, 0, 0, 0, "sci"), {"Re": 0.07, "Rct": 0.14}, None),
            ("discharge", (2008, 4, 6, 10, 0, 0, "sci"), {"Capacity": 1.70}, {"n": 200, "drop_at": None}),
        ],
    },
    "B0006": {  # training battery, cutoff 2.5 V, ambient 43; metadata rows are shuffled below
        "ambient": 43,
        "rows": [
            ("discharge", (2008, 5, 3, 10, 0, 0), {"Capacity": 1.90}, {"n": 120, "drop_at": 90}),
            ("impedance", (2008, 5, 3, 10, 0, 0), {"Re": 0.09, "Rct": 0.18}, None),  # exact tie
            ("impedance", (2008, 5, 2, 9, 0, 0), {"Re": 0.08, "Rct": 0.16}, None),
            ("discharge", (2008, 5, 1, 10, 0, 0), {"Capacity": 2.00}, {"n": 60, "drop_at": None}),
        ],
    },
    "B0025": {  # held-out battery, cutoff 2.0 V, ambient 4
        "ambient": 4,
        "rows": [
            ("impedance", (2008, 6, 1, 8, 0, 0), {"Re": 0.11, "Rct": 0.22}, None),
            ("discharge", (2008, 6, 1, 10, 0, 17.234), {"Capacity": 1.60}, {"n": 75, "drop_at": 70}),
            ("discharge", (2008, 6, 2, 10, 0, 0), {"Capacity": 1.50}, {"n": 75, "drop_at": None}),
        ],
    },
    "B0030": {  # held-out, cutoff 2.2 V, ambient 43
        "ambient": 43,
        "rows": [
            ("impedance", (2009, 1, 1, 8, 0, 0), {"Re": 0.12, "Rct": 0.24}, None),
            ("discharge", (2009, 1, 1, 10, 0, 0), {"Capacity": 1.70}, {"n": 80, "drop_at": None}),
        ],
    },
    "B0036": {  # held-out, cutoff 2.7 V, ambient 24
        "ambient": 24,
        "rows": [
            ("impedance", (2009, 2, 1, 8, 0, 0), {"Re": 0.13, "Rct": 0.26}, None),
            ("discharge", (2009, 2, 1, 10, 0, 0), {"Capacity": 1.90}, {"n": 90, "drop_at": 85}),
        ],
    },
    "B0042": {  # held-out, cutoff 2.2 V, ambient 22; second cycle is too short
        "ambient": 22,
        "rows": [
            ("impedance", (2009, 3, 1, 8, 0, 0), {"Re": 0.14, "Rct": 0.28}, None),
            ("discharge", (2009, 3, 1, 10, 0, 0), {"Capacity": 1.65}, {"n": 100, "drop_at": 90}),
            ("discharge", (2009, 3, 2, 10, 0, 0), {"Capacity": 1.60}, {"n": 100, "drop_at": 40}),
        ],
    },
    "B0047": {  # held-out, cutoff 2.5 V, ambient 4: best cycle above 1.4 Ah, later ones below it
        "ambient": 4,
        "rows": [
            ("impedance", (2010, 1, 1, 8, 0, 0), {"Re": 0.15, "Rct": 0.30}, None),
            ("discharge", (2010, 1, 1, 10, 0, 0), {"Capacity": 1.60}, {"n": 100, "drop_at": None}),
            ("discharge", (2010, 1, 2, 10, 0, 0), {"Capacity": 1.20}, {"n": 100, "drop_at": None}),  # > 70 % of 1.60
            ("discharge", (2010, 1, 3, 10, 0, 0), {"Capacity": 1.00}, {"n": 100, "drop_at": None}),  # < 70 % of 1.60
        ],
    },
    "B0056": {  # held-out, cutoff 2.7 V, ambient 4: never reaches 1.4 Ah (the real B0056 case)
        "ambient": 4,
        "rows": [
            ("impedance", (2010, 9, 1, 8, 0, 0), {"Re": 0.16, "Rct": 0.32}, None),
            ("discharge", (2010, 9, 1, 10, 0, 0), {"Capacity": 1.30}, {"n": 100, "drop_at": 95}),
            ("discharge", (2010, 9, 2, 10, 0, 0), {"Capacity": 1.25}, {"n": 100, "drop_at": 95}),
            ("discharge", (2010, 9, 3, 10, 0, 0), {"Capacity": 0.80}, {"n": 100, "drop_at": 95}),  # < 70 % of 1.30
        ],
    },
    "B0049": {  # excluded at the bronze -> silver step
        "ambient": 4,
        "rows": [
            ("impedance", (2008, 7, 1, 8, 0, 0), {"Re": 0.2, "Rct": 0.4}, None),
            ("discharge", (2008, 7, 1, 10, 0, 0), {"Capacity": 1.8}, {"n": 100, "drop_at": None}),
        ],
    },
    "B0099": {  # valid but not a target battery: reaches gold, never the processed tables
        "ambient": 24,
        "rows": [
            ("discharge", "[2008. 8.]", {"Capacity": 1.8}, {"n": 100, "drop_at": None}),  # malformed -> NaT
            ("discharge", (2008, 8, 2, 10, 0, 0), {"Capacity": 1.7}, {"n": 100, "drop_at": None}),
        ],
    },
}


def build_raw_dataset(root, spec=RAW_SPEC, shuffle_seed=1):
    """Write ``metadata.csv`` and the cycle files under ``root``; return the metadata frame."""
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    records = []
    uid = 0
    for battery_id, bspec in spec.items():
        for row_type, when, extra, cycle in bspec["rows"]:
            uid += 1
            filename = f"{uid:05d}.csv"
            if isinstance(when, str):
                start_time = when
            else:
                sci = len(when) == 7 and when[6] == "sci"
                start_time = np_style(*when[:6], sci=sci)
            records.append({
                "type": row_type,
                "start_time": start_time,
                "ambient_temperature": bspec["ambient"],
                "battery_id": battery_id,
                "test_id": uid,
                "uid": uid,
                "filename": filename,
                "Capacity": extra.get("Capacity", np.nan),
                "Re": extra.get("Re", np.nan),
                "Rct": extra.get("Rct", np.nan),
            })
            if cycle is not None and cycle.get("write", True):
                frame = make_cycle_frame(cycle["n"], bspec["ambient"], cycle["drop_at"], seed=uid)
                frame.to_csv(os.path.join(root, "data", filename), index=False)
    meta = pd.DataFrame(records)
    # Out-of-order rows: the ETL must not rely on file order for cycle numbering.
    meta = meta.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    meta.to_csv(os.path.join(root, "metadata.csv"), index=False)
    return meta


@pytest.fixture(scope="session")
def raw_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw_nasa")
    build_raw_dataset(str(root))
    return str(root)


def write_fake_mat(path, battery_id, n_samples=120, ambient=24.0):
    """A minimal NASA-style ``B00xx.mat``: charge, impedance and two discharge cycles.

    Mirrors the real structure (``<battery_id>.cycle`` struct array with ``type``,
    ``ambient_temperature``, ``time`` and a ``data`` struct) closely enough for
    ``battery_pinn.nasa_mat.read_cycles``.
    """
    from scipy.io import savemat

    t = np.arange(n_samples, dtype=float) * 10.0
    volts = np.linspace(4.1, 2.6, n_samples)

    def struct(**fields):
        return np.array([[tuple(fields.values())]], dtype=[(k, "O") for k in fields])

    charge = struct(Voltage_measured=volts[::-1], Current_measured=np.full(n_samples, 1.5),
                    Temperature_measured=np.full(n_samples, ambient), Current_charge=np.full(n_samples, 1.5),
                    Voltage_charge=volts[::-1], Time=t)
    impedance = struct(Sense_current=np.ones(5, dtype=complex), Battery_current=np.ones(5, dtype=complex),
                       Current_ratio=np.ones(5, dtype=complex), Battery_impedance=np.ones(5, dtype=complex),
                       Rectified_Impedance=np.ones(3, dtype=complex), Re=np.array([[0.05]]), Rct=np.array([[0.1]]))

    def discharge(capacity):
        return struct(Voltage_measured=volts, Current_measured=np.full(n_samples, -2.0),
                      Temperature_measured=ambient + np.linspace(0, 6, n_samples),
                      Current_load=np.full(n_samples, 2.0), Voltage_load=volts - 0.05, Time=t,
                      Capacity=np.array([[capacity]]))

    cycles = [
        ("charge", [2008, 4, 2, 12, 0, 0.0], charge),
        ("impedance", [2008, 4, 2, 13, 0, 17.234], impedance),
        ("discharge", [2008, 4, 2, 15, 25, 41.0], discharge(1.85)),
        ("discharge", [2008, 4, 3, 10, 0, 0.0], discharge(1.80)),
    ]
    cycle_arr = np.empty((1, len(cycles)), dtype=[("type", "O"), ("ambient_temperature", "O"),
                                                  ("time", "O"), ("data", "O")])
    for i, (ctype, when, data) in enumerate(cycles):
        cycle_arr[0, i] = (np.array([ctype]), np.array([[ambient]]), np.array([when], dtype=float), data)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    savemat(str(path), {battery_id: {"cycle": cycle_arr}})
