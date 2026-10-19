"""The local (pandas/NumPy) raw-to-processed pipeline.

Covers each medallion layer against hand-computed expectations from the fixture in
``conftest.py``, and checks the feature-engineering stage against a verbatim copy of the
original Databricks loop so the port is proven equivalent rather than merely plausible.
"""
import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from battery_pinn import config, etl
from battery_pinn.data import BatteryCoverageError, load_dataset
from conftest import RAW_SPEC, build_raw_dataset, make_cycle_frame


@pytest.fixture(scope="module")
def layers(raw_dataset):
    bronze = etl.load_raw_metadata(raw_dataset)
    discharge = etl.build_silver_discharge(bronze)
    impedance = etl.build_silver_impedance(bronze)
    gold = etl.build_gold(discharge, impedance)
    return {"bronze": bronze, "discharge": discharge, "impedance": impedance, "gold": gold}


def _gold_row(gold, battery_id, cycle_number):
    rows = gold[(gold["battery_id"] == battery_id) & (gold["cycle_number"] == cycle_number)]
    assert len(rows) == 1
    return rows.iloc[0]


# ----------------------------------------------------------------------------------------
# Bronze
# ----------------------------------------------------------------------------------------
def test_bronze_matches_metadata_columns_case_insensitively(layers):
    bronze = layers["bronze"]
    assert "capacity" in bronze.columns and "Capacity" not in bronze.columns
    assert set(config.RAW_METADATA_COLUMNS) <= set(bronze.columns)
    assert len(bronze) == sum(len(b["rows"]) for b in RAW_SPEC.values())


def test_bronze_requires_metadata_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata.csv"):
        etl.load_raw_metadata(str(tmp_path))


def test_bronze_reports_missing_columns(tmp_path):
    (tmp_path / "metadata.csv").write_text("battery_id,type\nB0005,discharge\n")
    with pytest.raises(ValueError, match="missing required columns"):
        etl.load_raw_metadata(str(tmp_path))


# ----------------------------------------------------------------------------------------
# Silver: start_time parsing and cycle numbering
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("[2008.    4.    2.   15.   25.   41.]", pd.Timestamp("2008-04-02 15:25:41")),
    ("[2.008e+03 4.000e+00 2.000e+00 1.500e+01 2.500e+01 4.100e+01]", pd.Timestamp("2008-04-02 15:25:41")),
    ("[2008.  6.  1. 10.  0. 17.234]", pd.Timestamp("2008-06-01 10:00:17.234")),
    ("[2008. 6. 1. 10. 0. 17.3]", pd.Timestamp("2008-06-01 10:00:17.300")),  # no binary round-off
    ("[2.00800000e+03 4.00000000e+00 6.00000000e+00 1.00000000e+01\n 0.00000000e+00 0.00000000e+00]",
     pd.Timestamp("2008-04-06 10:00:00")),  # numpy line-wraps long arrays
])
def test_parse_start_time_formats(text, expected):
    parsed = etl.parse_start_time(pd.Series([text]))
    assert parsed.loc[0, "n_parts"] == 6
    assert parsed.loc[0, "start_datetime"] == expected


@pytest.mark.parametrize("text", ["[2008.   4.]", "", "not a date", None, "[2008. 13. 1. 0. 0. 0.]"])
def test_parse_start_time_invalid_gives_nat(text):
    parsed = etl.parse_start_time(pd.Series([text]))
    assert pd.isna(parsed.loc[0, "start_datetime"])


def test_parse_start_time_truncates_components_like_cast_to_int():
    parsed = etl.parse_start_time(pd.Series(["[2008.9 4.7 2.2 15.9 25.1 41.0]"]))
    assert parsed.loc[0, ["y", "m", "d", "hr", "mi"]].tolist() == [2008, 4, 2, 15, 25]
    assert parsed.loc[0, "start_datetime"] == pd.Timestamp("2008-04-02 15:25:41")


def test_silver_discharge_only_keeps_discharge_rows_of_non_excluded_batteries(layers):
    discharge = layers["discharge"]
    assert set(discharge["type"]) == {"discharge"}
    assert not set(discharge["battery_id"]) & set(config.EXCLUDED_BATTERIES)
    assert set(discharge["battery_id"]) == {"B0005", "B0006", "B0025", "B0030", "B0036", "B0042",
                                            "B0047", "B0056", "B0099"}
    assert list(discharge.columns) == [
        "battery_id", "type", "filename", "ambient_temperature", "capacity",
        "start_datetime", "cycle_number",
    ]


def test_silver_discharge_cycle_numbers_follow_time_not_file_order(layers):
    discharge = layers["discharge"].sort_values(["battery_id", "cycle_number"])
    b6 = discharge[discharge["battery_id"] == "B0006"]
    assert b6["cycle_number"].tolist() == [1, 2]
    assert b6["start_datetime"].tolist() == [pd.Timestamp("2008-05-01 10:00"), pd.Timestamp("2008-05-03 10:00")]
    assert b6["capacity"].tolist() == [2.0, 1.9]  # row order in metadata.csv was reversed
    b5 = discharge[discharge["battery_id"] == "B0005"]
    assert b5["cycle_number"].tolist() == [1, 2, 3, 4, 5]
    assert b5["start_datetime"].is_monotonic_increasing


def test_silver_discharge_keeps_unparseable_start_time_as_nat(layers):
    b99 = layers["discharge"][layers["discharge"]["battery_id"] == "B0099"]
    assert len(b99) == 2
    nat_rows = b99[b99["start_datetime"].isna()]
    assert len(nat_rows) == 1
    assert nat_rows["cycle_number"].item() == 1  # NULLs sort first, as in Spark


def test_silver_impedance_drops_malformed_rows_and_numbers_cycles(layers):
    impedance = layers["impedance"]
    assert list(impedance.columns) == ["battery_id", "Re", "Rct", "start_datetime", "cycle_number"]
    assert not (impedance["Re"] == 0.99).any()  # the "[2008.   4.]" row
    assert "B0049" not in set(impedance["battery_id"])
    b5 = impedance[impedance["battery_id"] == "B0005"].sort_values("cycle_number")
    assert b5["cycle_number"].tolist() == [1, 2, 3]
    assert b5["Re"].tolist() == pytest.approx([0.05, 0.06, 0.07])


# ----------------------------------------------------------------------------------------
# Gold: as-of impedance join
# ----------------------------------------------------------------------------------------
def test_gold_has_one_row_per_discharge_file(layers):
    gold, discharge = layers["gold"], layers["discharge"]
    assert len(gold) == len(discharge) == gold["filename"].nunique()
    assert list(gold.columns) == [
        "battery_id", "type", "filename", "ambient_temperature", "start_datetime",
        "cycle_number", "capacity", "Re", "Rct",
    ]


def test_gold_attaches_latest_impedance_at_or_before_discharge(layers):
    gold = layers["gold"]
    f32 = np.float32
    expected = {  # (battery, cycle) -> (Re, Rct)
        ("B0005", 1): (0.05, 0.10),  # only the 04-01 measurement precedes it
        ("B0005", 2): (0.05, 0.10),  # 04-03 12:00 measurement is two hours *after* this discharge
        ("B0005", 3): (0.06, 0.12),
        ("B0005", 4): (0.06, 0.12),  # the malformed 0.99 row must not be used
        ("B0005", 5): (0.07, 0.14),
        ("B0006", 2): (0.09, 0.18),  # measurement at exactly the discharge start counts (<=)
        ("B0025", 1): (0.11, 0.22),
        ("B0025", 2): (0.11, 0.22),
    }
    for (bid, cyc), (re, rct) in expected.items():
        row = _gold_row(gold, bid, cyc)
        assert row["Re"] == f32(re) and row["Rct"] == f32(rct), (bid, cyc)


def test_gold_discharge_without_prior_impedance_has_nan(layers):
    row = _gold_row(layers["gold"], "B0006", 1)
    assert np.isnan(row["Re"]) and np.isnan(row["Rct"])
    for cyc in (1, 2):  # B0099 has no impedance rows at all; cycle 1 also has a NaT start
        row = _gold_row(layers["gold"], "B0099", cyc)
        assert np.isnan(row["Re"]) and np.isnan(row["Rct"])


def test_gold_numeric_columns_are_float32_like_the_delta_table(layers):
    gold = layers["gold"]
    for col in ("capacity", "Re", "Rct"):
        assert gold[col].dtype == np.float32
    assert _gold_row(gold, "B0005", 1)["capacity"] == np.float32(1.85)


def test_gold_keeps_first_row_per_duplicate_filename():
    discharge = pd.DataFrame({
        "battery_id": ["B0005", "B0005"], "type": "discharge", "filename": "same.csv",
        "ambient_temperature": 24, "capacity": [1.8, 1.7],
        "start_datetime": pd.to_datetime(["2008-04-05", "2008-04-02"]), "cycle_number": [2, 1],
    })
    impedance = pd.DataFrame({
        "battery_id": ["B0005"], "Re": [0.05], "Rct": [0.1],
        "start_datetime": pd.to_datetime(["2008-04-01"]), "cycle_number": [1],
    })
    gold = etl.build_gold(discharge, impedance)
    assert len(gold) == 1
    assert gold.loc[0, "start_datetime"] == pd.Timestamp("2008-04-02")


# ----------------------------------------------------------------------------------------
# Feature engineering: the per-cycle steps
# ----------------------------------------------------------------------------------------
def test_truncate_at_cutoff_keeps_rows_before_first_sample_below_limit():
    volts = [3.0] * 10 + [2.7, 2.7, 2.69, 3.0, 2.0]
    cycle = pd.DataFrame({"Voltage_measured": volts})
    truncated = etl.truncate_at_cutoff(cycle, 2.7)
    assert len(truncated) == 12            # 2.7 == cutoff is kept; 2.69 is the first below
    assert truncated.index.tolist() == list(range(12))
    assert len(etl.truncate_at_cutoff(cycle, 2.0)) == 15   # nothing below -> untouched
    assert len(etl.truncate_at_cutoff(cycle, 3.5)) == 0    # first sample already below


def test_soc_is_coulomb_counting_from_100_percent():
    cycle = pd.DataFrame({"Time": [0.0, 3600.0, 7200.0], "Current_measured": [-1.0, -1.0, -0.5]})
    out = etl.add_state_of_charge(cycle, capacity_ah=2.0)
    # Delta_Q uses the *current* row's current times the elapsed time since the previous row.
    assert out["Delta_Q"].tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert out["SoC"].tolist() == pytest.approx([100.0, 50.0, 25.0])


def test_downsample_cycle_averages_fixed_number_of_bins():
    n = 200
    cycle = pd.DataFrame({
        "Time": np.arange(n, dtype=float), "Voltage_measured": np.linspace(4, 3, n),
        "Current_measured": -2.0, "Temperature_measured": 25.0, "SoC": np.linspace(100, 0, n),
    })
    agg = etl.downsample_cycle(cycle.copy(), num_bins=50)
    assert len(agg) == 50
    assert agg["bin"].tolist() == list(range(50))
    assert agg["Time"].tolist() == pytest.approx([4 * i + 1.5 for i in range(50)])  # mean of 4 rows
    assert list(agg.columns) == ["bin", "Time", "Voltage_measured", "Current_measured",
                                 "Temperature_measured", "SoC"]
    assert len(etl.downsample_cycle(cycle.copy())) == config.BINS_PER_CYCLE


def _meta(**overrides):
    meta = {"battery_id": "B0005", "cycle_number": 3, "capacity": np.float32(1.8),
            "Re": np.float32(0.05), "Rct": np.float32(0.1), "ambient_temperature": 24}
    meta.update(overrides)
    return pd.Series(meta)


def test_process_cycle_skip_reasons():
    short = make_cycle_frame(100, 24, drop_at=30)
    assert etl.process_cycle(short, _meta(), 2.0, cutoff_voltage=2.7) == (None, "too_short")
    # Exactly MIN_SAMPLES_PER_CYCLE rows is enough; one fewer is not.
    ok = make_cycle_frame(100, 24, drop_at=config.MIN_SAMPLES_PER_CYCLE)
    assert etl.process_cycle(ok, _meta(), 2.0, 2.7)[1] is None
    one_short = make_cycle_frame(100, 24, drop_at=config.MIN_SAMPLES_PER_CYCLE - 1)
    assert etl.process_cycle(one_short, _meta(), 2.0, 2.7)[1] == "too_short"
    # Capacity at the floor is skipped, just above it is kept; too_short is checked first.
    full = make_cycle_frame(100, 24, drop_at=None)
    floor = config.capacity_floor(2.0)
    assert floor == config.MIN_VALID_CAPACITY_AH  # 70 % of a 2.0 Ah best cycle is 1.4 Ah
    assert etl.process_cycle(full, _meta(capacity=floor), 2.0, 2.7)[1] == "low_capacity"
    assert etl.process_cycle(full, _meta(capacity=floor + 1e-3), 2.0, 2.7)[1] is None
    assert etl.process_cycle(short, _meta(capacity=1.0), 2.0, 2.7) == (None, "too_short")
    # The floor follows the battery's own best cycle when that is below 1.4 Ah ...
    assert etl.process_cycle(full, _meta(capacity=1.0), max_capacity=1.3, cutoff_voltage=2.7)[1] is None
    assert etl.process_cycle(full, _meta(capacity=0.9), max_capacity=1.3, cutoff_voltage=2.7)[1] == "low_capacity"
    # ... and an explicit floor overrides it.
    assert etl.process_cycle(full, _meta(capacity=1.0), 1.3, 2.7, capacity_floor_ah=1.4)[1] == "low_capacity"


def test_process_cycle_labels_and_normalised_time():
    frame, reason = etl.process_cycle(make_cycle_frame(100, 24, drop_at=80), _meta(), np.float32(2.0), 2.7)
    assert reason is None
    assert list(frame.columns) == config.PROCESSED_COLUMNS
    assert len(frame) == config.BINS_PER_CYCLE
    assert (frame["battery_id"] == "B0005").all() and (frame["cycle_number"] == 3).all()
    assert frame["SoH"].unique().tolist() == pytest.approx([1.8 / 2.0 * 100])
    assert frame["Re"].unique().tolist() == [float(np.float32(0.05))]
    assert frame["ambient_temperature"].unique().tolist() == [24.0]
    assert frame["Time_normalized"].max() == 1.0
    assert frame["Time_normalized"].is_monotonic_increasing
    assert frame["SoC"].is_monotonic_decreasing and frame["SoC"].iloc[0] <= 100.0


def test_process_cycle_missing_impedance_and_capacity_pass_through():
    frame, reason = etl.process_cycle(
        make_cycle_frame(100, 24, drop_at=None), _meta(Re=np.nan, Rct=np.nan, capacity=np.nan), 2.0, 2.7)
    assert reason is None  # NaN capacity is not "<= 1.4": the original did not skip it either
    assert frame["Re"].isna().all() and frame["Rct"].isna().all()
    assert frame["SoH"].isna().all() and frame["SoC"].isna().all()
    assert frame["Re"].dtype == float


# ----------------------------------------------------------------------------------------
# Feature table, split and the end-to-end run
# ----------------------------------------------------------------------------------------
def test_feature_table_summary_and_skip_counts(layers, raw_dataset):
    dataset, summary = etl.build_feature_table(layers["gold"], raw_dataset)
    assert summary["total_candidates"] == 19          # B0049 excluded, B0099 not a target
    assert summary["total_kept"] == 13
    assert summary["skip_counts"] == {"file_not_found": 1, "too_short": 2, "low_capacity": 3}
    per = summary["per_battery"]
    assert list(per) == config.TARGET_BATTERIES          # every designated battery, in order
    assert per["B0005"]["candidates"] == 5 and per["B0005"]["kept"] == 2
    assert per["B0005"]["dropped"] == {"file_not_found": 1, "too_short": 1, "low_capacity": 1}
    assert per["B0005"]["role"] == "train" and per["B0025"]["role"] == "held-out"
    assert per["B0042"]["dropped"]["too_short"] == 1 and per["B0042"]["kept"] == 1
    absent = per["B0007"]
    assert absent == {"role": "train", "present": False, "capacity_floor_ah": config.MIN_VALID_CAPACITY_AH,
                      "max_capacity_ah": pytest.approx(np.nan, nan_ok=True), "candidates": 0, "kept": 0,
                      "dropped": {"file_not_found": 0, "too_short": 0, "low_capacity": 0}}
    assert len(dataset) == 13 * config.BINS_PER_CYCLE
    assert list(dataset.columns) == config.PROCESSED_COLUMNS
    assert set(dataset["battery_id"]) == {"B0005", "B0006", "B0025", "B0030", "B0036", "B0042", "B0047", "B0056"}


def test_feature_table_uses_per_battery_cutoff_and_max_capacity(layers, raw_dataset):
    dataset, _ = etl.build_feature_table(layers["gold"], raw_dataset)
    b5 = dataset[dataset["battery_id"] == "B0005"]
    assert sorted(b5["cycle_number"].unique()) == [1, 5]
    soh = b5.groupby("cycle_number")["SoH"].first()
    assert soh[1] == pytest.approx(100.0)
    assert soh[5] == pytest.approx(float(np.float32(1.70)) / float(np.float32(1.85)) * 100)
    # B0006 cycle 1 (60 samples, never below 2.5 V): all samples survive, spread over 50 bins.
    b6c1 = dataset[(dataset["battery_id"] == "B0006") & (dataset["cycle_number"] == 1)]
    assert len(b6c1) == 50 and b6c1["Re"].isna().all()


def test_capacity_floor_is_nominal_or_relative_whichever_is_lower(layers, raw_dataset):
    """The B0056 case: a 4 C battery whose best cycle never reaches the 1.4 Ah nominal floor."""
    dataset, summary = etl.build_feature_table(layers["gold"], raw_dataset)
    per = summary["per_battery"]
    # B0056: best 1.30 Ah -> floor 0.91 Ah; 1.30 and 1.25 kept, 0.80 dropped.
    assert per["B0056"]["capacity_floor_ah"] == pytest.approx(0.7 * float(np.float32(1.30)))
    assert per["B0056"]["kept"] == 2 and per["B0056"]["dropped"]["low_capacity"] == 1
    # B0047: best 1.60 Ah -> floor 1.12 Ah; the 1.20 Ah cycle (below 1.4) is kept, 1.00 dropped.
    assert per["B0047"]["capacity_floor_ah"] == pytest.approx(0.7 * float(np.float32(1.60)))
    assert sorted(dataset[dataset["battery_id"] == "B0047"]["cycle_number"].unique()) == [1, 2]
    # B0005: best 1.85 Ah -> floor 1.295 Ah (below the nominal floor), the 1.20 Ah cycle stays dropped.
    assert per["B0005"]["capacity_floor_ah"] == pytest.approx(0.7 * float(np.float32(1.85)))
    # The floor is never stricter than the nominal one.
    assert all(r["capacity_floor_ah"] <= config.MIN_VALID_CAPACITY_AH for r in per.values())


def test_capacity_floor_can_be_restored_to_the_absolute_rule(layers, raw_dataset):
    dataset, summary = etl.build_feature_table(layers["gold"], raw_dataset, min_capacity_fraction=None)
    per = summary["per_battery"]
    assert all(r["capacity_floor_ah"] == config.MIN_VALID_CAPACITY_AH for r in per.values())
    assert per["B0056"]["kept"] == 0 and per["B0056"]["dropped"]["low_capacity"] == 3
    assert per["B0047"]["kept"] == 1
    assert "B0056" not in set(dataset["battery_id"])
    strict = etl.build_feature_table(layers["gold"], raw_dataset, min_capacity=2.1, min_capacity_fraction=None)[1]
    assert strict["total_kept"] == 0


def test_feature_table_default_cutoff_for_unknown_battery(layers, raw_dataset):
    gold = layers["gold"]
    with_unknown = etl.build_feature_table(gold, raw_dataset, target_batteries=["B0099"], cutoff_map={})
    assert with_unknown[1]["total_kept"] == 2
    strict = etl.build_feature_table(gold, raw_dataset, target_batteries=["B0099"], cutoff_map={},
                                     default_cutoff=5.0)
    assert strict[1]["skip_counts"]["too_short"] == 2 and strict[0].empty


def test_split_is_by_battery(layers, raw_dataset):
    dataset, _ = etl.build_feature_table(layers["gold"], raw_dataset)
    train, held_out = etl.split_train_held_out(dataset)
    assert set(train["battery_id"]) == {"B0005", "B0006"}
    assert set(held_out["battery_id"]) == set(config.HELD_OUT_BATTERIES)
    assert set(train["battery_id"]).isdisjoint(config.HELD_OUT_BATTERIES)
    assert len(train) + len(held_out) == len(dataset)
    assert_frame_equal(pd.concat([train, held_out], ignore_index=True), dataset)


def _battery(role, candidates, kept, present=True, **dropped):
    d = {"file_not_found": 0, "too_short": 0, "low_capacity": 0}
    d.update(dropped)
    return {"role": role, "present": present, "capacity_floor_ah": 1.4, "max_capacity_ah": 1.8,
            "candidates": candidates, "kept": kept, "dropped": d}


def test_format_summary_reports_every_designated_battery_and_flags_problems():
    summary = {
        "total_candidates": 20, "total_kept": 3,
        "skip_counts": {"file_not_found": 0, "too_short": 8, "low_capacity": 9},
        "per_battery": {
            "B0005": _battery("train", 5, 0, too_short=5),
            "B0007": _battery("train", 0, 0, present=False),
            "B0025": _battery("held-out", 5, 2, low_capacity=3),
            "B0030": _battery("held-out", 10, 1, too_short=3, low_capacity=6),
            "B0056": _battery("held-out", 0, 0, present=False),
        },
    }
    text = etl.format_summary(summary)
    assert "Kept:                     3 (15.0%)" in text
    assert "WARNING: over half of all cycles were skipped" in text
    assert "B0005: 0/5 cycles kept, dropped: 5 too_short" in text and "ZERO cycles survived!" in text
    assert "B0007: MISSING from raw metadata" in text
    assert "B0025 [HELD-OUT]: 2/5 cycles kept, dropped: 3 low_capacity (capacity floor 1.400 Ah, best cycle 1.800 Ah)" in text
    assert "B0030 [HELD-OUT]: 1/10 cycles kept, dropped: 3 too_short, 6 low_capacity" in text
    assert "<-- fewer than 20% of cycles survived" in text
    assert "B0056 [HELD-OUT]: MISSING from raw metadata" in text
    assert "2 designated batteries are absent from the raw metadata: ['B0007', 'B0056']" in text
    assert "1 batteries had ZERO surviving cycles: ['B0005']" in text
    assert "ERROR: held-out batteries without validation data" in text
    assert etl.coverage_problems(summary) == ["B0056: not present in the raw metadata"]


def test_coverage_problems_explains_zero_survival_of_held_out_battery():
    summary = {"per_battery": {
        "B0005": _battery("train", 5, 0, too_short=5),                       # training battery: not fatal
        "B0056": _battery("held-out", 102, 0, low_capacity=102) | {"capacity_floor_ah": 1.4, "max_capacity_ah": 1.344},
    }}
    problems = etl.coverage_problems(summary)
    assert problems == ["B0056: 0 of 102 cycles survived (102 low_capacity; capacity floor 1.400 Ah, "
                        "best cycle 1.344 Ah)"]


def test_run_pipeline_writes_tables_the_trainer_can_load(raw_dataset, tmp_path):
    out = tmp_path / "processed"
    messages = []
    summary = etl.run_pipeline(raw_dataset, str(out), log=messages.append)

    for table in (config.SILVER_DISCHARGE_TABLE, config.SILVER_IMPEDANCE_TABLE, config.GOLD_TABLE,
                  config.TRAIN_TABLE, config.HELD_OUT_TABLE):
        assert os.path.isfile(summary["outputs"][table])
        assert summary["outputs"][table] == config.table_path(str(out), table)

    train = load_dataset(summary["outputs"][config.TRAIN_TABLE])
    held_out = load_dataset(summary["outputs"][config.HELD_OUT_TABLE])
    assert summary["outputs"][config.TRAIN_TABLE] == str(out / "train.csv")
    assert summary["outputs"][config.HELD_OUT_TABLE] == str(out / "held_out.csv")
    assert list(pd.read_csv(summary["outputs"][config.TRAIN_TABLE]).columns) == config.PROCESSED_COLUMNS
    # B0006 cycle 1 has no Re, so load_dataset drops its 50 rows -- the same rows the
    # Databricks physics stage dropped with its dropna.
    assert len(train) == 4 * config.BINS_PER_CYCLE - config.BINS_PER_CYCLE
    assert len(held_out) == 9 * config.BINS_PER_CYCLE
    assert set(held_out["battery_id"]) == set(config.HELD_OUT_BATTERIES)
    text = "\n".join(messages)
    assert "Total candidate cycles:   19" in text
    assert "B0007: MISSING from raw metadata" in text      # designated but not in this fixture
    assert "B0056 [HELD-OUT]: 2/3 cycles kept, dropped: 1 low_capacity" in text


def test_run_pipeline_fails_loudly_when_nothing_survives(tmp_path):
    root = tmp_path / "raw"
    spec = {"B0005": {"ambient": 24, "rows": [
        ("discharge", (2008, 4, 2, 15, 25, 41), {"Capacity": 1.8}, {"n": 100, "drop_at": 10}),  # too short
    ]}}
    build_raw_dataset(str(root), spec=spec)
    with pytest.raises(RuntimeError, match="No cycles survived"):
        etl.run_pipeline(str(root), str(tmp_path / "out"), log=lambda *_: None)


def _spec_without(*battery_ids):
    return {bid: spec for bid, spec in RAW_SPEC.items() if bid not in battery_ids}


def test_run_pipeline_held_out_missing_from_metadata_is_a_hard_error(tmp_path):
    root = tmp_path / "raw"
    build_raw_dataset(str(root), spec=_spec_without("B0056"))
    out = tmp_path / "out"
    with pytest.raises(BatteryCoverageError, match=r"B0056: not present in the raw metadata"):
        etl.run_pipeline(str(root), str(out), log=lambda *_: None)
    assert not out.exists()  # nothing is written on a hard error


def test_run_pipeline_held_out_zero_survivors_is_a_hard_error(tmp_path, monkeypatch):
    """With the original absolute floor, the fixture's B0056 loses every cycle, as the real one did."""
    monkeypatch.setattr(config, "MIN_CAPACITY_FRACTION_OF_MAX", None)
    root = tmp_path / "raw"
    build_raw_dataset(str(root))
    messages = []
    with pytest.raises(BatteryCoverageError, match=r"B0056: 0 of 3 cycles survived \(3 low_capacity"):
        etl.run_pipeline(str(root), str(tmp_path / "out"), log=messages.append)
    assert "B0056 [HELD-OUT]: 0/3 cycles kept" in "\n".join(messages)
    assert not (tmp_path / "out").exists()


def test_run_pipeline_missing_training_battery_is_only_a_warning(tmp_path):
    root = tmp_path / "raw"
    build_raw_dataset(str(root), spec=_spec_without("B0006"))
    messages = []
    etl.run_pipeline(str(root), str(tmp_path / "out"), log=messages.append)
    assert "B0006: MISSING from raw metadata" in "\n".join(messages)
    assert (tmp_path / "out" / config.TRAIN_FILE).is_file()


# ----------------------------------------------------------------------------------------
# Parity with the original Databricks feature-engineering loop
# ----------------------------------------------------------------------------------------
def _original_feature_engineering(discharge_metadata, max_caps_df, raw_data_dir):
    """Verbatim copy of the pandas loop from the Databricks ``feature_engineering`` stage.

    Only the Spark reads/writes and the progress bar were removed; every data operation is
    exactly as it was -- except the capacity comparison, which (like the stage file itself)
    now goes through ``config.capacity_floor`` instead of the bare ``MIN_VALID_CAPACITY_AH``.
    With ``MIN_CAPACITY_FRACTION_OF_MAX = None`` that is the original comparison.
    """
    cutoff_map = config.CUTOFF_VOLTAGE_MAP
    MIN_SAMPLES_PER_CYCLE = config.MIN_SAMPLES_PER_CYCLE
    processed_dfs = []
    for _, row in discharge_metadata.iterrows():
        file_path = f"{raw_data_dir}/data/{row['filename']}"
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path).copy()

        v_limit = cutoff_map.get(row['battery_id'], 2.7)
        cutoff_idx = df[df['Voltage_measured'] < v_limit].index.min()
        truncated_df = df if pd.isna(cutoff_idx) else df.iloc[:cutoff_idx].copy()

        if len(truncated_df) < MIN_SAMPLES_PER_CYCLE:
            continue

        db_capacity = float(row['capacity'])
        max_capacity = max_caps_df.loc[row['battery_id'], 'max_cap']

        if db_capacity <= config.capacity_floor(max_capacity):
            continue

        truncated_df['Time_diff_hr'] = truncated_df['Time'].diff().fillna(0) / 3600
        truncated_df['Delta_Q'] = abs(truncated_df['Current_measured'] * truncated_df['Time_diff_hr'])
        truncated_df['Cumulative_Q'] = truncated_df['Delta_Q'].cumsum()
        truncated_df['SoC'] = 100 * (1 - truncated_df['Cumulative_Q'] / db_capacity)

        num_bins = 50
        truncated_df['bin'] = pd.cut(truncated_df.index, bins=num_bins, labels=False)

        agg_cycle = truncated_df.groupby('bin').agg({
            'Time': 'mean',
            'Voltage_measured': 'mean',
            'Current_measured': 'mean',
            'Temperature_measured': 'mean',
            'SoC': 'mean'
        }).reset_index()

        agg_cycle['battery_id'] = row['battery_id']
        agg_cycle['cycle_number'] = row['cycle_number']
        agg_cycle['SoH'] = float((db_capacity / max_capacity) * 100)
        agg_cycle['Re'] = float(row['Re']) if pd.notna(row['Re']) else None
        agg_cycle['Rct'] = float(row['Rct']) if pd.notna(row['Rct']) else None
        agg_cycle['ambient_temperature'] = float(row['ambient_temperature']) if pd.notna(row['ambient_temperature']) else None
        agg_cycle['Time_normalized'] = agg_cycle['Time'] / agg_cycle['Time'].max()

        processed_dfs.append(agg_cycle)

    full_dataset = pd.concat(processed_dfs, ignore_index=True)
    train_dataset = full_dataset[~full_dataset['battery_id'].isin(config.HELD_OUT_BATTERIES)].reset_index(drop=True)
    held_out_dataset = full_dataset[full_dataset['battery_id'].isin(config.HELD_OUT_BATTERIES)].reset_index(drop=True)
    return train_dataset, held_out_dataset


@pytest.mark.filterwarnings("ignore::FutureWarning")  # the oracle concatenates None-valued columns
@pytest.mark.parametrize("fraction", [config.MIN_CAPACITY_FRACTION_OF_MAX, None], ids=["relative-floor", "absolute-floor"])
def test_feature_engineering_matches_original_databricks_loop(layers, raw_dataset, monkeypatch, fraction):
    monkeypatch.setattr(config, "MIN_CAPACITY_FRACTION_OF_MAX", fraction)
    gold = layers["gold"]
    # The original read these from Spark with float32 (FLOAT) columns and float64 `max_cap`
    # computed from them; reproduce those inputs exactly.
    discharge_metadata = gold[gold["battery_id"].isin(config.TARGET_BATTERIES)]
    max_caps_df = gold.groupby("battery_id")["capacity"].max().astype(float).rename("max_cap").to_frame()
    expected_train, expected_held_out = _original_feature_engineering(discharge_metadata, max_caps_df, raw_dataset)

    dataset, _ = etl.build_feature_table(gold, raw_dataset)
    train, held_out = etl.split_train_held_out(dataset)

    for got, expected in ((train, expected_train), (held_out, expected_held_out)):
        # The original left None in object columns where Re/Rct were missing; Spark stored
        # them as nullable doubles, which is what the port produces directly.
        for col in ("Re", "Rct", "ambient_temperature"):
            expected[col] = expected[col].astype(float)
        assert list(got.columns) == list(expected.columns)
        assert_frame_equal(got, expected, check_exact=True)


def test_run_pipeline_accepts_a_mat_folder(tmp_path):
    """A folder of NASA .mat files is converted on the fly, then processed as usual."""
    from conftest import write_fake_mat

    mat_dir = tmp_path / "mat"
    for bid in ("B0005", "B0025", "B0030", "B0036", "B0042", "B0047", "B0056"):
        write_fake_mat(mat_dir / f"{bid}.mat", bid)
    out = tmp_path / "out"
    messages = []
    summary = etl.run_pipeline(str(mat_dir), str(out), log=messages.append)
    assert (out / "raw_csv" / "metadata.csv").is_file()
    assert (out / "train.csv").is_file() and (out / "held_out.csv").is_file()
    assert "holds NASA .mat files" in messages[0]
    assert summary["per_battery"]["B0056"]["kept"] == 2
    held_out = pd.read_csv(out / "held_out.csv")
    assert set(held_out["battery_id"]) == set(config.HELD_OUT_BATTERIES)
