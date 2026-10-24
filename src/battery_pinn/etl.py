"""Local raw-to-processed pipeline (pandas + NumPy).

A line-for-line port of the Databricks medallion ETL (``databricks/medallion_etl.py``, Spark
SQL) and feature-engineering stage (``databricks/feature_engineering.py``). The layers are:

* **bronze** -- ``metadata.csv`` as read from disk.
* **silver** -- discharge rows and impedance rows (excluded batteries removed) with the
  numpy-array-style ``start_time`` parsed into a timestamp and a per-battery cycle number.
* **gold** -- one row per discharge file, joined to the impedance measurement (``Re``,
  ``Rct``) taken at or before that discharge started.
* **processed tables** -- per cycle: truncation at the battery's cutoff voltage,
  coulomb-counting SoC, SoH from capacity fade, downsampling to a fixed number of bins;
  then a battery-level split into ``final_pinn_training_data`` and
  ``held_out_pinn_validation_data``.

Numerical details that the Spark version implied are reproduced on purpose: ``capacity``,
``Re`` and ``Rct`` are rounded to 32-bit floats (the gold table stored them as ``FLOAT``),
a discharge with no earlier impedance measurement gets ``NaN`` for ``Re``/``Rct``, and a
cycle whose capacity is missing is neither skipped nor repaired (its SoC/SoH are ``NaN``
and the training loader drops those rows later), exactly as before. SoH is evaluated in
float64 from those float32 inputs, so it agrees with the Delta tables to float32 precision
(bit-for-bit where the cluster's NumPy promoted the ratio to float64).

Two deliberate deviations, both documented in ``docs/CHANGELOG.md`` item 21:

* The Spark gold join left unspecified which impedance row wins when several precede a
  discharge (``ROW_NUMBER() ... ORDER BY d.start_datetime`` ranked the ties arbitrarily).
  This port always uses the most recent measurement at or before the discharge.
* The per-battery survival report counts kept *cycles*; the original compared kept rows
  (cycles x bins) with candidate cycles, so its "fewer than 20 %" flag never fired.
"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from battery_pinn import config
from battery_pinn.data import BatteryCoverageError

_NUMBER_RE = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?")
_TIME_COMPONENTS = ["y", "m", "d", "hr", "mi", "sec"]

# Aggregations applied to each downsampling bin (same columns and order as the original).
_BIN_AGGREGATIONS = {
    "Time": "mean",
    "Voltage_measured": "mean",
    "Current_measured": "mean",
    "Temperature_measured": "mean",
    "SoC": "mean",
}


# ----------------------------------------------------------------------------------------
# Bronze
# ----------------------------------------------------------------------------------------
def load_raw_metadata(raw_dir):
    """Read ``metadata.csv`` and normalise its column names.

    The Spark SQL matched columns case-insensitively (``Capacity`` in the file became
    ``capacity``), so the same is done here. Numeric columns are coerced the way a SQL
    ``CAST`` would: anything unparseable becomes NaN.
    """
    path = config.metadata_path(raw_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Expected the raw dataset folder to contain "
            f"{config.METADATA_FILENAME} and a {config.CYCLE_DATA_SUBDIR}/ folder of cycle CSVs."
        )
    df = pd.read_csv(path)

    by_lower = {}
    for col in df.columns:
        by_lower.setdefault(col.lower(), []).append(col)
    rename, missing = {}, []
    for canonical in config.RAW_METADATA_COLUMNS:
        matches = by_lower.get(canonical.lower(), [])
        if len(matches) > 1:
            raise ValueError(f"{path}: column {canonical!r} is ambiguous ({matches}).")
        if matches:
            rename[matches[0]] = canonical
        else:
            missing.append(canonical)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.rename(columns=rename)

    for col in ("ambient_temperature", "capacity", "Re", "Rct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("battery_id", "type", "filename"):
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


# ----------------------------------------------------------------------------------------
# Silver
# ----------------------------------------------------------------------------------------
def _seconds_to_whole_and_micros(sec):
    """Split a seconds value the way ``to_timestamp`` parsed the concatenated string.

    Spark cast the seconds to DOUBLE, rendered it back to text and parsed the fraction as
    a variable-length field, truncating to microseconds; doing it through the decimal text
    (rather than ``(sec - floor(sec)) * 1e6``) avoids binary round-off such as
    ``17.3 -> 299999 us``.
    """
    text = repr(float(sec))
    if "e" in text or "E" in text:
        text = f"{float(sec):.9f}"
    whole, _, frac = text.partition(".")
    return int(whole), int((frac + "000000")[:6])


def _timestamp(y, m, d, hr, mi, sec):
    """``to_timestamp(concat(y,'-',m,'-',d,' ',hr,':',mi,':',sec))``; NaT where Spark gave NULL."""
    if any(pd.isna(v) for v in (y, m, d, hr, mi, sec)):
        return pd.NaT
    try:
        whole, micros = _seconds_to_whole_and_micros(sec)
        return pd.Timestamp(
            year=int(y),
            month=int(m),
            day=int(d),
            hour=int(hr),
            minute=int(mi),
            second=whole,
            microsecond=micros,
        )
    except (ValueError, OverflowError):
        return pd.NaT


def parse_start_time(start_time):
    """Parse the numpy-array-style ``start_time`` strings (e.g. ``[2008. 4. 2. 15. 25. 41.]``).

    Returns a DataFrame aligned with ``start_time`` holding ``n_parts`` (number of numeric
    tokens found), the integer components ``y, m, d, hr, mi``, the float ``sec`` and the
    assembled ``start_datetime``. Plain and scientific notation are both accepted. Strings
    with fewer than six tokens get NaN components and a NaT timestamp, which is what the
    Spark array indexing and ``to_timestamp`` returned for them.
    """
    tokens = [_NUMBER_RE.findall(str(v)) if pd.notna(v) else [] for v in start_time]
    comps = np.full((len(tokens), 6), np.nan)
    for i, parts in enumerate(tokens):
        if len(parts) >= 6:
            comps[i] = [float(p) for p in parts[:6]]
    out = pd.DataFrame(comps, columns=_TIME_COMPONENTS, index=start_time.index)
    # CAST(CAST(part AS DOUBLE) AS INT) truncates toward zero; seconds stay a double.
    for col in _TIME_COMPONENTS[:-1]:
        out[col] = np.trunc(out[col])
    out.insert(0, "n_parts", [len(p) for p in tokens])
    out["start_datetime"] = pd.Series(
        [_timestamp(*row) for row in out[_TIME_COMPONENTS].itertuples(index=False)],
        index=start_time.index,
        dtype="datetime64[ns]",
    )
    return out


def _assign_cycle_numbers(df):
    """``ROW_NUMBER() OVER (PARTITION BY battery_id ORDER BY y, m, d, hr, mi, sec)``.

    NULL components sort first (Spark's ASC default); exact ties keep file order.
    """
    ordered = df.sort_values(["battery_id"] + _TIME_COMPONENTS, kind="mergesort", na_position="first")
    cycle_number = ordered.groupby("battery_id", sort=False).cumcount() + 1
    return df.assign(cycle_number=cycle_number.reindex(df.index).astype(int))


def _silver_base(bronze, excluded_batteries):
    df = bronze[~bronze["battery_id"].isin(excluded_batteries)]
    return df.join(parse_start_time(df["start_time"]))


def build_silver_discharge(bronze, excluded_batteries=None):
    """Discharge rows with ``start_datetime`` and a per-battery ``cycle_number``."""
    if excluded_batteries is None:
        excluded_batteries = config.EXCLUDED_BATTERIES
    df = _silver_base(bronze, excluded_batteries)
    df = df[df["type"] == "discharge"]
    df = _assign_cycle_numbers(df)
    cols = ["battery_id", "type", "filename", "ambient_temperature", "capacity", "start_datetime", "cycle_number"]
    return df[cols].reset_index(drop=True)


def build_silver_impedance(bronze, excluded_batteries=None):
    """Impedance rows (``Re``, ``Rct``) with ``start_datetime`` and a per-battery ``cycle_number``.

    Unlike the discharge layer, rows whose ``start_time`` has fewer than six numeric parts
    are dropped (``WHERE size(parts) >= 6`` in the original).
    """
    if excluded_batteries is None:
        excluded_batteries = config.EXCLUDED_BATTERIES
    df = _silver_base(bronze, excluded_batteries)
    df = df[(df["n_parts"] >= 6) & (df["type"] == "impedance")]
    df = _assign_cycle_numbers(df)
    cols = ["battery_id", "Re", "Rct", "start_datetime", "cycle_number"]
    return df[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Gold
# ----------------------------------------------------------------------------------------
def _latest_impedance_at_or_before(discharge, impedance):
    """For each discharge row, the impedance row with the greatest ``start_datetime`` that is
    ``<=`` the discharge start (same battery). Rows with NaT never match, as in SQL."""
    re_out = np.full(len(discharge), np.nan)
    rct_out = np.full(len(discharge), np.nan)
    imp_by_battery = {
        bid: grp.dropna(subset=["start_datetime"]).sort_values(["start_datetime", "cycle_number"], kind="mergesort")
        for bid, grp in impedance.groupby("battery_id", sort=False)
    }
    for pos, (bid, ts) in enumerate(zip(discharge["battery_id"], discharge["start_datetime"])):
        grp = imp_by_battery.get(bid)
        if grp is None or grp.empty or pd.isna(ts):
            continue
        k = np.searchsorted(grp["start_datetime"].values, ts.to_datetime64(), side="right") - 1
        if k >= 0:
            re_out[pos] = grp["Re"].iloc[k]
            rct_out[pos] = grp["Rct"].iloc[k]
    return re_out, rct_out


def build_gold(silver_discharge, silver_impedance):
    """One row per discharge file with the impedance (``Re``, ``Rct``) in force at that time.

    ``capacity``, ``Re`` and ``Rct`` are rounded to float32 because the gold table declared
    them ``FLOAT``, so ``Re`` and the SoH inputs carry the same rounding as the Delta table.
    """
    gold = silver_discharge.copy()
    gold["Re"], gold["Rct"] = _latest_impedance_at_or_before(gold, silver_impedance)
    for col in ("capacity", "Re", "Rct"):
        gold[col] = pd.to_numeric(gold[col], errors="coerce").astype(np.float32)
    # QUALIFY ROW_NUMBER() OVER (PARTITION BY d.filename ORDER BY d.start_datetime ASC) = 1
    gold = gold.sort_values("start_datetime", kind="mergesort", na_position="first")
    gold = gold.drop_duplicates("filename", keep="first")
    gold = gold.sort_values(["battery_id", "cycle_number"], kind="mergesort")
    cols = [
        "battery_id",
        "type",
        "filename",
        "ambient_temperature",
        "start_datetime",
        "cycle_number",
        "capacity",
        "Re",
        "Rct",
    ]
    return gold[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------------------
def load_cycle_file(path):
    """Read one raw per-cycle CSV, checking it has the columns the pipeline uses."""
    df = pd.read_csv(path)
    missing = [c for c in config.RAW_CYCLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.reset_index(drop=True)


def truncate_at_cutoff(cycle, cutoff_voltage):
    """Keep the samples before the first one whose measured voltage is below the cutoff."""
    cutoff_idx = cycle[cycle["Voltage_measured"] < cutoff_voltage].index.min()
    return cycle if pd.isna(cutoff_idx) else cycle.iloc[:cutoff_idx].copy()


def add_state_of_charge(cycle, capacity_ah):
    """Coulomb-counting SoC (%) from the measured current; starts at 100 %."""
    cycle["Time_diff_hr"] = cycle["Time"].diff().fillna(0) / 3600
    cycle["Delta_Q"] = abs(cycle["Current_measured"] * cycle["Time_diff_hr"])
    cycle["Cumulative_Q"] = cycle["Delta_Q"].cumsum()
    cycle["SoC"] = 100 * (1 - cycle["Cumulative_Q"] / capacity_ah)
    return cycle


def downsample_cycle(cycle, num_bins=None):
    """Average each measured column over ``num_bins`` equal-width slices of the row index."""
    if num_bins is None:
        num_bins = config.BINS_PER_CYCLE
    cycle["bin"] = pd.cut(cycle.index, bins=num_bins, labels=False)
    return cycle.groupby("bin").agg(_BIN_AGGREGATIONS).reset_index()


def process_cycle(cycle, meta, max_capacity, cutoff_voltage, min_samples=None, capacity_floor_ah=None, num_bins=None):
    """Turn one raw discharge cycle into its downsampled, labelled rows.

    ``meta`` is the cycle's gold-table row (``battery_id``, ``cycle_number``, ``capacity``,
    ``Re``, ``Rct``, ``ambient_temperature``). ``capacity_floor_ah`` defaults to
    :func:`battery_pinn.config.capacity_floor` for ``max_capacity``. Returns
    ``(frame, None)`` when the cycle is kept or ``(None, reason)`` with ``reason`` in
    ``{"too_short", "low_capacity"}``.
    """
    if min_samples is None:
        min_samples = config.MIN_SAMPLES_PER_CYCLE
    if capacity_floor_ah is None:
        capacity_floor_ah = config.capacity_floor(max_capacity)

    truncated = truncate_at_cutoff(cycle, cutoff_voltage)
    if len(truncated) < min_samples:
        return None, "too_short"

    db_capacity = float(meta["capacity"])
    if db_capacity <= capacity_floor_ah:
        return None, "low_capacity"

    truncated = add_state_of_charge(truncated, db_capacity)
    agg = downsample_cycle(truncated, num_bins)
    agg["battery_id"] = meta["battery_id"]
    agg["cycle_number"] = meta["cycle_number"]
    agg["SoH"] = float(db_capacity / float(max_capacity) * 100)
    # Missing values are NaN in a float column (the original assigned None, which Spark then
    # stored as a nullable double -- same table content, cleaner in-memory dtype).
    for col in ("Re", "Rct", "ambient_temperature"):
        agg[col] = float(meta[col]) if pd.notna(meta[col]) else np.nan
    agg["Time_normalized"] = agg["Time"] / agg["Time"].max()
    return agg, None


SKIP_REASONS = ("file_not_found", "too_short", "low_capacity")


def build_feature_table(
    gold,
    raw_dir,
    target_batteries=None,
    held_out_batteries=None,
    cutoff_map=None,
    default_cutoff=None,
    min_capacity=config.UNSET,
    min_capacity_fraction=config.UNSET,
    **cycle_kwargs,
):
    """Run the per-cycle feature engineering over every gold row of the target batteries.

    Returns ``(dataset, summary)``. ``dataset`` is the concatenation of all kept cycles (an
    empty frame with the processed columns if none survived). ``summary`` records the
    candidate count, the skip reasons, and for *every* designated battery -- present in the
    metadata or not -- its role, capacity floor, candidate/kept counts and dropped counts by
    reason. ``min_capacity`` / ``min_capacity_fraction`` override the config constants
    behind :func:`battery_pinn.config.capacity_floor`.
    """
    if target_batteries is None:
        target_batteries = config.TARGET_BATTERIES
    if held_out_batteries is None:
        held_out_batteries = config.HELD_OUT_BATTERIES
    if cutoff_map is None:
        cutoff_map = config.CUTOFF_VOLTAGE_MAP
    if default_cutoff is None:
        default_cutoff = config.DEFAULT_CUTOFF_VOLTAGE

    discharge_metadata = gold[gold["battery_id"].isin(target_batteries)]
    max_caps = gold.groupby("battery_id")["capacity"].max()
    floors = {
        bid: config.capacity_floor(max_caps.get(bid, np.nan), min_capacity, min_capacity_fraction)
        for bid in target_batteries
    }

    processed, skip_counts = [], dict.fromkeys(SKIP_REASONS, 0)
    kept = Counter()
    dropped = {bid: dict.fromkeys(SKIP_REASONS, 0) for bid in target_batteries}
    for _, row in discharge_metadata.iterrows():
        bid = row["battery_id"]
        path = config.cycle_file_path(raw_dir, row["filename"])
        if not os.path.exists(path):
            reason, frame = "file_not_found", None
        else:
            frame, reason = process_cycle(
                load_cycle_file(path),
                row,
                max_caps.loc[bid],
                cutoff_map.get(bid, default_cutoff),
                capacity_floor_ah=floors[bid],
                **cycle_kwargs,
            )
        if frame is None:
            skip_counts[reason] += 1
            dropped[bid][reason] += 1
        else:
            processed.append(frame)
            kept[bid] += 1

    if processed:
        dataset = pd.concat(processed, ignore_index=True)[config.PROCESSED_COLUMNS]
    else:
        dataset = pd.DataFrame(columns=config.PROCESSED_COLUMNS)

    candidates = discharge_metadata["battery_id"].value_counts()
    summary = {
        "total_candidates": int(len(discharge_metadata)),
        "total_kept": len(processed),
        "skip_counts": skip_counts,
        "per_battery": {
            bid: {
                "role": "held-out" if bid in held_out_batteries else "train",
                "present": bid in candidates.index,
                "capacity_floor_ah": floors[bid],
                "max_capacity_ah": float(max_caps.get(bid, np.nan)),
                "candidates": int(candidates.get(bid, 0)),
                "kept": kept[bid],
                "dropped": dropped[bid],
            }
            for bid in target_batteries
        },
    }
    return dataset, summary


def coverage_problems(summary):
    """Why a held-out battery is unusable: absent from the metadata or zero surviving cycles."""
    problems = []
    for bid, r in summary["per_battery"].items():
        if r["role"] != "held-out":
            continue
        if not r["present"]:
            problems.append(f"{bid}: not present in the raw metadata")
        elif r["kept"] == 0:
            reasons = ", ".join(f"{n} {why}" for why, n in r["dropped"].items() if n)
            problems.append(
                f"{bid}: 0 of {r['candidates']} cycles survived ({reasons}; "
                f"capacity floor {r['capacity_floor_ah']:.3f} Ah, best cycle "
                f"{r['max_capacity_ah']:.3f} Ah)"
            )
    return problems


def split_train_held_out(dataset, held_out_batteries=None):
    """Battery-level split: a held-out battery never contributes rows to the training set."""
    if held_out_batteries is None:
        held_out_batteries = config.HELD_OUT_BATTERIES
    mask = dataset["battery_id"].isin(held_out_batteries)
    return dataset[~mask].reset_index(drop=True), dataset[mask].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Reporting and the end-to-end driver
# ----------------------------------------------------------------------------------------
def format_summary(summary):
    """The cycle-processing summary and the per-battery survival report for every designated battery."""
    total, kept, skips = summary["total_candidates"], summary["total_kept"], summary["skip_counts"]
    lines = [
        "--- Cycle processing summary ---",
        f"Total candidate cycles:   {total}",
        f"Kept:                     {kept} ({100 * kept / max(total, 1):.1f}%)",
        f"Skipped (file not found): {skips['file_not_found']}",
        f"Skipped (too short):      {skips['too_short']}",
        f"Skipped (low capacity):   {skips['low_capacity']}",
    ]
    if sum(skips.values()) > 0.5 * total:
        lines.append(
            "WARNING: over half of all cycles were skipped -- check CUTOFF_VOLTAGE_MAP / "
            "MIN_VALID_CAPACITY_AH / MIN_CAPACITY_FRACTION_OF_MAX in battery_pinn/config.py."
        )

    lines.append("--- Per-battery cycle survival (every designated battery) ---")
    missing, zero_kept = [], []
    for bid, r in summary["per_battery"].items():
        tag = " [HELD-OUT]" if r["role"] == "held-out" else ""
        if not r["present"]:
            missing.append(bid)
            lines.append(f"  {bid}{tag}: MISSING from raw metadata")
            continue
        dropped = ", ".join(f"{n} {why}" for why, n in r["dropped"].items() if n) or "none"
        flag = ""
        if r["kept"] == 0:
            flag = "  <-- ZERO cycles survived!"
            zero_kept.append(bid)
        elif r["kept"] < 0.2 * r["candidates"]:
            flag = "  <-- fewer than 20% of cycles survived"
        lines.append(
            f"  {bid}{tag}: {r['kept']}/{r['candidates']} cycles kept, dropped: {dropped} "
            f"(capacity floor {r['capacity_floor_ah']:.3f} Ah, best cycle "
            f"{r['max_capacity_ah']:.3f} Ah){flag}"
        )

    if missing:
        lines.append(f"WARNING: {len(missing)} designated batteries are absent from the raw metadata: {missing}")
    if zero_kept:
        lines.append(f"WARNING: {len(zero_kept)} batteries had ZERO surviving cycles: {zero_kept}")
    problems = coverage_problems(summary)
    if problems:
        lines.append("ERROR: held-out batteries without validation data (the pipeline stops here):")
        lines.extend(f"  {p}" for p in problems)
    return "\n".join(lines)


def run_pipeline(raw_dir, out_dir, log=print, min_capacity_fraction=config.UNSET, allow_missing_held_out=False):
    """Raw NASA folder -> processed train / held-out CSVs. Returns the processing summary.

    ``raw_dir`` is either the per-cycle CSV layout (``metadata.csv`` + ``data/``) or a folder
    of NASA's ``B00xx.mat`` files, which is converted first. Writes ``train.csv``,
    ``held_out.csv`` and ``<table>.csv`` for the silver and gold tables into ``out_dir``.
    ``min_capacity_fraction`` overrides ``config.MIN_CAPACITY_FRACTION_OF_MAX`` (``None``
    reproduces the legacy absolute 1.4 Ah floor). A held-out battery without data is a hard
    error unless ``allow_missing_held_out`` is set, which downgrades it to a warning so the
    legacy tables can still be produced for comparison runs.
    """
    if config.is_mat_folder(raw_dir):
        from battery_pinn.nasa_mat import convert_mat_dir

        csv_dir = os.path.join(out_dir, config.MAT_CONVERSION_SUBDIR)
        log(f"{raw_dir} holds NASA .mat files; converting them to the per-cycle CSV layout in {csv_dir}")
        convert_mat_dir(raw_dir, csv_dir, log=log)
        raw_dir = csv_dir
    bronze = load_raw_metadata(raw_dir)
    log(f"Read {len(bronze)} metadata rows from {config.metadata_path(raw_dir)}")

    silver_discharge = build_silver_discharge(bronze)
    silver_impedance = build_silver_impedance(bronze)
    gold = build_gold(silver_discharge, silver_impedance)
    log(f"{config.GOLD_TABLE}: {len(gold)} discharge cycles across {gold['battery_id'].nunique()} batteries")

    dataset, summary = build_feature_table(gold, raw_dir, min_capacity_fraction=min_capacity_fraction)
    log(format_summary(summary))
    if dataset.empty:
        raise RuntimeError("No cycles survived processing -- check the skip-reason counts above.")
    problems = coverage_problems(summary)
    if problems and allow_missing_held_out:
        log(
            "WARNING: --allow-missing-held-out set; writing tables although held-out batteries lack data:\n  "
            + "\n  ".join(problems)
        )
    elif problems:
        raise BatteryCoverageError(
            "Held-out batteries without validation data -- nothing was written:\n  "
            + "\n  ".join(problems)
            + "\nCheck CUTOFF_VOLTAGE_MAP, MIN_VALID_CAPACITY_AH and MIN_CAPACITY_FRACTION_OF_MAX "
            "in battery_pinn/config.py, or the raw dataset folder."
        )

    train, held_out = split_train_held_out(dataset)

    os.makedirs(out_dir, exist_ok=True)
    outputs = {
        config.SILVER_DISCHARGE_TABLE: silver_discharge,
        config.SILVER_IMPEDANCE_TABLE: silver_impedance,
        config.GOLD_TABLE: gold,
        config.TRAIN_TABLE: train,
        config.HELD_OUT_TABLE: held_out,
    }
    for table, frame in outputs.items():
        frame.to_csv(config.table_path(out_dir, table), index=False)

    log(
        f"'{config.TRAIN_TABLE}' created: {train['battery_id'].nunique()} batteries, "
        f"{len(train)} rows -> {config.table_path(out_dir, config.TRAIN_TABLE)}"
    )
    log(
        f"'{config.HELD_OUT_TABLE}' created: {held_out['battery_id'].nunique()} batteries, "
        f"{len(held_out)} rows -> {config.table_path(out_dir, config.HELD_OUT_TABLE)}"
    )
    summary["outputs"] = {t: config.table_path(out_dir, t) for t in outputs}
    return summary
