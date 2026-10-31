"""Loading and cleaning of the processed per-cycle dataset, with per-battery coverage checks."""

from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from battery_pinn import config
from battery_pinn.config import REQUIRED_COLUMNS

BatteryReport = dict[str, dict[str, Any]]
"""Per-battery coverage record, as returned by :func:`battery_report`."""


class BatteryCoverageError(RuntimeError):
    """A designated battery is missing from a dataset or has no surviving cycles."""


def _cycles(df: pd.DataFrame) -> pd.Series:
    """Number of distinct cycles per battery (rows per battery when there is no cycle column)."""
    if "cycle_number" in df.columns:
        return cast("pd.Series", df.drop_duplicates(["battery_id", "cycle_number"])["battery_id"].value_counts())
    return cast("pd.Series", df["battery_id"].value_counts())


def battery_report(
    raw: pd.DataFrame, clean: pd.DataFrame, expected_batteries: Sequence[str] | None = None
) -> BatteryReport:
    """Per-battery cycles/rows before and after cleaning.

    ``expected_batteries`` (default: the batteries present in ``raw``) fixes which batteries
    are reported, so a battery that is absent altogether still appears with zeros.
    Returns ``{battery_id: {"present", "cycles_in_file", "cycles_kept", "cycles_dropped",
    "rows_in_file", "rows_kept"}}`` in the order of ``expected_batteries`` followed by any
    unexpected batteries found in the file.
    """
    present = list(pd.unique(raw["battery_id"].dropna()))
    if expected_batteries is None:
        expected_batteries = sorted(present)
    order = list(expected_batteries) + sorted(b for b in present if b not in set(expected_batteries))
    raw_cycles, clean_cycles = _cycles(raw), _cycles(clean)
    raw_rows, clean_rows = raw["battery_id"].value_counts(), clean["battery_id"].value_counts()
    report: BatteryReport = {}
    for bid in order:
        in_file, kept = int(raw_cycles.get(bid, 0)), int(clean_cycles.get(bid, 0))
        report[bid] = {
            "present": bid in present,
            "expected": bid in set(expected_batteries),
            "cycles_in_file": in_file,
            "cycles_kept": kept,
            "cycles_dropped": in_file - kept,
            "rows_in_file": int(raw_rows.get(bid, 0)),
            "rows_kept": int(clean_rows.get(bid, 0)),
        }
    return report


def coverage_problems(report: BatteryReport) -> list[str]:
    """Human-readable reasons why an expected battery is unusable (empty list when all is well)."""
    problems: list[str] = []
    for bid, r in report.items():
        if not r["expected"]:
            continue
        if not r["present"]:
            problems.append(f"{bid}: not present in the file")
        elif r["cycles_kept"] == 0:
            problems.append(
                f"{bid}: 0 of {r['cycles_in_file']} cycles survived cleaning "
                f"(all {r['rows_in_file']} rows had a missing/non-finite required column)"
            )
    return problems


def format_battery_report(
    report: BatteryReport,
    title: str = "Per-battery coverage",
    held_out_batteries: Sequence[str] | None = None,
) -> str:
    """Render a :func:`battery_report` as indented text, flagging held-out and lost batteries."""
    if held_out_batteries is None:
        held_out_batteries = config.HELD_OUT_BATTERIES
    lines = [f"--- {title} ---"]
    for bid, r in report.items():
        tag = " [HELD-OUT]" if bid in held_out_batteries else ""
        if not r["present"]:
            lines.append(f"  {bid}{tag}: MISSING from file")
            continue
        flag = ""
        if r["cycles_kept"] == 0:
            flag = "  <-- ZERO cycles survived!"
        elif r["cycles_dropped"]:
            flag = f"  ({r['cycles_dropped']} cycles dropped: missing/non-finite values)"
        extra = "" if r["expected"] else "  (not a designated battery)"
        lines.append(
            f"  {bid}{tag}: {r['cycles_kept']}/{r['cycles_in_file']} cycles kept, "
            f"{r['rows_kept']}/{r['rows_in_file']} rows{flag}{extra}"
        )
    return "\n".join(lines)


def load_dataset(
    path: str | PathLike[str],
    expected_batteries: Sequence[str] | None = None,
    required: bool = True,
    log: Callable[[str], object] | None = print,
) -> pd.DataFrame:
    """Read a processed dataset CSV and drop rows with missing critical values.

    Prints how many cycles/rows of every battery survived the cleaning. With
    ``expected_batteries`` the report covers exactly those batteries (plus any extras in the
    file), and -- when ``required`` -- a :class:`BatteryCoverageError` is raised if any of
    them is absent or left with zero surviving cycles. Without an expected list, every battery
    present in the file is expected, so a battery that loses all of its cycles still fails.
    """
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = raw.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=REQUIRED_COLUMNS)

    report = battery_report(raw, df, expected_batteries)
    if log is not None:
        log(format_battery_report(report, title=f"Per-battery coverage of {path}"))
    problems = coverage_problems(report)
    if problems:
        message = f"{path}: unusable designated batteries:\n  " + "\n  ".join(problems)
        if required:
            raise BatteryCoverageError(message)
        if log is not None:
            log("WARNING: " + message)

    if df.empty:
        raise ValueError(f"{path} has no usable rows after cleaning.")
    return cast("pd.DataFrame", df.reset_index(drop=True))


def representative_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average current / voltage / SoC against normalised time across all cycles."""
    profile = (
        df.sort_values("Time_normalized")
        .groupby("Time_normalized")
        .agg({"Current_measured": "mean", "Voltage_measured": "mean", "SoC": "mean"})
        .reset_index()
    )
    return cast("pd.DataFrame", profile)


def _column(df: pd.DataFrame, name: str) -> npt.NDArray[np.float64]:
    """One column as a plain float64 array."""
    return np.asarray(df[name].to_numpy(), dtype=np.float64)


def build_points_and_targets(
    df: pd.DataFrame,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Surface (x=1) network inputs and the scaled T_delta target for each row."""
    from battery_pinn.config import TEMP_SCALE_C

    points = np.column_stack(
        (
            np.ones(len(df)),
            _column(df, "SoH") / 100.0,
            _column(df, "Re") * 10.0,
            _column(df, "ambient_temperature"),
            _column(df, "Time_normalized"),
        )
    ).astype(np.float32)
    targets = (_column(df, "Temperature_measured") - _column(df, "ambient_temperature")) / TEMP_SCALE_C
    return points, targets.astype(np.float32)
