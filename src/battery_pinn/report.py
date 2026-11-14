"""Summary report of a processed dataset table, behind the ``inspect-data`` command.

The report lists rows and cycles per battery, the range and the number of missing
(NaN or infinite) values of every numeric column, and which cycles the loader would drop
because a required column is missing, with the reason.
"""

from os import PathLike
from typing import Any

import numpy as np
import pandas as pd

from battery_pinn import config
from battery_pinn.data import battery_report
from battery_pinn.validation import check_processed_table

_TEXT_COLUMNS = ("battery_id",)


def column_report(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per numeric column: ``rows``, ``missing`` (NaN or infinite), ``min`` and ``max``.

    The range covers the finite values only and is ``None`` when there are none.
    """
    report: dict[str, dict[str, Any]] = {}
    for column in df.columns:
        if column in _TEXT_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        report[column] = {
            "rows": int(len(values)),
            "missing": int(len(values) - len(finite)),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    return report


def dropped_cycles(df: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    """Cycles that have a missing required value, as ``{battery_id: {column: [cycle, ...]}}``.

    A cycle is listed under every required column that is missing (NaN or infinite) in at
    least one of its rows; the loader drops those rows. Cycle numbers are sorted.
    """
    if "cycle_number" not in df.columns:
        return {}
    out: dict[str, dict[str, list[int]]] = {}
    for column in config.REQUIRED_COLUMNS:
        if column not in df.columns or column == "battery_id":
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        bad = df[~np.isfinite(values.to_numpy(dtype=np.float64))]
        for bid, cycle in bad[["battery_id", "cycle_number"]].drop_duplicates().itertuples(index=False):
            out.setdefault(str(bid), {}).setdefault(column, []).append(int(cycle))
    return {bid: {c: sorted(v) for c, v in cols.items()} for bid, cols in sorted(out.items())}


def inspect_table(df: pd.DataFrame) -> dict[str, Any]:
    """Build the report for an already-validated processed table."""
    required = [c for c in config.REQUIRED_COLUMNS if c in df.columns]
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    return {
        "rows": int(len(df)),
        "batteries": battery_report(df, clean),
        "columns": column_report(df),
        "dropped": dropped_cycles(df),
    }


def inspect_file(path: str | PathLike[str]) -> dict[str, Any]:
    """Read a processed CSV, check its schema and return :func:`inspect_table` of it.

    Raises :class:`~battery_pinn.validation.DataValidationError` for a missing or
    non-numeric required column, naming the file, column and row.
    """
    df = pd.read_csv(path)
    check_processed_table(df, path, [c for c in config.REQUIRED_COLUMNS if c != "battery_id"], text=["battery_id"])
    return inspect_table(df)


def format_report(report: dict[str, Any], title: str = "Data report") -> str:
    """Render :func:`inspect_table` output as plain text."""
    lines = [f"--- {title}: {report['rows']} rows ---", "Batteries:"]
    for bid, r in report["batteries"].items():
        lines.append(
            f"  {bid}: {r['cycles_kept']}/{r['cycles_in_file']} cycles kept, {r['rows_kept']}/{r['rows_in_file']} rows"
        )
    lines.append("Columns (range over finite values):")
    for name, c in report["columns"].items():
        span = "no finite values" if c["min"] is None else f"{c['min']:.6g} .. {c['max']:.6g}"
        lines.append(f"  {name}: {span}, {c['missing']} missing")
    if report["dropped"]:
        lines.append("Cycles dropped for missing or non-finite values:")
        for bid, cols in report["dropped"].items():
            for column, cycles in cols.items():
                lines.append(f"  {bid}: cycles {cycles} (column '{column}')")
    else:
        lines.append("Cycles dropped: none")
    return "\n".join(lines)
