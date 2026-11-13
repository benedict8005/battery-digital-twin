"""Errors raised when input data is malformed.

Every check in the data pipeline that rejects bad input raises
:class:`DataValidationError`, so callers (and the command line) can show one readable
message instead of a stack trace from deep inside NumPy or pandas. Messages name the
file and, where it applies, the cycle, column or row at fault.
"""

from collections.abc import Sequence

import pandas as pd


class DataValidationError(ValueError):
    """Input data is missing, truncated or inconsistent; the message says where."""


def check_processed_table(df: pd.DataFrame, path: object, required: Sequence[str], text: Sequence[str] = ()) -> None:
    """Check a processed per-cycle table against its schema.

    ``required`` columns must exist and be numeric; ``text`` columns must only exist. Missing (NaN) and infinite values are
    allowed because the loader drops those rows and reports them. Messages
    name ``path``, the column and the first offending row (as a CSV line number, where line 1
    is the header). Raises :class:`DataValidationError`.
    """
    missing = [c for c in [*text, *required] if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path} is missing required columns: {missing}")
    for column in required:
        values = df[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce")
            bad = values.notna() & numeric.isna()
            line = int(bad.to_numpy().argmax()) + 2
            raise DataValidationError(
                f"{path}: column '{column}' is not numeric (row {line}: {values.iloc[line - 2]!r})"
            )
