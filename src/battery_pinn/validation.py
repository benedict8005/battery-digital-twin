"""Errors raised when input data is malformed.

Every check in the data pipeline that rejects bad input raises
:class:`DataValidationError`, so callers (and the command line) can show one readable
message instead of a stack trace from deep inside NumPy or pandas. Messages name the
file and, where it applies, the cycle, column or row at fault.
"""


class DataValidationError(ValueError):
    """Input data is missing, truncated or inconsistent; the message says where."""
