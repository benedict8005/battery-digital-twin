"""Logging setup for the command line: one ``battery_pinn`` logger, a level chosen by ``-v``/``-q``."""

import logging
import sys

__all__ = [
    "LOGGER_NAME",
    "configure_logging",
    "level_for",
]

LOGGER_NAME = "battery_pinn"


def level_for(verbose: bool = False, quiet: bool = False) -> int:
    """Return the logging level for the flags: DEBUG with ``verbose``, WARNING with ``quiet``, else INFO."""
    if verbose and quiet:
        raise ValueError("verbose and quiet cannot both be set")
    if verbose:
        return logging.DEBUG
    if quiet:
        return logging.WARNING
    return logging.INFO


def configure_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Send ``battery_pinn`` log records to stderr at the level the flags ask for.

    Calling it again replaces the handler it installed earlier, so repeated calls
    (as in tests, or several ``main`` calls in one process) never duplicate output.
    Results printed by the commands stay on stdout; only status messages go here.
    """
    logger = logging.getLogger(LOGGER_NAME)
    level = level_for(verbose, quiet)
    for handler in [h for h in logger.handlers if getattr(h, "_battery_pinn_cli", False)]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler._battery_pinn_cli = True  # type: ignore[attr-defined]
    fmt = "%(levelname)s %(name)s: %(message)s" if verbose else "%(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
