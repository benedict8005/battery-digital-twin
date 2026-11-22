"""Read command-line settings from a TOML or YAML file.

The file holds one table per command, named as on the command line, whose keys are the long option
names (``n-t`` and ``n_t`` both work)::

    [simulate]
    current = 3.0
    newton-cooling = true

Values from the file only replace the defaults of that command; a flag given on the command line wins.
"""

import argparse
import os
from typing import Any

from battery_pinn.validation import DataValidationError

__all__ = [
    "apply_config",
    "load_config_file",
]


def load_config_file(path: str) -> dict[str, dict[str, Any]]:
    """Read ``path`` (``.toml``, ``.yaml`` or ``.yml``) and return ``{command: {option: value}}``.

    Raises ``DataValidationError`` naming the file when it is missing, unreadable, of an unknown type or
    not shaped as one table per command.
    """
    if not os.path.isfile(path):
        raise DataValidationError(f"{path}: config file not found")
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".toml":
            try:
                import tomllib
            except ModuleNotFoundError:  # Python 3.10
                import tomli as tomllib  # type: ignore[no-redef]
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise DataValidationError(
                    f"{path}: reading YAML needs the PyYAML package (pip install pyyaml)"
                ) from exc
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if raw is None:
                raw = {}
        else:
            raise DataValidationError(f"{path}: unsupported config type {ext or '(none)'}; use .toml, .yaml or .yml")
    except DataValidationError:
        raise
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"{path}: cannot parse config file: {exc}") from exc
    except Exception as exc:  # YAMLError and TOMLDecodeError subclasses vary by library
        raise DataValidationError(f"{path}: cannot parse config file: {exc}") from exc
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        raise DataValidationError(f"{path}: expected one table per command, e.g. [simulate]")
    return {str(k): {str(o): v for o, v in table.items()} for k, table in raw.items()}


def apply_config(parser: argparse.ArgumentParser, path: str) -> None:
    """Make the settings in ``path`` the defaults of the matching commands of ``parser``.

    An unknown command or option raises ``DataValidationError`` naming the file and the key. An option
    that is required on the command line stops being required once the file supplies it.
    """
    tables = load_config_file(path)
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for command, table in tables.items():
        if command not in subparsers.choices:
            known = ", ".join(sorted(subparsers.choices))
            raise DataValidationError(f"{path}: unknown command [{command}]; known commands: {known}")
        sub = subparsers.choices[command]
        by_dest = {a.dest: a for a in sub._actions if a.dest not in ("help", "config")}
        updates: dict[str, Any] = {}
        for key, value in table.items():
            dest = key.lstrip("-").replace("-", "_")
            if dest not in by_dest:
                raise DataValidationError(f"{path}: [{command}] has unknown option '{key}'")
            action = by_dest[dest]
            if action.choices is not None and value not in action.choices:
                raise DataValidationError(
                    f"{path}: [{command}] {key} = {value!r} is not one of {sorted(action.choices)}"
                )
            if action.required:
                action.required = False
            updates[dest] = value
        sub.set_defaults(**updates)
