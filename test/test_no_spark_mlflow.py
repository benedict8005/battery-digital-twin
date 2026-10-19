"""``src/battery_pinn`` must be completely free of Spark, MLflow and Databricks dependencies.

Three independent checks:

1. Static: no module under ``src/`` imports a forbidden package or references the Databricks
   notebook globals (``spark``, ``dbutils``, ``display``).
2. Dynamic: with an import hook that makes every forbidden package unimportable, the whole
   package imports and the data pipeline runs end to end.
3. Packaging: neither ``pyproject.toml`` nor the requirements files declare them.
"""
import ast
import glob
import importlib
import importlib.abc
import importlib.util
import os
import pkgutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src", "battery_pinn")
SRC_FILES = sorted(glob.glob(os.path.join(SRC_DIR, "*.py")))

FORBIDDEN_PACKAGES = ("pyspark", "mlflow", "databricks", "delta", "py4j", "dbutils")
FORBIDDEN_GLOBALS = ("spark", "dbutils", "display", "sqlContext")
HEAVY_OPTIONAL = ("torch", "deepxde")


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _top_level(name):
    return name.split(".")[0]


def _available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


@pytest.mark.parametrize("path", SRC_FILES, ids=os.path.basename)
def test_source_has_no_forbidden_imports_or_notebook_globals(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    offenders = sorted({m for m in _imports(tree) if _top_level(m) in FORBIDDEN_PACKAGES})
    assert not offenders, f"{os.path.basename(path)} imports {offenders}"
    names = sorted({
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in FORBIDDEN_GLOBALS
    })
    assert not names, f"{os.path.basename(path)} references notebook globals {names}"


class _Blocker(importlib.abc.MetaPathFinder):
    """Makes the forbidden packages unimportable and records any attempt."""

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if _top_level(fullname) in FORBIDDEN_PACKAGES:
            self.attempts.append(fullname)
            raise ImportError(f"{fullname} is blocked: src/battery_pinn must not depend on it")
        return None


@pytest.fixture
def blocked_imports():
    blocker = _Blocker()
    saved = {k: v for k, v in sys.modules.items()
             if _top_level(k) in FORBIDDEN_PACKAGES or _top_level(k) == "battery_pinn"}
    for k in saved:
        del sys.modules[k]
    sys.meta_path.insert(0, blocker)
    try:
        yield blocker
    finally:
        sys.meta_path.remove(blocker)
        for k in list(sys.modules):
            if _top_level(k) == "battery_pinn":
                del sys.modules[k]
        sys.modules.update(saved)


def test_blocker_actually_blocks(blocked_imports):
    for name in ("pyspark", "mlflow", "pyspark.sql"):
        with pytest.raises(ImportError, match="blocked"):
            importlib.import_module(name)


def test_every_module_imports_with_spark_and_mlflow_blocked(blocked_imports):
    heavy_available = all(_available(m) for m in HEAVY_OPTIONAL)
    package = importlib.import_module("battery_pinn")
    imported, skipped = [], []
    for info in pkgutil.iter_modules(package.__path__):
        path = os.path.join(SRC_DIR, info.name + ".py")
        with open(path, encoding="utf-8") as f:
            needs_heavy = any(_top_level(m) in HEAVY_OPTIONAL for m in _imports(ast.parse(f.read())))
        if needs_heavy and not heavy_available:
            skipped.append(info.name)
            continue
        importlib.import_module(f"battery_pinn.{info.name}")
        imported.append(info.name)
    assert {"config", "etl", "data", "synthetic", "cli"} <= set(imported)
    assert blocked_imports.attempts == []
    if skipped:
        pytest.skip(f"imported {imported}; {skipped} need torch/deepxde which are not installed")


def test_prepare_data_runs_with_spark_and_mlflow_blocked(blocked_imports, raw_dataset, tmp_path, capsys):
    from battery_pinn.cli import main

    out = tmp_path / "processed"
    main(["prepare-data", "--raw", raw_dataset, "--out", str(out)])
    assert (out / "train.csv").is_file()
    assert (out / "held_out.csv").is_file()
    assert blocked_imports.attempts == []
    assert "final_pinn_training_data" in capsys.readouterr().out


@pytest.mark.parametrize("filename", ["pyproject.toml", "requirements.txt", "requirements-dev.txt"])
def test_packaging_declares_no_spark_or_mlflow(filename):
    with open(os.path.join(ROOT, filename), encoding="utf-8") as f:
        text = f.read().lower()
    for forbidden in ("pyspark", "mlflow", "databricks"):
        assert forbidden not in text, f"{filename} mentions {forbidden}"
