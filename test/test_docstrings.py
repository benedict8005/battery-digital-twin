"""Every public function, class and method in the core modules has a docstring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_MODULES = ["config", "data", "etl", "nasa_mat", "interpolation", "ocv", "physics"]
SRC = Path(__file__).resolve().parents[1] / "src" / "battery_pinn"


def _public_defs(tree: ast.Module):
    """Yield public module-level functions and classes, and public methods of those classes.

    Functions nested inside other functions are implementation details and are skipped.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            yield node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                        yield child


@pytest.mark.parametrize("module", CORE_MODULES)
def test_module_and_public_definitions_have_docstrings(module: str) -> None:
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{module}.py has no module docstring"
    missing = [f"{module}.{n.name} (line {n.lineno})" for n in _public_defs(tree) if not ast.get_docstring(n)]
    assert not missing, "missing docstrings: " + ", ".join(missing)
