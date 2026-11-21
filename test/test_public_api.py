"""The public API is declared with ``__all__`` and the package ships a ``py.typed`` marker.

These checks read the source with ``ast`` so they need neither PyTorch nor DeepXDE.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "battery_pinn"

# Modules whose public surface is declared so far (extend as each module is done).
DECLARED = [
    "checkpoint",
    "config",
    "data",
    "device",
    "etl",
    "fdm",
    "identifiability",
    "inference",
    "interpolation",
    "nasa_mat",
    "ocv",
    "physics",
    "report",
    "synthetic",
    "validation",
]


def _parse(module):
    return ast.parse((PACKAGE / f"{module}.py").read_text())


def _declared_all(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return [elt.value for elt in node.value.elts]
    return None


def _top_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


@pytest.mark.parametrize("module", DECLARED)
def test_all_is_declared_and_names_exist(module):
    tree = _parse(module)
    exported = _declared_all(tree)
    assert exported is not None, f"{module} has no __all__"
    assert len(exported) == len(set(exported)), f"{module}.__all__ has duplicates"
    assert not [n for n in exported if n.startswith("_")], f"{module}.__all__ lists a private name"
    missing = set(exported) - _top_level_names(tree)
    assert not missing, f"{module}.__all__ names things it does not define: {sorted(missing)}"


@pytest.mark.parametrize("module", DECLARED)
def test_every_public_function_and_class_is_exported(module):
    tree = _parse(module)
    exported = set(_declared_all(tree) or [])
    public = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    }
    assert public <= exported, f"{module}: public but not in __all__: {sorted(public - exported)}"


def test_py_typed_marker_is_shipped():
    assert (PACKAGE / "py.typed").is_file()
    pyproject = (PACKAGE.parents[1] / "pyproject.toml").read_text()
    assert '[tool.setuptools.package-data]\nbattery_pinn = ["py.typed"]' in pyproject


def test_config_exports_every_public_constant():
    """``config`` is mostly constants, so check them as well as its functions."""
    tree = _parse("config")
    exported = set(_declared_all(tree) or [])
    public = {n for n in _top_level_names(tree) if not n.startswith("_") and n != "__all__"}
    assert public <= exported, f"config: public but not in __all__: {sorted(public - exported)}"
