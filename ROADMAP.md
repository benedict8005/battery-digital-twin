# Roadmap

## Overall goal

Turn the battery PINN code from a research prototype into a well-tested, typed, documented and installable Python package. Its physics, data handling and command line should be checkable without training any model. Success means that linting, type checking, the full test suite and the documentation build all run in minutes on an ordinary CPU, that every existing test still passes, and that none of the work changes what the training or evaluation code computes.

## Ground rules

All work is plain Python in the existing package. No task trains, fine-tunes, evaluates or benchmarks a model, and none needs a GPU. Tests use tiny synthetic fixtures that run in seconds. The training and evaluation code is touched only where an interface has to change, and a test pins the old behaviour first.

Each task is a separate pull request with its own tests. Commands marked **to add** are placeholders for names chosen during the work. Run them from the folder that contains `pyproject.toml`, with the virtual environment active.

---

## 1. Code quality gates

Add `ruff` and `mypy` to the dev extras and fix what they report. Start with a permissive configuration, then tighten it module by module until the core modules (`config`, `data`, `etl`, `nasa_mat`, `interpolation`, `ocv`, `physics`) pass `mypy --strict`. Add type hints and short docstrings to every public function in those modules. Add a coverage report and record the current figure.

The test workflow already covers three operating systems, Python 3.10 and 3.12, and a run without PyTorch. Add a lint and type-check job beside it.

Done when `ruff check`, `ruff format --check` and `mypy` pass in CI, the coverage figure is in the README, and the test suite is otherwise unchanged.

```bash
python -m pytest -q                       # runs today
ruff check . && ruff format --check .     # to add
mypy src                                  # to add
python -m pytest --cov=battery_pinn -q    # to add
```

## 2. Independent physics check

The thermal PDE is currently only checked through the neural-network residual. Add a plain NumPy finite-difference solver for the same one-dimensional heat equation, with the same boundary conditions (including the optional Newton cooling). It is used in two places: tests that compare it with analytical cases (steady state, uniform heating, cooling to ambient), and a small `simulate` command that prints a temperature profile for a given current and ambient temperature.

Also test the PDE residual function against fixed analytic temperature fields, so a wrong sign or unit in the physics code is caught without any network.

Done when the solver agrees with the analytical cases within a stated tolerance, the residual tests pass, and `simulate` runs in under a second.

```bash
python -m pytest test/test_physics.py -q          # runs today
python -m pytest test/test_fdm.py -q              # to add
python -m battery_pinn simulate --ambient 25 ...  # to add
```

## 3. Data validation and reporting

Make the conversion and preparation of the NASA files safer. Add a schema check for the raw and processed tables, with clear error messages naming the file, the column and the row. Add an `inspect-data` command that prints a short report: rows and cycles per battery, ranges and missing values per column, and which cycles were dropped and why.

Add property-based tests (Hypothesis) and malformed-input tests for the `.mat` reader and the ETL: truncated files, missing columns, empty cycles and out-of-order timestamps. Each should fail with a readable error, never a stack trace from deep inside NumPy.

Done when every malformed case in the test list raises the intended error, the report matches hand-checked numbers on a synthetic dataset, and the ETL output is unchanged on the existing fixtures.

```bash
python -m pytest test/test_etl.py test/test_nasa_mat.py -q   # runs today
python -m pytest test/test_validation.py -q                   # to add
python -m battery_pinn inspect-data --data data/processed     # to add
```

## 4. Stable API and command line

Define the public Python API (`__all__` in each module, no reliance on private helpers) and ship a `py.typed` marker so users get type hints. Add a `--config` option that reads command-line settings from a YAML or TOML file, structured logging with a `--verbose` and `--quiet` option, `--json` output for the commands that print results, and consistent exit codes.

Done when each command documents its options in `--help`, a test checks every exit code, and a config file reproduces the same run as the equivalent flags.

```bash
python -m pytest test/test_cli.py -q                # runs today
python -m pytest test/test_cli_config.py -q         # to add
python -m battery_pinn abuse-test --config run.toml # to add
```

## 5. Packaging and documentation

Make the package installable from a built wheel. Complete the metadata in `pyproject.toml` (classifiers, URLs, optional extras), build the sdist and wheel in CI, and install the wheel in a clean environment to run the tests against it. Add a documentation site (Sphinx or MkDocs) with an API reference, a page that derives the PDE system and its units, and a tutorial based on the synthetic demo. Add `CONTRIBUTING.md` and issue templates, and tag a release with notes taken from the changelog.

Done when the documentation builds without warnings in CI, the wheel installs and passes the test suite in a fresh environment, and the release is tagged.

```bash
python -m build                       # to add
pip install dist/*.whl && python -m pytest -q   # to add
sphinx-build -W docs docs/_build      # to add
```

---

## Final check

On a clean checkout, the following must run as written and finish in minutes on a CPU:

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src
python -m pytest -q
python -m build
sphinx-build -W docs docs/_build
```
