.PHONY: install test demo convert-mat prepare-data
install:
	pip install -e ".[dev]"
test:
	python -m pytest -q
demo:
	python -m battery_pinn demo --workdir .
prepare-data:
	python -m battery_pinn prepare-data --raw data/nasa_mat --out data/processed
convert-mat:
	python -m battery_pinn convert-mat --mat-dir data/nasa_mat --out data/raw
