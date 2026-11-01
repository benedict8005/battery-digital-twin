"""``scripts/make_results_table.py``: checkpoint metrics, per-battery errors and verdict spread -> RESULTS.md."""

import importlib.util
import json
import math
import os

import numpy as np
import pytest

from battery_pinn import config, synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "make_results_table.py")


@pytest.fixture(scope="module")
def mrt():
    spec = importlib.util.spec_from_file_location("make_results_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(seed, train_mae, held_out_mae, verdict, a_sei=1e22, per_ho=None, per_tr=None, init=None, losses=None):
    return {
        "seed": seed,
        "checkpoint": f"battery_pinn_model_seed{seed}.pt",
        "A_sei": a_sei,
        "log10_A_sei": math.log10(a_sei),
        "log10_A_sei_init": init,
        "final_losses": losses or {},
        "train_mae_c": train_mae,
        "train_rmse_c": train_mae + 0.5,
        "held_out_mae_c": held_out_mae,
        "held_out_rmse_c": held_out_mae + 0.5,
        "per_battery_train_mae_c": per_tr or {"B0005": train_mae, "B0006": train_mae + 1},
        "per_battery_held_out_mae_c": per_ho or {"B0025": held_out_mae, "B0056": held_out_mae + 2},
        "per_ambient_train_mae_c": {"24 °C": train_mae, "43 °C": train_mae + 2},
        "per_ambient_held_out_mae_c": {"4 °C": held_out_mae - 1, "24 °C": held_out_mae},
        "verdict": verdict,
        "criterion": {"SAFE": "none", "WARNING": "core_temp_warning", "DANGER": "sei_below_threshold"}[verdict],
        "core_temp_max_c": 55.0,
        "sei_min": 0.9,
        "threshold": 0.8,
        "training": {},
    }


def _run(name, seeds, held_out_batteries=("B0025", "B0056")):
    return {
        "name": name,
        "artifacts_dir": f"/runs/{name}/artifacts",
        "data_dir": f"/runs/{name}/data",
        "train": {
            "batteries": ["B0005", "B0006"],
            "cycles": 10,
            "rows": 500,
            "rows_in_file": 500,
            "cycles_dropped_on_load": 0,
        },
        "held_out": {
            "batteries": list(held_out_batteries),
            "cycles": 4,
            "rows": 200,
            "rows_in_file": 250,
            "cycles_dropped_on_load": 1,
        },
        "seeds": seeds,
    }


# ----------------------------------------------------------------------------------------
# Pure helpers (no torch needed)
# ----------------------------------------------------------------------------------------
def test_find_checkpoints_sorted_by_seed(mrt, tmp_path):
    for name in (
        "battery_pinn_model_seed10.pt",
        "battery_pinn_model_seed2.pt",
        "battery_pinn_model_seed0.pt",
        "global_pinn_model_seed0.pt",
        "notes.txt",
    ):
        (tmp_path / name).write_bytes(b"")
    found = mrt.find_checkpoints(str(tmp_path))
    assert list(found) == [0, 2, 10]
    assert found[2].endswith("battery_pinn_model_seed2.pt")
    assert mrt.find_checkpoints(str(tmp_path / "nope")) == {}


def test_mean_std_and_verdict_spread(mrt):
    assert mrt.mean_std([2.0, 4.0]) == (3.0, 1.0)
    assert mrt.mean_std([5.0]) == (5.0, 0.0)
    assert mrt.mean_std([]) == (None, None)
    assert mrt.mean_std([1.0, float("nan"), 3.0]) == (2.0, 1.0)
    seeds = [_seed(0, 5, 6, "SAFE"), _seed(1, 5, 6, "DANGER"), _seed(2, 5, 6, "SAFE")]
    assert mrt.verdict_spread(seeds) == {"DANGER": 1, "WARNING": 0, "SAFE": 2}


def test_aggregate_metrics_gap_and_per_battery(mrt):
    run = _run("r", [_seed(0, 4.0, 6.0, "SAFE", a_sei=1e20), _seed(1, 6.0, 10.0, "DANGER", a_sei=1e22)])
    agg = mrt.aggregate(run)
    assert agg["n_seeds"] == 2
    assert agg["metrics"]["train_mae_c"] == (5.0, 1.0)
    assert agg["metrics"]["held_out_mae_c"] == (8.0, 2.0)
    assert agg["metrics"]["generalisation_gap_c"] == (3.0, 1.0)
    assert agg["metrics"]["log10_A_sei"] == pytest.approx((21.0, 1.0))
    assert agg["verdicts"] == {"DANGER": 1, "WARNING": 0, "SAFE": 1}
    assert agg["verdict_consistent"] is False
    assert agg["per_battery_held_out_mae_c"]["B0025"] == (8.0, 2.0)
    assert agg["per_battery_held_out_mae_c"]["B0056"] == (10.0, 2.0)
    assert agg["per_battery_train_mae_c"]["B0006"] == (6.0, 1.0)


def test_aggregate_handles_battery_present_in_only_some_seeds(mrt):
    run = _run(
        "r",
        [
            _seed(0, 4.0, 6.0, "SAFE", per_ho={"B0025": 1.0}),
            _seed(1, 4.0, 6.0, "SAFE", per_ho={"B0025": 3.0, "B0056": 9.0}),
        ],
    )
    per = mrt.aggregate(run)["per_battery_held_out_mae_c"]
    assert per["B0025"] == (2.0, 1.0) and per["B0056"] == (9.0, 0.0)


def test_render_markdown_has_every_section_and_compares_runs(mrt):
    legacy = _run("legacy", [_seed(0, 5.0, 7.0, "SAFE"), _seed(1, 5.5, 7.5, "DANGER")], held_out_batteries=("B0025",))
    legacy["seeds"][0]["per_battery_held_out_mae_c"] = {"B0025": 7.0}
    legacy["seeds"][1]["per_battery_held_out_mae_c"] = {"B0025": 7.5}
    dynamic = _run("dynamic", [_seed(0, 4.0, 6.0, "SAFE"), _seed(1, 4.0, 6.0, "SAFE")])
    text = mrt.render_markdown([legacy, dynamic], title="T", protocol="P", generated="now")

    for heading in (
        "# T",
        "## Protocol",
        "## Runs",
        "## Summary across seeds",
        "## Per-seed checkpoint metrics",
        "## Per-battery held-out MAE",
        "## Per-battery training MAE",
        "## Abuse-test verdict spread",
    ):
        assert heading in text
    assert "Generated now" in text and "\nP\n" in text
    # Runs table and the coverage notes.
    assert "| legacy | 2 | 10 | 500 | 1 | 4 | 200 | 0, 1 | `/runs/legacy/artifacts` |" in text
    assert "**legacy**: held-out table lacks designated batteries ['B0030', 'B0036', 'B0042', 'B0047', 'B0056']" in text
    assert "**dynamic**: 0 training and 1 held-out cycles are dropped on load" in text
    # Summary row: mean ± std, gap, verdict spread with the inconsistency flag.
    assert "| legacy | 2 | 22.00 ± 0.00 | 5.25 ± 0.25 | 5.75 ± 0.25 | 7.25 ± 0.25 | 7.75 ± 0.25 | 2.00 ± 0.00 |" in text
    assert (
        "| legacy | 1 | 1.00e+22 | 5.50 | 6.00 | 7.50 | 8.00 | 55.0 | 0.900 | 0.800 | DANGER | sei_below_threshold |"
        in text
    )
    assert "1 DANGER / 1 SAFE **(inconsistent)**" in text
    assert "| dynamic | 2 |" in text and "2 SAFE |" in text
    # Per-battery held-out table: B0056 absent from the legacy run.
    assert "| B0056 | -- (absent) | 8.00 ± 0.00 |" in text
    assert "| B0025 | 7.25 ± 0.25 | 6.00 ± 0.00 |" in text
    # Training table marks held-out batteries only if they appear there (they do not here).
    assert "| B0005 | 5.25 ± 0.25 | 4.00 ± 0.00 |" in text
    # Verdict spread table.
    assert "| legacy | 2 | 1 | 0 | 1 | **no** |" in text
    assert "| dynamic | 2 | 0 | 0 | 2 | yes |" in text


def test_aggregate_identifiability_and_loss_terms(mrt):
    losses = {"pde_thermal": 1.0, "pde_sei": 6.0, "ic_T": 1.0, "ic_c": 1.0, "observe_T": 1.0}
    run = _run(
        "r",
        [
            _seed(0, 4, 6, "SAFE", a_sei=10**22.1, init=21.0, losses=losses),
            _seed(1, 4, 6, "SAFE", a_sei=10**22.0, init=22.0, losses=losses),
            _seed(2, 4, 6, "SAFE", a_sei=10**22.2, init=23.0, losses=losses),
        ],
    )
    agg = mrt.aggregate(run)
    assert agg["identifiability"]["status"] == "identifiable"
    assert agg["identifiability"]["initial"]["range"] == pytest.approx(2.0)
    assert agg["dominant_term"] == "pde_sei" and agg["loss_shares"]["pde_sei"] == pytest.approx(0.6)
    assert agg["loss_terms"]["pde_sei"] == (6.0, 0.0)
    stuck = _run(
        "s", [_seed(0, 4, 6, "SAFE", a_sei=10**22.88, init=22.88), _seed(1, 4, 6, "SAFE", a_sei=10**22.87, init=22.88)]
    )
    assert mrt.aggregate(stuck)["identifiability"]["status"] == "not identifiable"
    assert mrt.aggregate(stuck)["loss_terms"] == {} and mrt.aggregate(stuck)["dominant_term"] is None


def test_render_markdown_identifiability_and_loss_sections(mrt):
    losses = {"pde_thermal": 1.0, "pde_sei": 6.0, "ic_T": 1.0, "ic_c": 1.0, "observe_T": 1.0}
    run = _run(
        "r",
        [
            _seed(0, 4, 6, "SAFE", a_sei=10**22.1, init=21.0, losses=losses),
            _seed(1, 4, 6, "SAFE", a_sei=10**22.0, init=23.0, losses=losses),
        ],
    )
    text = mrt.render_markdown([run], generated="x")
    assert "## A_sei identifiability across seeds" in text
    assert "| r | 2 | 21.00..23.00 | 22.05 ± 0.05 | 0.10 | **identifiable** |" in text
    assert "## Final weighted loss terms" in text
    assert (
        "| r | 1.00e+00 (10%) | 6.00e+00 (60%) | 1.00e+00 (10%) | 1.00e+00 (10%) | 1.00e+00 (10%) | pde_sei |" in text
    )
    bare = mrt.render_markdown([_run("b", [_seed(0, 1, 2, "SAFE"), _seed(1, 1, 2, "SAFE")])], generated="x")
    assert "## Final weighted loss terms" not in bare and "**consistent across seeds**" in bare


def test_training_from_history_reads_partial_runs(mrt, tmp_path):
    import pandas as pd

    assert mrt._training_from_history(str(tmp_path), 0) == {}
    pd.DataFrame(
        {
            "step": [0, 100],
            "log10_A_sei": [21.0, 21.4],
            "pde_thermal": [1, 0.1],
            "pde_sei": [9, 0.9],
            "ic_T": [1, 0.1],
            "ic_c": [1, 0.1],
            "observe_T": [1, 0.2],
        }
    ).to_csv(tmp_path / "loss_history_seed0.csv", index=False)
    t = mrt._training_from_history(str(tmp_path), 0)
    assert t["log10_A_sei_init"] == 21.0 and t["final_losses"]["pde_sei"] == 0.9 and t["seed"] == 0


def test_compare_runs_single_shared_seed_and_means(mrt):
    a = _run("subset", [_seed(0, 2.0, 4.0, "SAFE", a_sei=1e20), _seed(1, 2.2, 4.4, "SAFE", a_sei=1e20)])
    b = _run("full", [_seed(0, 1.8, 4.1, "SAFE", a_sei=1e21)])
    cmp = mrt.compare_runs(a, b)  # exactly one shared seed -> per-seed basis
    assert cmp["basis"] == "seed 0 of each run"
    rows = {label: (va, vb, d) for label, va, vb, d in cmp["rows"]}
    assert rows["train MAE (°C)"] == (2.0, 1.8, pytest.approx(-0.2))
    assert rows["log10 A_sei"] == (20.0, 21.0, pytest.approx(1.0))
    assert cmp["per_battery"] == [("B0025", 4.0, 4.1), ("B0056", 6.0, 6.1)]
    both = _run("full2", [_seed(0, 1.8, 4.1, "SAFE"), _seed(1, 1.6, 3.9, "SAFE")])
    cmp = mrt.compare_runs(a, both)  # two shared seeds -> means
    assert cmp["basis"] == "means over 2 and 2 seeds"
    assert dict((r[0], r[3]) for r in cmp["rows"])["held-out MAE (°C)"] == pytest.approx(4.0 - 4.2)
    assert mrt.compare_runs(a, both, seed=1)["basis"] == "seed 1 of each run"


def test_render_markdown_comparison_section(mrt):
    a = _run("subset", [_seed(0, 2.0, 4.0, "SAFE")])
    b = _run("full", [_seed(0, 1.8, 4.1, "DANGER")])
    text = mrt.render_markdown([a, b], generated="x", compare=[("subset", "full")])
    assert "## Run-to-run comparisons" in text and "### subset vs full (seed 0 of each run)" in text
    assert "| train MAE (°C) | 2.00 | 1.80 | -0.20 |" in text
    assert "| B0056 | 6.00 | 6.10 | +0.10 |" in text
    with pytest.raises(KeyError, match="unknown run name"):
        mrt.render_markdown([a], generated="x", compare=[("subset", "nope")])
    assert "## Run-to-run comparisons" not in mrt.render_markdown([a, b], generated="x")


def test_main_compare_flag(mrt, tmp_path):
    pytest.importorskip("deepxde")
    a_art, a_data, _, _ = _write_run(tmp_path, "a", n_seeds=1)
    b_art, b_data, _, _ = _write_run(tmp_path, "b", n_seeds=1)
    out = tmp_path / "R.md"
    mrt.main(
        ["--run", "a", a_art, a_data, "--run", "b", b_art, b_data, "--out", str(out), "--compare", "a", "b"],
        twin_factory=StubTwin,
        log=None,
    )
    assert "### a vs b (seed 0 of each run)" in out.read_text(encoding="utf-8")


def test_render_markdown_reference_per_ambient_and_verdict_sections(mrt):
    run = _run("r", [_seed(0, 2.0, 4.0, "SAFE"), _seed(1, 2.2, 4.4, "DANGER")])
    text = mrt.render_markdown([run], generated="x")
    # Databricks v9 reference comes before the per-battery tables, which come before the summary.
    i_ref, i_bat, i_amb, i_sum = (
        text.index(h)
        for h in (
            "## Databricks reference result (v9)",
            "## Per-battery held-out MAE",
            "## Per-ambient held-out MAE",
            "## Summary across seeds",
        )
    )
    assert i_ref < i_bat < i_amb < i_sum
    assert "| **overall** | 7.65 | 4.20 ± 0.20 |" in text
    assert "| B0036 | 11.40 | -- (absent) |" in text and "| B0025 | 4.20 | 4.20 ± 0.20 |" in text
    assert "| B0056 | -- (absent in v9) | 6.20 ± 0.20 |" in text
    assert "| 4 °C | 3.20 ± 0.20 |" in text and "| 24 °C | 4.20 ± 0.20 |" in text
    # Labels everywhere the core temperature / verdict appear.
    assert "core T max (°C, unvalidated)" in text and "verdicts (ambient ≤ 45 °C only)" in text
    assert "**unvalidated**" in text and "**excludes ambient > 45 °C**" in text
    # Verdict spread with deciding criteria and threshold spread.
    assert (
        "| r | 2 | 1 | 0 | 1 | **no** | 1 x none, 1 x sei_below_threshold | 0.800..0.800 (0.800 ± 0.000) | 55.0..55.0 |"
        in text
    )
    assert (
        "| r | 1 | 1.00e+22 | 2.20 | 2.70 | 4.40 | 4.90 | 55.0 | 0.900 | 0.800 | DANGER | sei_below_threshold |" in text
    )
    assert "## Databricks reference" not in mrt.render_markdown([run], generated="x", reference=False)


def test_option_decision_rule(mrt):
    def run_with(name, per):
        seeds = [_seed(k, 2.0, 4.0, "SAFE", per_ho={b: v + 0.1 * k for b, v in per.items()}) for k in range(3)]
        return _run(name, seeds)

    base = run_with(
        "base", {"B0025": 5.0, "B0036": 9.0, "B0042": 1.5, "B0056": 2.0}
    )  # worst: B0036, B0025; spread 0.08
    better = run_with("better", {"B0025": 4.0, "B0036": 7.0, "B0042": 1.5, "B0056": 2.0})
    d = mrt.option_decision(base, better)
    assert d["worst"] == ["B0036", "B0025"] and d["adopt"] is True
    assert all(r["status"] == "improved beyond spread" for r in d["rows"] if r["worst"])
    noisy = run_with("noisy", {"B0025": 4.95, "B0036": 8.95, "B0042": 1.5, "B0056": 2.0})  # within spread
    d = mrt.option_decision(base, noisy)
    assert d["adopt"] is False and "does not lower" in d["reason"]
    tradeoff = run_with("tradeoff", {"B0025": 4.0, "B0036": 7.0, "B0042": 3.0, "B0056": 2.0})  # degrades B0042
    d = mrt.option_decision(base, tradeoff)
    assert d["adopt"] is False and d["reason"].startswith("degrades B0042")
    text = mrt.render_markdown([base, better, tradeoff], generated="x", grid=("base", ["better", "tradeoff"]))
    assert "## Physics options grid" in text
    assert (
        "### better vs base: **ADOPT**" in text
        and "### tradeoff vs base: **REJECTED (negative result, left off)**" in text
    )
    assert "| B0036 (worst) | 9.10 | 7.10 | -2.00 | 0.08 | improved beyond spread |" in text


def test_fixed_a_sei_runs_are_not_assessed_for_identifiability(mrt):
    seeds = [_seed(k, 2.0, 4.0, "SAFE", a_sei=1e18, init=18.0) for k in range(2)]
    for sd in seeds:
        sd["training"] = {"physics": {"fix_log_a_sei": 18.0}}
    agg = mrt.aggregate(_run("fixed", seeds))
    assert agg["identifiability"]["status"] == "fixed (not trained)"
    text = mrt.render_markdown([_run("fixed", seeds)], generated="x")
    assert "**fixed (not trained)** | log10(A_sei) held at 18 with --fix-log-a-sei" in text


def test_render_markdown_without_protocol_omits_section(mrt):
    text = mrt.render_markdown([_run("r", [_seed(0, 1.0, 2.0, "SAFE")])], generated="x")
    assert "## Protocol" not in text and "## Runs" in text


def test_fmt_helpers(mrt):
    assert mrt.fmt(None) == "--" and mrt.fmt(float("nan")) == "--" and mrt.fmt(1.2345) == "1.23"
    assert mrt.fmt_ms((None, None)) == "--" and mrt.fmt_ms((1.0, 0.25)) == "1.00 ± 0.25"
    assert mrt.fmt_sci(7.66e22) == "7.66e+22" and mrt.fmt_sci(None) == "--"


# ----------------------------------------------------------------------------------------
# Collection (needs torch / deepxde for battery_pinn.inference)
# ----------------------------------------------------------------------------------------
class StubTwin:
    """Predicts a fixed T_delta and c; the SEI output drops at the abuse-test core (x=0) points."""

    def __init__(self, path, t_delta=0.0, c=1.0):
        self.checkpoint_path, self.t_delta, self.c = path, t_delta, c
        self.log_a_sei = 20.0

    @property
    def a_sei(self):
        return 10**self.log_a_sei

    def predict(self, points):
        out = np.zeros((len(points), 2))
        out[:, 0] = self.t_delta
        out[:, 1] = np.where(points[:, 0] == 0.0, self.c, 1.0)
        return out


def _write_run(tmp_path, name, n_seeds=2, held_out_ids=None):
    """Synthetic processed tables plus empty checkpoint files and a train_summary.json."""
    data_dir, artifacts = tmp_path / name / "data", tmp_path / name / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts.mkdir()
    train, held_out = synthetic.generate_train_and_held_out(n_batteries=4, n_held_out=2, cycles_per_battery=2)
    if held_out_ids:
        held_out["battery_id"] = held_out["battery_id"].map(
            dict(zip(sorted(held_out["battery_id"].unique()), held_out_ids))
        )
    train.to_csv(config.table_path(str(data_dir), config.TRAIN_TABLE), index=False)
    held_out.to_csv(config.table_path(str(data_dir), config.HELD_OUT_TABLE), index=False)
    for seed in range(n_seeds):
        (artifacts / f"battery_pinn_model_seed{seed}.pt").write_bytes(b"stub")
    (artifacts / "train_summary.json").write_text(
        json.dumps([{"seed": s, "device": "cpu", "A_sei": 1e20, "train_mae_c": 1.0} for s in range(n_seeds)])
    )
    return str(artifacts), str(data_dir), train, held_out


def test_collect_run_with_stub_twin(mrt, tmp_path):
    pytest.importorskip("deepxde")
    artifacts, data_dir, train, held_out = _write_run(tmp_path, "stub", n_seeds=2)
    twins = {0: dict(t_delta=0.0, c=1.0), 1: dict(t_delta=1.0, c=0.0)}  # seed 1: DANGER

    def factory(path):
        seed = int(path[-4])
        return StubTwin(path, **twins[seed])

    run = mrt.collect_run("stub", artifacts, data_dir, twin_factory=factory, log=None)
    assert [s["seed"] for s in run["seeds"]] == [0, 1]
    assert run["train"]["batteries"] == sorted(train["battery_id"].unique())
    assert run["held_out"]["rows"] == len(held_out) and run["train"]["cycles"] == 4
    s0, s1 = run["seeds"]
    expected_mae = float(np.abs(train["Temperature_measured"] - train["ambient_temperature"]).mean())
    assert s0["train_mae_c"] == pytest.approx(expected_mae, rel=1e-5)  # T_delta = 0 predicts ambient
    assert set(s0["per_battery_held_out_mae_c"]) == set(held_out["battery_id"].unique())
    assert s0["verdict"] == "SAFE" and s1["verdict"] == "DANGER"
    assert s1["train_mae_c"] > s0["train_mae_c"]  # T_delta = 1 -> +50 °C everywhere
    assert s0["A_sei"] == 1e20 and s0["training"]["device"] == "cpu"
    assert s0["log10_A_sei"] == pytest.approx(20.0)
    assert s0["log10_A_sei_init_assumed"] is True and s0["log10_A_sei_init"] == pytest.approx(22.885, abs=1e-3)
    assert math.isfinite(s0["threshold"]) and 0.0 <= s0["sei_min"] <= 1.0


def test_collect_run_requires_checkpoints(mrt, tmp_path):
    pytest.importorskip("deepxde")
    artifacts, data_dir, _, _ = _write_run(tmp_path, "empty", n_seeds=0)
    with pytest.raises(FileNotFoundError, match="no battery_pinn_model_seed"):
        mrt.collect_run("empty", artifacts, data_dir, twin_factory=StubTwin, log=None)


def test_main_writes_markdown_and_json_for_two_runs(mrt, tmp_path):
    pytest.importorskip("deepxde")
    a_art, a_data, _, _ = _write_run(tmp_path, "legacy", n_seeds=2, held_out_ids=config.HELD_OUT_BATTERIES[:2])
    b_art, b_data, _, _ = _write_run(tmp_path, "dynamic", n_seeds=3)
    out, js = tmp_path / "RESULTS.md", tmp_path / "RESULTS.json"
    runs = mrt.main(
        [
            "--run",
            "legacy",
            a_art,
            a_data,
            "--run",
            "dynamic",
            b_art,
            b_data,
            "--out",
            str(out),
            "--json",
            str(js),
            "--protocol",
            "tiny",
        ],
        twin_factory=StubTwin,
        log=None,
    )
    assert [r["name"] for r in runs] == ["legacy", "dynamic"]
    text = out.read_text(encoding="utf-8")
    assert "| legacy | 2 |" in text and "| dynamic | 3 |" in text
    assert "**legacy**: held-out table lacks designated batteries" in text
    assert "| B0025 |" in text and "-- (absent)" in text  # B0025 only in the legacy held-out set
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert set(payload) == {"runs", "aggregates"}
    assert payload["aggregates"]["dynamic"]["n_seeds"] == 3
    assert payload["aggregates"]["dynamic"]["verdicts"] == {"DANGER": 0, "WARNING": 0, "SAFE": 3}


def test_main_reads_protocol_and_notes_files(mrt, tmp_path):
    pytest.importorskip("deepxde")
    art, data, _, _ = _write_run(tmp_path, "r", n_seeds=1)
    (tmp_path / "protocol.md").write_text("Adam 1000 iterations\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Both runs are under-trained.\n", encoding="utf-8")
    mrt.main(
        [
            "--run",
            "r",
            art,
            data,
            "--out",
            str(tmp_path / "R.md"),
            "--protocol-file",
            str(tmp_path / "protocol.md"),
            "--notes-file",
            str(tmp_path / "notes.md"),
        ],
        twin_factory=StubTwin,
        log=None,
    )
    text = (tmp_path / "R.md").read_text(encoding="utf-8")
    assert "Adam 1000 iterations" in text
    assert text.rstrip().endswith("## Notes\n\nBoth runs are under-trained.")


def test_render_markdown_notes_section_is_optional(mrt):
    run = _run("r", [_seed(0, 1.0, 2.0, "SAFE")])
    assert "## Notes" not in mrt.render_markdown([run], generated="x")
    assert mrt.render_markdown([run], generated="x", notes="  hello  ").rstrip().endswith("## Notes\n\nhello")


def test_end_to_end_with_real_checkpoints(mrt, tmp_path):
    """Two genuinely trained (very short) seeds go through DigitalTwin, evaluate and abuse_test."""
    pytest.importorskip("deepxde")
    from battery_pinn.training import train_ensemble

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train, held_out = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=1, cycles_per_battery=2)
    train.to_csv(config.table_path(str(data_dir), config.TRAIN_TABLE), index=False)
    held_out.to_csv(config.table_path(str(data_dir), config.HELD_OUT_TABLE), index=False)
    artifacts = tmp_path / "artifacts"
    train_ensemble(
        train,
        [0, 1],
        str(artifacts),
        iterations=2,
        lbfgs_iterations=0,
        num_domain=20,
        num_boundary=5,
        num_initial=20,
        num_test=10,
        device="cpu",
    )

    out = tmp_path / "RESULTS.md"
    runs = mrt.main(["--run", "real", str(artifacts), str(data_dir), "--out", str(out)], log=None)
    seeds = runs[0]["seeds"]
    assert [s["seed"] for s in seeds] == [0, 1]
    assert all(math.isfinite(s["train_mae_c"]) and math.isfinite(s["held_out_mae_c"]) for s in seeds)
    assert all(s["verdict"] in mrt.VERDICTS for s in seeds)
    # The checkpoint's calibrated A_sei agrees with what training recorded in train_summary.json.
    assert all(s["A_sei"] == pytest.approx(s["training"]["A_sei"], rel=1e-5) for s in seeds)
    assert "| real | 2 |" in out.read_text(encoding="utf-8")
