"""A very short real training run: checkpoint plumbing and device handling end to end."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("deepxde")

from battery_pinn import synthetic  # noqa: E402
from battery_pinn.checkpoint import load_checkpoint  # noqa: E402
from battery_pinn.training import train_ensemble, train_one_seed  # noqa: E402


@pytest.fixture(scope="module")
def train_df():
    df, _ = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=0, cycles_per_battery=2)
    return df


def test_train_one_seed_on_cpu_writes_a_loadable_checkpoint(train_df, tmp_path, capsys):
    result = train_one_seed(train_df, seed=0, out_dir=str(tmp_path), iterations=5, lbfgs_iterations=0,
                            num_domain=20, num_boundary=5, num_initial=20, num_test=10, device="cpu")
    assert result["device"] == "cpu"
    assert "Training on cpu" in capsys.readouterr().out
    checkpoint = load_checkpoint(result["checkpoint"])
    assert all(v.device.type == "cpu" for v in checkpoint["model_state_dict"].values())
    assert np.isfinite(result["train_mae_c"]) and len(checkpoint["ocv_coeffs"]) == 5


def test_train_ensemble_passes_device_through_and_auto_falls_back(train_df, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    results = train_ensemble(train_df, [1], str(tmp_path), iterations=3, lbfgs_iterations=0,
                             num_domain=20, num_boundary=5, num_initial=20, num_test=10, device="cuda")
    out = capsys.readouterr().out
    assert "falling back to CPU" in out and results[0]["device"] == "cpu"
    summary = json.load(open(tmp_path / "train_summary.json"))
    assert summary[0]["seed"] == 1 and summary[0]["device"] == "cpu"


def test_select_anchor_subset_is_seeded_and_bounded():
    from battery_pinn.training import select_anchor_subset

    assert select_anchor_subset(100, None, 0) is None
    assert select_anchor_subset(100, 100, 0) is None and select_anchor_subset(100, 500, 0) is None
    a, b, c = (select_anchor_subset(100, 10, s) for s in (0, 0, 1))
    assert len(a) == 10 and len(set(a)) == 10 and (a == np.sort(a)).all()
    assert (a == b).all() and not (a == c).all()
    with pytest.raises(ValueError):
        select_anchor_subset(100, 0, 0)


def test_anchor_subset_and_loss_logging(train_df, tmp_path, capsys):
    import pandas as pd

    from battery_pinn import config

    result = train_one_seed(train_df, seed=3, out_dir=str(tmp_path), iterations=6, lbfgs_iterations=2,
                            num_domain=20, num_boundary=5, num_initial=20, num_test=10, display_every=2, device="cpu",
                            anchor_subset=40, log_a_sei_init=21.0)
    out = capsys.readouterr().out
    assert f"Observation points: {len(train_df)}; used as anchors / data fit: 40 (anchor subset, seed 3)" in out
    assert "Dominant term:" in out
    assert result["anchor_subset"] == 40 and result["log10_A_sei_init"] == 21.0
    assert set(result["final_losses"]) == set(config.LOSS_TERM_NAMES)
    breakdown = result["loss_breakdown"]
    assert breakdown["dominant"] in config.LOSS_TERM_NAMES
    assert sum(breakdown["shares"].values()) == pytest.approx(1.0)
    history = pd.read_csv(tmp_path / "loss_history_seed3.csv")
    assert list(history.columns) == ["step", "log10_A_sei"] + config.LOSS_TERM_NAMES
    assert history["step"].is_monotonic_increasing and history["step"].is_unique
    assert history["step"].iloc[0] <= 2 and history["step"].iloc[-1] >= 6   # Adam steps, then L-BFGS
    assert np.isfinite(history[config.LOSS_TERM_NAMES].values).all()
    assert history["log10_A_sei"].iloc[0] == pytest.approx(21.0, abs=1e-3)


def test_train_ensemble_spreads_a_sei_init_and_writes_identifiability(train_df, tmp_path, capsys):
    results = train_ensemble(train_df, [0, 1, 2], str(tmp_path), iterations=2, lbfgs_iterations=0,
                             num_domain=20, num_boundary=5, num_initial=20, num_test=10, device="cpu", a_sei_init_spread=1.0)
    inits = [r["log10_A_sei_init"] for r in results]
    assert inits[1] == pytest.approx(inits[0] + 1.0) and inits[2] == pytest.approx(inits[0] + 2.0)
    assessment = json.load(open(tmp_path / "identifiability.json"))
    assert assessment["n_seeds"] == 3 and assessment["initial"]["range"] == pytest.approx(2.0)
    assert assessment["status"] in ("identifiable", "not identifiable", "consistent across seeds")
    assert "A_sei identifiability (3 seeds):" in capsys.readouterr().out
    summary = json.load(open(tmp_path / "train_summary.json"))
    assert [s["log10_A_sei_init"] for s in summary] == pytest.approx(inits)
    # An explicit init pins every seed instead.
    pinned = train_ensemble(train_df, [0, 1], str(tmp_path / "pinned"), iterations=2, lbfgs_iterations=0,
                            num_domain=20, num_boundary=5, num_initial=20, num_test=10, device="cpu", log_a_sei_init=20.5)
    assert [r["log10_A_sei_init"] for r in pinned] == [20.5, 20.5]
