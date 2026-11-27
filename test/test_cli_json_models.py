"""``--json`` for ``evaluate`` and ``abuse-test``, with stubbed models so nothing is trained."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("deepxde")

from battery_pinn import inference, synthetic
from battery_pinn.cli import main


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    train, held_out = synthetic.generate_train_and_held_out(n_batteries=2, n_held_out=1, cycles_per_battery=2)
    train.to_csv(tmp_path / "train.csv", index=False)
    held_out.to_csv(tmp_path / "held_out.csv", index=False)
    for seed in (0, 1):
        (tmp_path / f"battery_pinn_model_seed{seed}.pt").write_bytes(b"")
    monkeypatch.setattr(inference, "DigitalTwin", lambda path: path)
    monkeypatch.setattr(
        inference,
        "evaluate",
        lambda twin, df: {"mae_c": 1.5, "rmse_c": 2.0, "per_battery_mae_c": {"B1": 1.25}},
    )
    verdict = {"verdict": "SAFE", "criterion": "sei", "core_temp_max_c": 40.0, "sei_min": 0.9, "threshold": 0.5}
    monkeypatch.setattr(inference, "abuse_test", lambda twin, df: dict(verdict))
    monkeypatch.setattr(inference, "plot_abuse_test", lambda results, path: None)
    return tmp_path


def evaluate_args(d, *extra):
    return [
        "evaluate",
        "--checkpoints",
        str(d),
        "--data",
        str(d / "train.csv"),
        "--held-out-data",
        str(d / "held_out.csv"),
        "--no-battery-check",
        *extra,
    ]


def test_evaluate_json_is_one_object_matching_the_text_lines(workdir, capsys):
    main(evaluate_args(workdir))
    text = capsys.readouterr().out
    main(evaluate_args(workdir, "--json"))
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    data = json.loads(out)
    assert data["command"] == "evaluate"
    assert sorted(data["models"]) == ["battery_pinn_model_seed0.pt", "battery_pinn_model_seed1.pt"]
    one = data["models"]["battery_pinn_model_seed0.pt"]
    assert one["train"]["mae_c"] == 1.5 and one["held_out"]["per_battery_mae_c"] == {"B1": 1.25}
    assert "train MAE 1.50 C / RMSE 2.00 C" in text and "B1: MAE 1.25 C" in text


def test_abuse_test_json_has_summary_and_models_and_no_text(workdir, capsys):
    argv = ["abuse-test", "--checkpoints", str(workdir), "--data", str(workdir / "train.csv"), "--json"]
    main(argv)
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    data = json.loads(out)
    assert data["command"] == "abuse-test"
    assert data["summary"]["n"] == 2 and data["summary"]["verdict_counts"]["SAFE"] == 2
    assert data["models"]["seed0"]["verdict"] == "SAFE"
    written = json.loads((workdir / "abuse_test_summary.json").read_text())
    assert written["models"] == data["models"]


def test_json_flag_can_come_from_a_config_file(workdir, tmp_path_factory, capsys):
    cfg = tmp_path_factory.mktemp("cfg") / "run.toml"
    cfg.write_text(f'[abuse-test]\ncheckpoints = "{workdir}"\ndata = "{workdir / "train.csv"}"\njson = true\n')
    main(["abuse-test", "--config", str(cfg)])
    assert json.loads(capsys.readouterr().out)["command"] == "abuse-test"
