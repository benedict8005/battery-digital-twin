"""``--json`` prints one machine-readable object for ``simulate`` (no PyTorch needed)."""

import json

import pytest

from battery_pinn.cli import main

BASE = ["simulate", "--current", "3", "--ambient", "30", "--duration", "120", "--n-x", "5", "--n-t", "20"]


def run(capsys, argv):
    main(argv)
    return capsys.readouterr().out


def text_rows(out):
    return [[float(v) for v in line.split()] for line in out.splitlines() if not line.startswith("#")]


def test_simulate_json_is_one_object_matching_the_text_output(capsys):
    rows = text_rows(run(capsys, BASE))
    out = run(capsys, [*BASE, "--json"])
    assert out.count("\n") == 1
    data = json.loads(out)
    assert data["command"] == "simulate"
    assert data["ambient_c"] == 30.0 and data["duration_s"] == 120.0
    assert len(data["x"]) == len(data["temperature_c"]) == 5
    assert "sei_stability" not in data
    for (x, temp), jx, jt in zip(rows, data["x"], data["temperature_c"]):
        assert x == pytest.approx(jx, abs=5e-5)
        assert temp == pytest.approx(jt, abs=5e-5)


def test_simulate_json_with_sei_adds_the_stability_profile(capsys):
    data = json.loads(run(capsys, [*BASE, "--log-a-sei", "1", "--json"]))
    assert data["log_a_sei"] == 1.0
    assert len(data["sei_stability"]) == 5
    assert all(0.0 <= c <= 1.0 for c in data["sei_stability"])


def test_json_flag_can_come_from_a_config_file(tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text("[simulate]\ncurrent = 3\nambient = 30\nduration = 120\nn-x = 5\nn-t = 20\njson = true\n")
    assert json.loads(run(capsys, ["simulate", "--config", str(cfg)])) == json.loads(run(capsys, [*BASE, "--json"]))
