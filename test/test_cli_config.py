"""``--config`` reads option defaults from a TOML or YAML file (no PyTorch needed)."""

import pytest

from battery_pinn.cli import main
from battery_pinn.cli_config import load_config_file

FLAGS = ["simulate", "--current", "3", "--ambient", "30", "--duration", "120", "--n-x", "5", "--n-t", "20"]
FLAGS += ["--newton-cooling", "--boundary", "dirichlet"]

TOML = """
[simulate]
current = 3
ambient = 30.0
duration = 120
n-x = 5
n_t = 20
newton-cooling = true
boundary = "dirichlet"
"""


def run(capsys, argv):
    main(argv)
    return capsys.readouterr().out


def test_toml_config_reproduces_the_same_run_as_flags(tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text(TOML)
    from_flags = run(capsys, FLAGS)
    assert from_flags.count("\n") > 5
    assert run(capsys, ["simulate", "--config", str(cfg)]) == from_flags


def test_yaml_config_reproduces_the_same_run_as_flags(tmp_path, capsys):
    pytest.importorskip("yaml")
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "simulate:\n  current: 3\n  ambient: 30\n  duration: 120\n  n-x: 5\n  n-t: 20\n"
        "  newton-cooling: true\n  boundary: dirichlet\n"
    )
    assert run(capsys, ["simulate", "--config", str(cfg)]) == run(capsys, FLAGS)


def test_flags_override_the_config_file(tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text(TOML)
    argv = ["simulate", "--config", str(cfg), "--ambient", "10"]
    assert run(capsys, argv) == run(capsys, [*FLAGS[:3], "--ambient", "10", *FLAGS[5:]])


def test_config_for_another_command_is_ignored(tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text('[simulate]\ncurrent = 3\n\n[inspect-data]\ndata = "somewhere"\n')
    assert "temperature" in run(capsys, ["simulate", "--config", str(cfg), "--n-x", "3", "--n-t", "5"])


def test_config_sets_an_option_of_another_command(tmp_path):
    cfg = tmp_path / "run.toml"
    cfg.write_text('[inspect-data]\ndata = "nowhere.csv"\n')
    with pytest.raises(SystemExit, match="No such data file: nowhere.csv"):
        main(["inspect-data", "--config", str(cfg)])


@pytest.mark.parametrize(
    "text, message",
    [
        ("[simulate]\ncurent = 3\n", "[simulate] has unknown option 'curent'"),
        ("[simulat]\ncurrent = 3\n", "unknown command [simulat]"),
        ('[simulate]\nboundary = "robin"\n', "is not one of"),
        ("current = 3\n", "one table per command"),
        ("[simulate\n", "cannot parse config file"),
    ],
)
def test_bad_config_exits_with_a_readable_message(tmp_path, capsys, text, message):
    cfg = tmp_path / "bad.toml"
    cfg.write_text(text)
    with pytest.raises(SystemExit) as exc:
        main(["simulate", "--config", str(cfg)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert message in err and str(cfg) in err and "Traceback" not in err


def test_missing_and_unknown_type_config_files(tmp_path):
    with pytest.raises(Exception, match="config file not found"):
        load_config_file(str(tmp_path / "nope.toml"))
    other = tmp_path / "run.json"
    other.write_text("{}")
    with pytest.raises(Exception, match="unsupported config type .json"):
        load_config_file(str(other))


def test_help_documents_the_config_option(capsys):
    with pytest.raises(SystemExit):
        main(["simulate", "--help"])
    assert "--config FILE" in capsys.readouterr().out
