"""Every exit code of the command line, checked through ``main`` and through a real process."""

import subprocess
import sys

import pytest

from battery_pinn import cli
from battery_pinn.cli import EXIT_DATA, EXIT_MISSING, EXIT_OK, EXIT_USAGE, main

SIM = ["simulate", "--duration", "60", "--n-x", "5", "--n-t", "10"]


def exit_code(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


def test_codes_are_distinct_and_documented_values():
    assert (EXIT_OK, EXIT_USAGE, EXIT_DATA, EXIT_MISSING) == (0, 2, 3, 4)


def test_success_returns_without_exiting(capsys):
    assert main(SIM) is None
    assert capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["no-such-command"],
        ["simulate", "--no-such-flag"],
        ["simulate", "--boundary", "robin"],
        ["simulate", "--current", "lots"],
        ["simulate", "-v", "-q"],
        ["simulate", "--config", "no-such-file.toml"],
    ],
)
def test_usage_errors_exit_2(argv, capsys):
    assert exit_code(argv) == EXIT_USAGE
    assert "Traceback" not in capsys.readouterr().err


def test_help_exits_0(capsys):
    assert exit_code(["simulate", "--help"]) == 0


def test_malformed_table_exits_3_with_a_message(tmp_path, capsys):
    bad = tmp_path / "train.csv"
    bad.write_text("battery_id,cycle\nB1,1\n")
    assert exit_code(["inspect-data", "--data", str(bad)]) == EXIT_DATA
    err = capsys.readouterr().err
    assert str(bad) in err and "error:" in err and "Traceback" not in err


def test_missing_data_file_exits_4(tmp_path, capsys):
    assert exit_code(["inspect-data", "--data", str(tmp_path)]) == EXIT_MISSING
    err = capsys.readouterr().err
    assert "No such data file" in err and "prepare-data" in err


def test_missing_checkpoints_exit_4_for_evaluate_and_abuse_test(tmp_path, capsys):
    pytest.importorskip("torch")
    from battery_pinn import synthetic

    train, held_out = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=1, cycles_per_battery=3, seed=0)
    train.to_csv(tmp_path / "train.csv", index=False)
    held_out.to_csv(tmp_path / "held_out.csv", index=False)
    common = ["--checkpoints", str(tmp_path / "none"), "--data", str(tmp_path / "train.csv")]
    assert exit_code(["abuse-test", *common]) == EXIT_MISSING
    args = ["evaluate", *common, "--held-out-data", str(tmp_path / "held_out.csv"), "--no-battery-check"]
    assert exit_code(args) == EXIT_MISSING
    assert capsys.readouterr().err.count("No checkpoints found") == 2


def test_exit_code_reaches_the_operating_system(tmp_path):
    def run(*argv):
        return subprocess.run([sys.executable, "-m", "battery_pinn", *argv], capture_output=True, text=True)

    assert run(*SIM).returncode == 0
    assert run("simulate", "--bogus").returncode == EXIT_USAGE
    missing = run("inspect-data", "--data", str(tmp_path))
    assert missing.returncode == EXIT_MISSING and "Traceback" not in missing.stderr


def test_exit_codes_are_public():
    assert {"EXIT_OK", "EXIT_USAGE", "EXIT_DATA", "EXIT_MISSING", "MissingInputError"} <= set(cli.__all__)
