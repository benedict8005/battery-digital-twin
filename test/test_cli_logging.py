"""--verbose and --quiet choose the level of the ``battery_pinn`` logger; results stay on stdout."""

import logging

import pytest

from battery_pinn.cli import main
from battery_pinn.logging_setup import LOGGER_NAME, configure_logging, level_for

SIM = ["simulate", "--n-x", "3", "--n-t", "5"]


@pytest.fixture(autouse=True)
def _restore_logger():
    yield
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_level_for_each_flag_combination():
    assert level_for() == logging.INFO
    assert level_for(verbose=True) == logging.DEBUG
    assert level_for(quiet=True) == logging.WARNING
    with pytest.raises(ValueError, match="both"):
        level_for(verbose=True, quiet=True)


@pytest.mark.parametrize(
    "flags, level",
    [([], logging.INFO), (["-v"], logging.DEBUG), (["--verbose"], logging.DEBUG), (["-q"], logging.WARNING)],
)
def test_main_sets_the_package_logger_level(flags, level):
    main([*SIM, *flags])
    assert logging.getLogger(LOGGER_NAME).level == level


def test_verbose_and_quiet_together_exit_with_code_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main([*SIM, "-v", "-q"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_every_command_accepts_both_flags():
    from battery_pinn.cli import build_parser

    sub = next(a for a in build_parser()._actions if hasattr(a, "choices") and a.choices)
    for name, command in sub.choices.items():
        flags = {s for a in command._actions for s in a.option_strings}
        assert {"-v", "--verbose", "-q", "--quiet"} <= flags, name


def test_records_go_to_stderr_and_results_stay_on_stdout(capsys):
    main([*SIM, "-v"])
    logging.getLogger(LOGGER_NAME).debug("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.err and "DEBUG" in captured.err
    assert "hello" not in captured.out and "temperature" in captured.out


def test_quiet_hides_info_records(capsys):
    configure_logging(quiet=True)
    log = logging.getLogger(LOGGER_NAME)
    log.info("chatty")
    log.warning("careful")
    err = capsys.readouterr().err
    assert "chatty" not in err and "careful" in err


def test_configuring_twice_does_not_duplicate_output(capsys):
    configure_logging()
    configure_logging()
    logging.getLogger(LOGGER_NAME).info("once")
    assert capsys.readouterr().err.count("once") == 1


def test_config_file_can_set_quiet(tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text("[simulate]\nquiet = true\n")
    main([*SIM, "--config", str(cfg)])
    assert logging.getLogger(LOGGER_NAME).level == logging.WARNING


def test_generate_data_status_goes_to_stderr_and_quiet_silences_it(tmp_path, capsys):
    main(["generate-data", "--out", str(tmp_path / "a"), "--batteries", "3", "--held-out", "1", "--cycles", "3"])
    first = capsys.readouterr()
    assert "Wrote" in first.err and "synthetic data" in first.err
    assert first.out == ""
    main(["generate-data", "--out", str(tmp_path / "b"), "--batteries", "3", "--held-out", "1", "--cycles", "3", "-q"])
    quiet = capsys.readouterr()
    assert quiet.err == "" and quiet.out == ""
    assert (tmp_path / "b" / "train.csv").is_file()
