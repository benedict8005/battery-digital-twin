"""Every command and every option documents itself in ``--help`` (no PyTorch needed)."""

import argparse

import pytest

from battery_pinn.cli import build_parser, main

PARSER = build_parser()
COMMANDS = PARSER._subparsers._group_actions[0].choices  # name -> subparser


def _options():
    found = []
    for name, sub in COMMANDS.items():
        for action in sub._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            found.append(pytest.param(name, action, id=f"{name}:{'/'.join(action.option_strings)}"))
    return found


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_command_has_a_help_line(name):
    listed = {a.dest: a for a in PARSER._subparsers._group_actions[0]._choices_actions}
    assert name in listed and listed[name].help


@pytest.mark.parametrize(("name", "action"), _options())
def test_every_option_has_help_text(name, action):
    assert action.help, f"{name} {action.option_strings} has no help text"


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_help_prints_every_long_option(name, capsys):
    with pytest.raises(SystemExit) as exc:
        main([name, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for action in COMMANDS[name]._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                assert opt in out
