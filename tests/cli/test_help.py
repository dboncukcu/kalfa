"""Every command explains itself: a description, its options in groups and examples."""

import pytest

from kalfa.cli import examples_text, main

COMMANDS = ["run", "check", "describe", "predict", "generate", "prepare", "export", "plots", "resume", "sweep",
            "collect", "docs", "ls", "board", "stop", "contract"]


def help_of(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 0
    return capsys.readouterr().out


def test_the_top_level_help_lists_every_command_with_a_first_pass(capsys):
    out = help_of(["--help"], capsys)
    assert "a first pass:" in out and "--version" in out
    assert all(f"  {name} " in out or f"  {name}\n" in out for name in COMMANDS)


@pytest.mark.parametrize("name", COMMANDS)
def test_every_command_has_a_description_and_examples(name, capsys):
    out = help_of([name, "--help"], capsys)
    assert out.startswith(f"usage: kalfa {name}") and "examples:" in out
    assert f"kalfa {name}" in out.split("examples:")[1]


def test_the_examples_align_their_notes():
    text = examples_text(["kalfa a    first", "kalfa longer one    second", "kalfa alone"])
    lines = text.split("\n")
    assert lines[0] == "examples:" and lines[3] == "  kalfa alone"
    assert lines[1].index("first") == lines[2].index("second")
