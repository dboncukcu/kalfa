import argparse
from pathlib import Path

import pytest

from kalfa import __version__
from kalfa.cli import build_parser, examples_text, main, version_text


def subcommands():
    parser = build_parser()
    action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return action.choices


def help_of(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 0
    return capsys.readouterr().out


def test_top_level_help_lists_every_command_and_a_first_pass(capsys):
    out = help_of(["--help"], capsys)
    assert out.startswith("usage: kalfa [-h] [--version] <command> ...\n")
    assert "--version" in out and "a first pass:" in out
    assert "kalfa <command> --help explains a command with examples" in out
    listed = out.split("commands:")[1].split("a first pass:")[0]
    for name, parser in subcommands().items():
        assert f"    {name}" in listed and parser.format_usage().split()[2] == name


@pytest.mark.parametrize("name", list(subcommands()))
def test_command_help_has_a_description_and_examples(name, capsys):
    out = help_of([name, "--help"], capsys)
    parser = subcommands()[name]
    assert out.startswith(f"usage: kalfa {name}")
    assert parser.description and parser.description in out
    assert parser.epilog and parser.epilog.startswith("examples:\n") and "examples:\n" in out
    examples = out.split("examples:\n", 1)[1]
    assert f"  kalfa {name}" in examples and examples == parser.epilog[len("examples:\n"):] + "\n"


def test_version_prints_kalfa_with_its_dependencies(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert out == version_text() + "\n"
    assert out.startswith(f"kalfa {__version__} (cirak ")
    assert all(f", {name} " in out for name in ("tezgah", "torch", "python")) and out.endswith(")\n")


def test_examples_text_aligns_the_notes_of_the_examples():
    text = examples_text(["kalfa a    first", "kalfa longer one    second", "kalfa alone"])
    assert text.split("\n") == ["examples:", "  " + "kalfa a".ljust(16) + "   first", "  kalfa longer one   second",
                                "  kalfa alone"]


def test_every_subcommand_is_named_in_a_cli_test():
    texts = [path.read_text() for path in Path(__file__).parent.glob("test_*.py")]
    missing = [name for name in subcommands() if not any(f'main(["{name}"' in text for text in texts)]
    assert missing == []
