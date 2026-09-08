"""DOCS.md is the lego reference generated from the registry; it must match the registry exactly."""

import re
from pathlib import Path

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.cli import main
from kalfa.docs import render, template_uris


def test_docs_file_matches_the_registry():
    expected = render()
    assert (ROOT / "DOCS.md").read_text() == expected, "DOCS.md is stale; run: uv run kalfa docs --write DOCS.md"
    assert "| `/criterion/kalfa/mae` | `mae` |" in expected and "### /alias/kalfa/tabular" in expected
    assert "## strategy" in expected and "## data" in expected and "/lego/test/" not in expected


def test_docs_command_prints_and_writes(tmp_path, capsys):
    assert main(["docs"]) == 0
    assert capsys.readouterr().out == render()
    target = tmp_path / "ref.md"
    assert main(["docs", "--write", str(target)]) == 0
    assert target.read_text() == render()


def test_docs_lists_the_legos_of_a_plugin(capsys):
    assert main(["docs", "--plugin", "myexample"]) == 0
    out = capsys.readouterr().out
    assert "## Plugin legos" in out and "| `/objective/myexample/alad_discriminator` |" in out
    assert "## Skeleton steps" in out and out != render()
    assert main(["docs", "--config", str(ROOT / "examples" / "alad" / "config.yaml")]) == 0
    assert "| `/objective/myexample/alad_generator` |" in capsys.readouterr().out
    assert render() == (ROOT / "DOCS.md").read_text(), "the plugin section must stay out of the std reference"


def test_docs_reports_a_plugin_it_cannot_import(capsys):
    assert main(["docs", "--plugin", "no_such_plugin_module"]) == 1
    captured = capsys.readouterr()
    assert "## Plugin legos" in captured.out
    assert "[plugin_import_failed]" in captured.err and "no_such_plugin_module" in captured.err


def listed_uris(section):
    return set(re.findall(r"^\| `(/[^`]+)`", section, re.M))


def test_the_skeleton_section_is_exactly_what_the_template_calls():
    text = (ROOT / "DOCS.md").read_text()
    catalog, _, rest = text.partition("## Skeleton steps")
    skeleton, _, packs = rest.partition("## Alias packs")
    called = template_uris()
    assert called, "the template calls no registered lego"
    assert listed_uris(skeleton) == called
    assert not listed_uris(catalog) & called
    assert "/lego/kalfa/fit" in called and "/builder/kalfa/module" in called
    assert "/criterion/kalfa/mae" not in called and "/adapter/kalfa/criterion" not in called
    assert "not written in a config" in skeleton and listed_uris(packs) == set()
