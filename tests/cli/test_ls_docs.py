import re

from helpers import example
from kalfa.cli import main
from kalfa.docs import render


def test_ls_lists_packs_and_legos(capsys):
    assert main(["ls", "/alias/kalfa/tabular"]) == 0
    out = capsys.readouterr().out
    assert "/alias/kalfa/tabular" in out and re.search(r"parquet\s+source\s+/source/kalfa/parquet", out)
    assert main(["ls", "/criterion", "--kind", "criterion"]) == 0
    out = capsys.readouterr().out
    assert "/criterion/kalfa/mse" in out and "partial: True" in out
    assert main(["ls"]) == 0
    out = capsys.readouterr().out
    assert "/alias/kalfa/tabular" in out and "/turn/kalfa/alternating" in out


def test_ls_takes_plugins(capsys):
    assert main(["ls", "alad", "--plugin", "myexample"]) == 0
    out = capsys.readouterr().out
    assert "/objective/myexample/alad_discriminator" in out and "/objective/myexample/alad_generator" in out
    assert main(["ls", "/objective/myexample", "--config", example("alad")]) == 0
    assert "/objective/myexample/alad_generator" in capsys.readouterr().out
    assert main(["ls", "alad", "--plugin", "no_such_plugin_module"]) == 1
    assert "cannot import plugin no_such_plugin_module" in capsys.readouterr().err


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
    assert main(["docs", "--config", example("alad")]) == 0
    assert "| `/objective/myexample/alad_generator` |" in capsys.readouterr().out


def test_docs_reports_a_plugin_it_cannot_import(capsys):
    assert main(["docs", "--plugin", "no_such_plugin_module"]) == 1
    captured = capsys.readouterr()
    assert "## Plugin legos" in captured.out
    assert "[plugin_import_failed]" in captured.err and "no_such_plugin_module" in captured.err
