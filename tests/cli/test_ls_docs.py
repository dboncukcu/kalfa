import re
from pathlib import Path

from helpers import example
from kalfa.cli import main
from kalfa.contract import Contract
from kalfa.docs import render


PLUGIN = str(Path(example("alad")).parent / "myexample.py")
PACKS = ["/alias/kalfa/base", "/alias/kalfa/lazy_tabular", "/alias/kalfa/tabular", "/alias/kalfa/text",
         "/alias/kalfa/vision"]


def uris_of(out):
    return [line.split("  ")[0] for line in out.splitlines() if line.startswith("/")]


def test_ls_lists_the_packs_then_every_lego(capsys):
    assert main(["ls"]) == 0
    out = capsys.readouterr().out
    assert [uri for uri in uris_of(out) if uri.startswith("/alias/")] == PACKS
    assert re.search(r"^/alias/kalfa/base  .*/kalfa/packs/base\.yaml$", out, re.M)
    assert re.search(r"^  parquet\s+source\s+/source/kalfa/parquet$", out, re.M)
    assert re.search(r"^/turn/kalfa/alternating\s+turn\s+", out, re.M)
    assert re.search(r"^/criterion/kalfa/mse\s+criterion\s+Mean squared error\s+\[alias: mse; partial: True\]\s+"
                     r"kalfa\.std\.criterion\.kalfa\.regression$", out, re.M)


def test_ls_prefix_narrows_to_a_pack_or_a_kind(capsys):
    assert main(["ls", "/alias/kalfa/tabular"]) == 0
    out = capsys.readouterr().out
    assert [uri for uri in uris_of(out) if uri.startswith("/alias/")] == ["/alias/kalfa/tabular"]
    assert out.startswith("/alias/kalfa/tabular  ") and out.splitlines()[0].endswith("/kalfa/packs/tabular.yaml")
    assert re.search(r"^  parquet\s+source\s+/source/kalfa/parquet$", out, re.M)
    assert re.search(r"^  standard_scaler\s+pre\s+/pre/sklearn/standard_scaler$", out, re.M)
    assert main(["ls", "/criterion"]) == 0
    out = capsys.readouterr().out
    assert uris_of(out) and all(uri.startswith("/criterion/kalfa/") for uri in uris_of(out))
    assert "/criterion/kalfa/mse" in uris_of(out) and "partial: True" in out
    assert main(["ls", "/criterion", "--kind", "plot"]) == 0
    assert capsys.readouterr().out == "nothing found\n"
    assert main(["ls", "--kind", "strategy"]) == 0
    out = capsys.readouterr().out
    assert [uri for uri in uris_of(out) if not uri.startswith("/alias/")] == [
        "/strategy/kalfa/grid", "/strategy/kalfa/optuna", "/strategy/kalfa/random", "/strategy/kalfa/sobol"]
    assert re.search(r"^  grid\s+strategy\s+/strategy/kalfa/grid$", out, re.M) and "  parquet" not in out


def test_ls_word_searches_names_aliases_and_descriptions(capsys):
    assert main(["ls", "scaler"]) == 0
    out = capsys.readouterr().out
    assert "/pre/sklearn/standard_scaler" in uris_of(out) and "/adapter/kalfa/objective" in uris_of(out)
    assert "  alias: standard_scaler (base, lazy_tabular, tabular, text, vision)" in out.splitlines()
    assert main(["ls", "scaler", "--kind", "pre"]) == 0
    out = capsys.readouterr().out
    assert uris_of(out) == ["/pre/kalfa/median_std_scaler", "/pre/sklearn/max_abs_scaler", "/pre/sklearn/minmax_scaler",
                            "/pre/sklearn/robust_scaler", "/pre/sklearn/standard_scaler"]


def test_ls_says_when_nothing_is_found(capsys):
    assert main(["ls", "zzz_nothing_here"]) == 0
    assert capsys.readouterr().out == "nothing found\n"
    assert main(["ls", "/nothing"]) == 0
    assert capsys.readouterr().out == "nothing found\n"


def test_ls_imports_the_plugins_of_a_module_or_a_config(capsys):
    assert main(["ls", "alad", "--plugin", PLUGIN]) == 0
    out = capsys.readouterr().out
    assert uris_of(out) == ["/objective/myexample/alad_discriminator", "/objective/myexample/alad_generator"]
    assert main(["ls", "/objective/myexample", "--config", example("alad")]) == 0
    assert uris_of(capsys.readouterr().out) == ["/objective/myexample/alad_discriminator",
                                                "/objective/myexample/alad_generator"]


def test_ls_reports_a_plugin_it_cannot_import(capsys):
    assert main(["ls", "alad", "--plugin", "no_such_plugin_module"]) == 1
    captured = capsys.readouterr()
    assert captured.err == ("1 problem found:\n  1. [plugin_import_failed] cannot import plugin no_such_plugin_module: "
                            "No module named 'no_such_plugin_module'\n")
    assert captured.out == "nothing found\n"


def test_docs_prints_the_reference_or_writes_it(tmp_path, capsys):
    assert main(["docs"]) == 0
    out = capsys.readouterr().out
    assert out == render() and out.startswith("# kalfa lego reference\n")
    assert [line for line in out.splitlines() if line.startswith("## ")] == ["## Catalog", "## Skeleton steps",
                                                                            "## Alias packs"]
    target = tmp_path / "reference.md"
    assert main(["docs", "--write", str(target)]) == 0
    assert capsys.readouterr().out == f"wrote {target}\n" and target.read_text() == render()


def test_docs_lists_the_legos_of_a_plugin(capsys):
    assert main(["docs", "--plugin", PLUGIN]) == 0
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("## ")] == ["## Catalog", "## Plugin legos",
                                                                            "## Skeleton steps", "## Alias packs"]
    assert (
        "| `/objective/myexample/alad_discriminator` |  | `(models, batch, criterion, latent_dim, rng=None)` |" in out)
    assert main(["docs", "--config", example("alad")]) == 0
    assert "| `/objective/myexample/alad_generator` |" in capsys.readouterr().out


def test_docs_reports_a_plugin_it_cannot_import(capsys):
    assert main(["docs", "--plugin", "no_such_plugin_module"]) == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("# kalfa lego reference\n") and "## Catalog" in captured.out
    assert captured.err == ("1 problem found:\n  1. [plugin_import_failed] cannot import plugin no_such_plugin_module: "
                            "No module named 'no_such_plugin_module'\n")


def test_contract_prints_the_contract_or_writes_it(tmp_path, capsys):
    assert main(["contract"]) == 0
    out = capsys.readouterr().out
    assert out == Contract.load().text() and out.startswith("# kalfa: the templates that take the surface")
    assert "\nwiring:\n" in out and "\nblocks:\n" in out
    target = tmp_path / "contract.yaml"
    assert main(["contract", "--write", str(target)]) == 0
    assert capsys.readouterr().out == f"wrote {target}\n" and target.read_text() == Contract.load().text()
    assert Contract.load(target).digest() == Contract.load().digest()
