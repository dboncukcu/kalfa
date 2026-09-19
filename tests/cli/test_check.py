
import pytest
from ruamel.yaml import YAML

from helpers import config_path, load_config, write_config
from kalfa.cli import main, version_text


REFERENCE = config_path("reference")


def line_of(path, needle):
    return next(number for number, line in enumerate(path.read_text().splitlines(), 1) if needle in line)


def joined(text):
    return " ".join(line.strip() for line in text.splitlines())


def test_check_finds_no_problem_in_the_reference_config(workdir, capsys):
    assert main(["check", REFERENCE]) == 0
    assert capsys.readouterr().out == f"{version_text()}\nno problems found\n"


def test_check_renders_the_problems_of_a_broken_copy(workdir, capsys):
    config = load_config("reference")
    config["losses"]["mse_lin"]["uri"] = "nonsense"
    broken = workdir / "broken.yaml"
    write_config(broken, config)
    assert main(["check", str(broken)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines() == [
        version_text(), "1 problem found:",
        f"  1. [unknown_alias] 'nonsense' is not a known alias and does not start with / "
        f"({broken}:{line_of(broken, 'uri: nonsense')}); include an alias pack such as /alias/kalfa/tabular or write "
        "the full URI"]
    config = load_config("reference")
    config["training"]["checkpoint"] = "last"
    config["training"]["targets"]["tail_logit"] = "ghost"
    write_config(broken, config)
    assert main(["check", str(broken)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[1] == "7 problems found:"
    assert (f"  1. [report_mismatch] training.report must be last or a checkpoint the policy writes; 'best' is not "
            f"among ['last'] ({broken}:{line_of(broken, 'report: best')}); report: best needs checkpoint: "
            "{uri: best, params: {monitor: ...}}") in out.splitlines()
    named = [line.split("] ", 1)[1].split(" ")[0] for line in out.splitlines()[3:]]
    assert named == ["training.targets.tail_logit", "losses.bce.target", "losses.bce_pos.target",
                     "metrics.auroc.target", "metrics.acc.target", "metrics.ap_every.target"]
    assert all("[target_not_a_field]" in line and "ghost" in line for line in out.splitlines()[3:])


def test_check_layers_prints_the_tree_and_the_overridden_leaves(workdir, capsys):
    write_config(workdir / "lower.yaml", load_config("reference"))
    top = workdir / "top.yaml"
    top.write_text("include: [lower.yaml]\ntraining:\n  epochs: 2\n")
    assert main(["check", str(top), "--layers", "-p", "lr=0.5", "--set", "training.epochs=3"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[1:6] == ["layers, bottom to top:", "  1. base.yaml (included by tabular.yaml)",
                                     "  2. tabular.yaml (included by "
                                     "lower.yaml)", "  3. lower.yaml (included by top.yaml)",
                                     "  4. top.yaml"]
    assert f"       overrides training.epochs ({workdir / 'lower.yaml'}:" in out
    assert "  5. --set" in out and f"       overrides training.epochs ({top}:3)" in out
    assert f"       overrides params.lr ({workdir / 'lower.yaml'}:" in out
    assert out.endswith("no problems found\n")


def test_check_measure_counts_the_sets_after_the_transforms(workdir, capsys):
    assert main(["check", REFERENCE, "--measure"]) == 0
    out = capsys.readouterr().out
    text = joined(out)
    assert "no problems found measured the data block:  reference.parquet 2 000 rows ─→ 6 transforms" in text
    assert "─→ split random  0.7 / 0.15 / 0.15  seed=11" in text
    assert ("─→ fitted std, target_std, robust, onehot, ordinal, squash, to_logit, impute, fill0, abs_train, to_float "
            "on train ─→ table feed ─→ 3 loaders, batch 64") in text
    assert out.endswith("\nsets after filters: train 1 250  ·  valid 273  ·  test 278\n")


def test_check_takes_set_and_param_overrides(workdir, capsys):
    assert main(["check", REFERENCE, "-p", "lr=0.5", "--set", "device=cpu", "--set", "training.epochs=1"]) == 0
    capsys.readouterr()
    assert main(["check", REFERENCE, "--set", "training.epochs=abc"]) == 1
    assert capsys.readouterr().out.splitlines()[1:] == [
        "1 problem found:", "  1. [invalid_value] training.epochs must be a non negative integer (--set:1)"]
    with pytest.raises(SystemExit) as failure:
        main(["check", REFERENCE, "--set", "lr=0.5"])
    assert failure.value.code == 2
    assert capsys.readouterr().err == ("kalfa: error: --set 'lr=0.5': 'lr' is not a top level key; write a dotted "
                                       "path from the root (--set training.epochs=5) or -p lr=0.5 for params.lr\n")
    with pytest.raises(SystemExit) as failure:
        main(["check", REFERENCE, "--set", "nonsense"])
    assert failure.value.code == 2
    assert capsys.readouterr().err == "kalfa: error: --set expects PATH=VALUE, got 'nonsense'\n"
    with pytest.raises(SystemExit) as failure:
        main(["check", REFERENCE, "-p", "nonsense"])
    assert failure.value.code == 2
    assert capsys.readouterr().err == "kalfa: error: -p expects NAME=VALUE, got 'nonsense'\n"


def test_check_recipe_and_dump_print_yaml_after_a_separator(workdir, capsys):
    assert main(["check", REFERENCE, "--recipe", "--dump"]) == 0
    out = capsys.readouterr().out
    header, recipe, dump = out.split("---\n")
    assert header == f"{version_text()}\nno problems found\n"
    assert list(YAML(typ="safe").load(recipe)) == ["losses", "metrics", "triggers", "plots", "calibrate", "checkpoint",
                                                   "blocks", "flow"]
    document = YAML(typ="safe").load(dump)
    assert list(document) == ["components", "blocks", "flow"]
    assert sorted(document["components"]) == ["calibrate", "checkpoint", "losses", "metrics", "plots", "triggers"]
    assert "# implicit: device" in dump and "  mse_lin: {uri: /adapter/kalfa/criterion, params: {criterion: " in recipe


def test_check_recipe_and_dump_have_nothing_to_print_without_a_shaped_config(workdir, capsys):
    assert main(["check", REFERENCE, "--set", "training.epochs=abc", "--recipe", "--dump"]) == 1
    captured = capsys.readouterr()
    assert "---" not in captured.out
    assert captured.err == ("the config could not be shaped, no recipe\n"
                            "the flow could not be compiled, nothing to dump\n")


def test_check_runs_against_a_written_contract(workdir, capsys):
    copy = workdir / "contract.yaml"
    assert main(["contract", "--write", str(copy)]) == 0
    capsys.readouterr()
    assert main(["check", REFERENCE, "--contract", str(copy)]) == 0
    assert capsys.readouterr().out == f"{version_text()}\nno problems found\n"
    assert main(["check", REFERENCE, "--contract", str(workdir / "missing.yaml")]) == 1
    assert capsys.readouterr().err == (f"contract {workdir / 'missing.yaml'} does not exist; kalfa contract --write "
                                       "writes the default one\n")


def test_check_reads_a_record_in_place_of_the_config(reference, capsys):
    assert main(["check", reference.record]) == 0
    assert capsys.readouterr().out == f"{version_text()}\nno problems found\n"


def test_check_prepared_reads_the_sizes_of_the_manifest(workdir, capsys):
    assert main(["prepare", REFERENCE, "--out", "prepared"]) == 0
    assert capsys.readouterr().out == "prepared train 1331, valid 298, test 298: prepared\n"
    assert sorted(path.name for path in (workdir / "prepared").iterdir()) == [
        "data.json", "fitted", "manifest.json", "test.parquet", "train.parquet", "valid.parquet"]
    assert main(["check", REFERENCE, "--prepared", "prepared", "--measure"]) == 0
    out = capsys.readouterr().out
    assert "─→ split prepared  path=prepared" in joined(out)
    assert out.endswith("\nsets after filters: train 1 331  ·  valid 298  ·  test 298\n")
    assert main(["check", REFERENCE, "--prepared", "prepared", "-p", "seed=9"]) == 1
    assert capsys.readouterr().out.splitlines()[1:] == [
        "1 problem found:",
        "  1. [prepared_mismatch] prepared was prepared from another data section; prepare it again from this config"]
    assert main(["check", REFERENCE, "--prepared", "elsewhere"]) == 1
    assert capsys.readouterr().err == ("elsewhere is no prepared directory; kalfa prepare cfg.yaml --out elsewhere "
                                       "writes one\n")


@pytest.mark.xfail(strict=True, reason="bug: prepare --out writes into a directory that exists although its help "
                                       "says the directory must not exist")
def test_prepare_refuses_a_directory_that_exists(workdir, capsys):
    assert main(["prepare", REFERENCE, "--out", "prepared"]) == 0
    capsys.readouterr()
    assert main(["prepare", REFERENCE, "--out", "prepared"]) == 1
