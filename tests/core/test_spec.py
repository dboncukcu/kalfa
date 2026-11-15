import re

import pytest

from kalfa.core.errors import ProblemsError
from kalfa.core.spec import Layer, load, merge, parse_param, parse_set, parse_value, read_yaml, substitute


EXAMPLE = """
params:
  epochs: 100
  lr: 1.0e-3
  seed: 7
seed: $seed$
data:
  use: table
  file: housing.parquet
  split: [0.7, 0.15, 0.15]
  features: {"x*": standard_scaler}
  targets: {price: standard_scaler}
  batch: 128
models:
  net: {use: mlp, hidden: [128, 128], out: 1}
losses:
  mse: mse
  huber: {use: torch.nn.HuberLoss, delta: 1.0}
  mae: torch.nn.L1Loss
  log_cosh: mylegos.log_cosh
metrics:
  rmse: rmse
  mae: mae
optimizers:
  main: {use: adam, lr: $lr$, over: "@models.net", minimizes: "@losses.mse"}
train:
  epochs: $epochs$
  checkpoint: {watch: val/rmse}
  report: best
rules:
  to_huber:
    when: {use: after_epoch, at: 50}
    set: {optimizers.main.minimizes: "@losses.huber"}
  to_mae:
    after: to_huber
    when: {use: below, watch: train/huber, value: 0.01}
    set: {optimizers.main.minimizes: "@losses.mae"}
  stop:
    when: {use: plateau, watch: val/rmse, patience: 6}
    stop: true
plots:
  loss_curve: loss_curve
  pred_vs_true: pred_vs_true
record: runs/housing_$datetime$
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def codes(holder):
    problems = getattr(holder, "problems", holder)
    return [problem.code for problem in problems]


def labels(config):
    return [layer.label for layer in config.layers]


def test_parse_value_keeps_yaml_types():
    assert parse_value("20") == 20 and isinstance(parse_value("20"), int)
    assert parse_value("1.0e-3") == 0.001
    assert parse_value("1e-3") == 0.001
    assert parse_value("true") is True
    assert parse_value("false") is False
    assert parse_value("null") is None
    assert parse_value("'7'") == "7"
    assert parse_value('"a b"') == "a b"
    assert parse_value("yes") == "yes"
    assert parse_value("[a, b, 3]") == ["a", "b", 3]
    assert parse_value("{k: v, n: 1}") == {"k": "v", "n": 1}
    assert parse_value("@losses.huber") == "@losses.huber"
    assert parse_value("runs/x_$lr$") == "runs/x_$lr$"


def test_parse_set_and_parse_param_split_at_the_first_equals():
    assert parse_set("train.epochs=20") == ("train.epochs", 20)
    assert parse_set("models.net.hidden=[64, 64]") == ("models.net.hidden", [64, 64])
    assert parse_set("optimizers.main={use: sgd, lr: 0.1}") == ("optimizers.main", {"use": "sgd", "lr": 0.1})
    assert parse_set("optimizers.main.minimizes=@losses.huber") == ("optimizers.main.minimizes", "@losses.huber")
    assert parse_set("data.filter=a == 1") == ("data.filter", "a == 1")
    assert parse_set("a=") == ("a", None)
    assert parse_param("lr=1e-3") == ("lr", 0.001)
    assert parse_param("name=x") == ("name", "x")
    assert parse_param("flags={a: true}") == ("flags", {"a": True})


def test_bad_set_and_bad_param():
    with pytest.raises(ProblemsError) as info:
        parse_set("novalue")
    assert codes(info.value.problems) == ["bad_set"]
    with pytest.raises(ProblemsError) as info:
        parse_set("=5")
    assert codes(info.value.problems) == ["bad_set"]
    with pytest.raises(ProblemsError) as info:
        parse_set("x={a: [")
    assert codes(info.value.problems) == ["bad_set"]
    with pytest.raises(ProblemsError) as info:
        parse_param("novalue")
    assert codes(info.value.problems) == ["bad_param"]
    config = load(sets=["novalue"], params=["broken"])
    assert codes(config) == ["bad_set", "bad_param"]
    assert not config.ok


def test_load_substitutes_typed_and_embedded_params(tmp_path):
    top = write(tmp_path / "config.yaml", EXAMPLE)
    config = load([top])
    assert config.ok and config.problems == []
    data = config.data
    assert data["seed"] == 7 and isinstance(data["seed"], int)
    assert data["optimizers"]["main"]["lr"] == 0.001 and isinstance(data["optimizers"]["main"]["lr"], float)
    assert data["train"]["epochs"] == 100 and isinstance(data["train"]["epochs"], int)
    assert re.fullmatch(r"runs/housing_\d{8}_\d{6}", data["record"])
    assert config.params == {"epochs": 100, "lr": 0.001, "seed": 7}
    assert "params" not in data
    assert data["data"]["features"] == {"x*": "standard_scaler"}
    assert data["losses"]["huber"] == {"use": "torch.nn.HuberLoss", "delta": 1.0}
    assert data["rules"]["to_mae"]["after"] == "to_huber"
    assert data["rules"]["to_huber"]["set"] == {"optimizers.main.minimizes": "@losses.huber"}
    assert data["rules"]["stop"]["stop"] is True
    assert config.plugins == []
    assert config.overrides == {}
    assert labels(config) == [top]
    assert config.directories == [str(tmp_path.resolve())]


def test_includes_layer_below_the_including_file(tmp_path):
    base = tmp_path / "base.yaml"
    write(base, "params: {lr: 0.1}\ntrain: {epochs: 3, report: best}\nrecord: runs/a\n")
    top = write(tmp_path / "config.yaml",
                "include: [base.yaml]\ntrain: {epochs: 5}\nlosses: {mse: {use: mse, factor: $lr$}}\n")
    config = load([top])
    assert config.ok
    assert config.data["train"] == {"epochs": 5, "report": "best"}
    assert config.data["record"] == "runs/a"
    assert config.data["losses"]["mse"] == {"use": "mse", "factor": 0.1}
    assert "include" not in config.data
    assert config.overrides == {"train.epochs": top}
    assert labels(config) == ["base.yaml", top]
    assert config.layers[0].file == str(base.resolve())
    assert config.layers[1].file == str((tmp_path / "config.yaml").resolve())
    assert config.layers[1].includes == [str(base.resolve())]
    assert config.layers[0].includes == []


def test_include_paths_are_relative_to_the_including_file(tmp_path):
    write(tmp_path / "common" / "defaults.yaml", "seed: 1\ntrain: {epochs: 1, report: last}\n")
    write(tmp_path / "sub" / "base.yaml", "include: [../common/defaults.yaml]\ntrain: {epochs: 2}\n")
    top = write(tmp_path / "sub" / "config.yaml", "include: [base.yaml]\nrecord: runs/x\n")
    config = load([top])
    assert config.ok
    assert config.data == {"seed": 1, "train": {"epochs": 2, "report": "last"}, "record": "runs/x"}
    assert config.overrides == {"train.epochs": "base.yaml"}
    assert labels(config) == ["../common/defaults.yaml", "base.yaml", top]
    assert config.directories == [str((tmp_path / "sub").resolve()), str((tmp_path / "common").resolve())]


def test_include_cycle_is_reported_once_and_the_rest_still_loads(tmp_path):
    write(tmp_path / "a.yaml", "include: [b.yaml]\nseed: 1\n")
    write(tmp_path / "b.yaml", "include: [a.yaml]\nrecord: runs/b\n")
    config = load([str(tmp_path / "a.yaml")])
    assert codes(config) == ["include_cycle"]
    assert "a.yaml -> b.yaml -> a.yaml" in config.problems[0].message
    assert config.data == {"seed": 1, "record": "runs/b"}


def test_missing_include_is_file_not_found(tmp_path):
    top = write(tmp_path / "config.yaml", "include: [nope.yaml]\nseed: 3\n")
    config = load([top])
    assert codes(config) == ["file_not_found"]
    assert "nope.yaml" in config.problems[0].message
    assert config.problems[0].hint == f"included by {top}"
    assert config.data == {"seed": 3}


def test_missing_top_level_file(tmp_path):
    config = load([str(tmp_path / "nope.yaml")])
    assert codes(config) == ["file_not_found"]
    assert config.problems[0].hint is None
    assert config.data == {} and config.layers == [] and config.directories == []


def test_yaml_error_names_the_line(tmp_path):
    top = write(tmp_path / "broken.yaml", "a: 1\nb: c: d\n")
    config = load([top])
    assert codes(config) == ["yaml_error"]
    assert re.search(r"broken\.yaml:2:", config.problems[0].message)


def test_duplicate_key_inside_a_file(tmp_path):
    top = write(tmp_path / "config.yaml", "record: runs/a\nrecord: runs/b\n")
    config = load([top])
    assert codes(config) == ["duplicate_key"]
    assert re.search(r"config\.yaml:2:", config.problems[0].message)


def test_top_level_must_be_a_mapping(tmp_path):
    listed = write(tmp_path / "list.yaml", "- a\n- b\n")
    text = write(tmp_path / "text.yaml", "just text\n")
    assert codes(load([listed])) == ["not_a_mapping"]
    assert codes(load([text])) == ["not_a_mapping"]
    assert codes(load(mappings=[["not", "a", "mapping"]])) == ["not_a_mapping"]


def test_empty_file_is_an_empty_mapping(tmp_path):
    top = write(tmp_path / "empty.yaml", "")
    config = load([top])
    assert config.ok and config.data == {}
    assert labels(config) == [top] and config.layers[0].data == {}


def test_a_call_replaces_the_whole_value_and_a_plain_mapping_merges(tmp_path):
    write(tmp_path / "base.yaml", "losses: {mse: {use: scaled, factor: 1, extra: 2}}\n")
    whole = write(tmp_path / "whole.yaml", "include: [base.yaml]\nlosses: {mse: {use: mse}}\n")
    partial = write(tmp_path / "partial.yaml", "include: [base.yaml]\nlosses: {mse: {factor: 2}}\n")
    replaced = load([whole])
    assert replaced.data["losses"]["mse"] == {"use": "mse"}
    assert replaced.overrides == {"losses.mse": whole}
    merged = load([partial])
    assert merged.data["losses"]["mse"] == {"use": "scaled", "factor": 2, "extra": 2}
    assert merged.overrides == {"losses.mse.factor": partial}


def test_lists_and_scalars_replace(tmp_path):
    write(tmp_path / "base.yaml", "models: {net: {use: mlp, hidden: [128, 128]}}\nseed: 1\ntrain: {epochs: 2}\n")
    top = write(tmp_path / "config.yaml", "include: [base.yaml]\nmodels: {net: {hidden: [64]}}\nseed: 2\ntrain: 3\n")
    config = load([top])
    assert config.data["models"]["net"] == {"use": "mlp", "hidden": [64]}
    assert config.data["seed"] == 2
    assert config.data["train"] == 3
    assert config.overrides == {"models.net.hidden": top, "seed": top, "train": top}


def test_mappings_sets_and_params_layer_in_order(tmp_path):
    top = write(tmp_path / "config.yaml", "params: {lr: 0.1}\ntrain: {epochs: 3}\nrecord: runs/$lr$\n")
    config = load([top], sets=["train.epochs=5"], params=["lr=0.2"],
                  mappings=[{"train": {"epochs": 4}, "seed": 1}, {"seed": 2}])
    assert config.ok
    assert config.data == {"train": {"epochs": 5}, "record": "runs/0.2", "seed": 2}
    assert config.params == {"lr": 0.2}
    assert labels(config) == [top, "mapping", "mapping", "--set", "-p"]
    assert config.overrides == {"train.epochs": "--set", "seed": "mapping", "params.lr": "-p"}
    assert config.layers[-2] == Layer("--set", {"train": {"epochs": 5}})
    assert config.layers[-1] == Layer("-p", {"params": {"lr": 0.2}})


def test_sets_accept_pairs_and_text_and_only_changed_leaves_are_overrides(tmp_path):
    top = write(tmp_path / "config.yaml", "train: {epochs: 3}\nseed: 1\n")
    config = load([top], sets=[("train.epochs", 7), "train.report=last", "seed=1"])
    assert config.data == {"train": {"epochs": 7, "report": "last"}, "seed": 1}
    assert config.overrides == {"train.epochs": "--set"}


def test_set_on_a_call_replaces_it_and_a_leaf_set_keeps_the_rest(tmp_path):
    top = write(tmp_path / "config.yaml", "optimizers: {main: {use: adam, lr: 0.001, over: '@models.net'}}\n")
    leaf = load([top], sets=["optimizers.main.lr=0.01"])
    assert leaf.data["optimizers"]["main"] == {"use": "adam", "lr": 0.01, "over": "@models.net"}
    assert leaf.overrides == {"optimizers.main.lr": "--set"}
    whole = load([top], sets=["optimizers.main={use: sgd, lr: 0.1}"])
    assert whole.data["optimizers"]["main"] == {"use": "sgd", "lr": 0.1}
    assert whole.overrides == {"optimizers.main": "--set"}


def test_params_merge_across_layers_and_later_layers_win(tmp_path):
    write(tmp_path / "base.yaml", "params: {lr: 0.1, name: a, width: 8}\n")
    top = write(tmp_path / "config.yaml", "include: [base.yaml]\nparams: {lr: 0.2}\nrecord: runs/$name$_$width$\n")
    config = load([top], params=[("name", "b")])
    assert config.params == {"lr": 0.2, "name": "b", "width": 8}
    assert config.data == {"record": "runs/b_8"}
    assert config.overrides == {"params.lr": top, "params.name": "-p"}
    assert "params" not in config.data


def test_set_can_write_a_param_and_substitution_runs_after_every_layer(tmp_path):
    top = write(tmp_path / "config.yaml", "params: {name: x}\nrecord: runs/a\n")
    config = load([top], sets=["record=runs/$name$", "params.name=y"])
    assert config.data["record"] == "runs/y"
    assert config.params == {"name": "y"}


def test_plugins_pool_without_duplicates_in_first_seen_order(tmp_path):
    write(tmp_path / "base.yaml", "plugins: [mylegos, other]\n")
    top = write(tmp_path / "config.yaml", "include: [base.yaml]\nplugins: [mylegos, third]\nseed: 1\n")
    config = load([top], mappings=[{"plugins": "fourth"}, {"plugins": ["other"]}])
    assert config.ok
    assert config.plugins == ["mylegos", "other", "third", "fourth"]
    assert config.data == {"seed": 1}


def test_layers_keep_data_as_loaded_and_loader_keys_never_reach_data(tmp_path):
    write(tmp_path / "base.yaml", "seed: 1\n")
    top = write(tmp_path / "config.yaml",
                "include: [base.yaml]\nparams: {lr: 0.1}\nplugins: [mylegos]\ntrain: {epochs: $lr$}\n")
    config = load([top])
    assert config.layers[-1].data == {"include": ["base.yaml"], "params": {"lr": 0.1}, "plugins": ["mylegos"],
                                      "train": {"epochs": "$lr$"}}
    assert config.data == {"seed": 1, "train": {"epochs": 0.1}}
    assert set(config.data).isdisjoint({"include", "params", "plugins"})


def test_bad_loader_keys(tmp_path):
    assert codes(load([write(tmp_path / "a.yaml", "include: 5\n")])) == ["bad_include"]
    assert codes(load([write(tmp_path / "b.yaml", "include: [1]\n")])) == ["bad_include"]
    assert codes(load([write(tmp_path / "c.yaml", "params: [1]\n")])) == ["bad_params"]
    assert codes(load([write(tmp_path / "d.yaml", "plugins: {a: 1}\n")])) == ["bad_plugins"]
    assert codes(load([write(tmp_path / "e.yaml", "plugins: [1]\n")])) == ["bad_plugins"]


def test_datetime_is_computed_once_and_double_dollar_is_a_literal(tmp_path):
    top = write(tmp_path / "config.yaml",
                "record: runs/$datetime$\nnote: at $datetime$ costs $$5\nstamp: $datetime$\nsign: $$\n")
    config = load([top])
    assert config.ok
    stamp = config.data["stamp"]
    assert re.fullmatch(r"\d{8}_\d{6}", stamp)
    assert config.data["record"] == f"runs/{stamp}"
    assert config.data["note"] == f"at {stamp} costs $5"
    assert config.data["sign"] == "$"


def test_unknown_param_names_the_path_and_leaves_the_text(tmp_path):
    top = write(tmp_path / "config.yaml", "models: {net: {use: mlp, hidden: [$h$, 8]}}\nrecord: runs/$nope$\n")
    config = load([top])
    assert codes(config) == ["unknown_param", "unknown_param"]
    assert [problem.path for problem in config.problems] == ["models.net.hidden[0]", "record"]
    assert config.data["models"]["net"]["hidden"] == ["$h$", 8]
    assert config.data["record"] == "runs/$nope$"


def test_nested_param_lookup(tmp_path):
    top = write(tmp_path / "config.yaml",
                "params: {opt: {lr: 0.01, name: adam}}\na: $opt.lr$\nb: $opt.name$_x\nc: $opt$\nd: $opt.missing$\n")
    config = load([top])
    assert codes(config) == ["unknown_param"]
    assert config.problems[0].path == "d"
    assert config.data["a"] == 0.01
    assert config.data["b"] == "adam_x"
    assert config.data["c"] == {"lr": 0.01, "name": "adam"}
    assert config.data["d"] == "$opt.missing$"


def test_params_may_use_other_params_and_cycles_are_reported(tmp_path):
    top = write(tmp_path / "config.yaml",
                "params: {base: 8, width: $base$, name: w$width$, sizes: [$base$, $width$]}\n"
                "models: {net: {use: mlp, hidden: $sizes$, tag: $name$}}\n")
    config = load([top])
    assert config.ok
    assert config.params == {"base": 8, "width": 8, "name": "w8", "sizes": [8, 8]}
    assert config.data["models"]["net"] == {"use": "mlp", "hidden": [8, 8], "tag": "w8"}
    cyclic = load([write(tmp_path / "cyclic.yaml", "params: {a: $b$, b: $a$}\nx: $a$\n")])
    assert codes(cyclic) == ["param_cycle"]
    assert "a -> b -> a" in cyclic.problems[0].message
    own = load([write(tmp_path / "own.yaml", "params: {a: x$a$}\n")])
    assert codes(own) == ["param_cycle"]


def test_merge_records_only_changed_leaves_and_copies_incoming_values():
    base = {"train": {"epochs": 3, "report": "best"}, "seed": 1, "losses": {"mse": {"use": "mse", "w": 1}}}
    incoming = {"train": {"epochs": 3, "track": ["x"]}, "seed": 2, "losses": {"mse": {"use": "mae"}}}
    overrides = {}
    problems = []
    result = merge(base, incoming, "", problems, overrides, "top")
    assert result is base
    assert base == {"train": {"epochs": 3, "report": "best", "track": ["x"]}, "seed": 2,
                    "losses": {"mse": {"use": "mae"}}}
    assert overrides == {"seed": "top", "losses.mse": "top"}
    assert problems == []
    incoming["train"]["track"].append("y")
    assert base["train"]["track"] == ["x"]
    merge(base, {"seed": True}, "", problems, overrides, "later")
    assert overrides["seed"] == "later"


def test_substitute_returns_typed_copies_and_keeps_the_input():
    params = {"hidden": [128, 128], "lr": 0.001, "flag": True, "nothing": None}
    data = {"net": {"hidden": "$hidden$", "lr": "$lr$", "tag": "lr=$lr$", "flag": "$flag$", "n": "$nothing$"}}
    problems = []
    out = substitute(data, params, problems)
    assert problems == []
    assert out["net"]["hidden"] == [128, 128] and out["net"]["hidden"] is not params["hidden"]
    assert out["net"]["lr"] == 0.001
    assert out["net"]["tag"] == "lr=0.001"
    assert out["net"]["flag"] is True
    assert out["net"]["n"] is None
    assert data["net"]["hidden"] == "$hidden$"
    out["net"]["hidden"].append(1)
    assert params["hidden"] == [128, 128]


def test_mapping_includes_resolve_against_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", "train: {epochs: 1, report: best}\n")
    config = load(mappings=[{"include": ["base.yaml"], "train": {"epochs": 2}}])
    assert config.ok
    assert config.data == {"train": {"epochs": 2, "report": "best"}}
    assert labels(config) == ["base.yaml", "mapping"]
    assert config.overrides == {"train.epochs": "mapping"}
    assert config.directories == [str(tmp_path.resolve())]


def test_a_file_enters_once_even_when_passed_and_included(tmp_path):
    base = write(tmp_path / "base.yaml", "seed: 1\n")
    top = write(tmp_path / "config.yaml", "include: [base.yaml, base.yaml]\nrecord: runs/a\n")
    config = load([base, top])
    assert config.ok
    assert labels(config) == [base, top]
    assert config.layers[1].includes == [str((tmp_path / "base.yaml").resolve())] * 2
    assert config.data == {"seed": 1, "record": "runs/a"}


def test_read_yaml_returns_plain_data_or_raises(tmp_path):
    path = write(tmp_path / "a.yaml", "n: 1\nf: 1.5\nb: true\ns: text\nday: 2026-11-07\nxs: [1, 2]\nm: {k: v}\n")
    assert read_yaml(path) == {"n": 1, "f": 1.5, "b": True, "s": "text", "day": "2026-11-07", "xs": [1, 2],
                               "m": {"k": "v"}}
    assert read_yaml(write(tmp_path / "empty.yaml", "")) is None
    with pytest.raises(ProblemsError) as info:
        read_yaml(tmp_path / "nope.yaml")
    assert codes(info.value.problems) == ["file_not_found"]
    with pytest.raises(ProblemsError) as info:
        read_yaml(write(tmp_path / "broken.yaml", "a: [1,\n"))
    assert codes(info.value.problems) == ["yaml_error"]


def test_config_errors_and_ok(tmp_path):
    top = write(tmp_path / "config.yaml", "record: runs/$nope$\n")
    config = load([top])
    assert not config.ok
    assert codes(config.errors) == ["unknown_param"]
    assert str(config.errors[0]).startswith("error unknown_param: record: unknown param $nope$")
