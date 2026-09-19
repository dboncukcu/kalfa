import sys
from pathlib import Path

import pytest
from cirak.errors import Problem, Source
from cirak.registry import registry

from helpers import config_path, load_config, write_config
from kalfa.config import (
    import_plugins,
    load_surface,
    module_name,
    pack_tables,
    parse_param,
    parse_set,
    parse_sets,
    plugin_aliases,
    resolve_alias,
    resolve_rule_sets,
    written_config,
)
from kalfa.registration import lego
from kalfa.schema import Schema


REFERENCE = config_path("reference")

BASE_TEXT = """params:
  epochs: 9
training:
  epochs: $epochs$
extra:
  a: 1
  b: 2
"""

TOP_TEXT = """include: [base.yaml, /alias/kalfa/tabular]
extra:
  b: 3
record: runs/x
"""

PLUGIN_TEXT = """import kalfa


@kalfa.lego("/criterion/{name}/zero", alias="zero_loss", partial=True, description="always zero")
def zero(predictions, targets):
    return 0.0
"""


def write_plugin(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(PLUGIN_TEXT.format(name=name))
    return path


def zero_criterion(predictions, targets):
    return 0.0


def test_schema_lists_the_documented_sections_and_short_calls():
    assert Schema.sections == ("include", "plugins", "params", "alias", "seed", "device", "rng", "data", "model",
                               "metrics", "losses", "calibrate", "optimizers", "training", "generate", "plots",
                               "figures", "sweep", "record")
    assert Schema.required == ("data", "model", "losses", "training", "record")
    assert Schema.unresolved == ("include", "plugins", "params", "alias")
    assert Schema.builtin_variables == ("datetime",)
    assert Schema.short_calls == (("training", "turn"), ("training", "checkpoint"), ("data", "feed"),
                                  ("sweep", "strategy"), ("device",), ("rng",))
    assert Schema.lego_types() == ["criterion", "objective", "metric", "schedule", "init", "trigger", "generate",
                                   "pre", "preprocessor"]


def test_reference_config_loads_without_problems(workdir):
    surface = load_surface([REFERENCE], parse_sets([]))
    assert surface.problems == []
    assert surface.errors == []
    assert surface.paths == [REFERENCE]
    assert surface.overrides == []
    assert sorted(surface.data) == sorted({*load_config("reference"), "alias"} - {"include"})


def test_include_places_the_included_file_below_the_including_one(tmp_path):
    base = tmp_path / "base.yaml"
    top = tmp_path / "top.yaml"
    base.write_text(BASE_TEXT)
    top.write_text(TOP_TEXT)
    surface = load_surface([str(top)], parse_sets([]))
    assert surface.problems == []
    assert surface.data["extra"] == {"a": 1, "b": 3}
    assert surface.data["training"] == {"epochs": 9}
    assert surface.data["record"] == "runs/x"
    assert surface.data["alias"]["parquet"] == "/source/kalfa/parquet"
    assert surface.overrides == [(("extra", "b"), Source(str(top.resolve()), 3), Source(str(base.resolve()), 7))]
    assert [Path(loaded.file).name for loaded in surface.layer.walk()] == ["base.yaml", "base.yaml", "tabular.yaml",
                                                                          "top.yaml"]
    assert surface.layers_text() == "\n".join([
        "layers, bottom to top:",
        "  1. base.yaml (included by top.yaml)",
        "  2. base.yaml (included by tabular.yaml)",
        "  3. tabular.yaml (included by top.yaml)",
        "  4. top.yaml",
        f"       overrides extra.b ({base.resolve()}:7)",
    ])


def test_two_paths_are_one_layer_that_merges_strictly(tmp_path):
    base = tmp_path / "base.yaml"
    beside = tmp_path / "beside.yaml"
    base.write_text(BASE_TEXT)
    beside.write_text("extra:\n  c: 5\nseed: 3\n")
    surface = load_surface([str(base), str(beside)], parse_sets([]))
    assert surface.problems == []
    assert surface.data["extra"] == {"a": 1, "b": 2, "c": 5} and surface.data["seed"] == 3
    assert surface.overrides == []
    assert surface.source(("extra", "c")) == Source(str(beside.resolve()), 2)
    assert surface.source(("extra", "a")) == Source(str(base.resolve()), 6)
    clashing = tmp_path / "clashing.yaml"
    clashing.write_text("extra:\n  b: 5\n")
    twice = load_surface([str(base), str(clashing)], parse_sets([]))
    assert [problem.kind for problem in twice.problems] == ["merge_conflict"]
    assert twice.problems[0].message == f"extra.b is defined twice: {base.resolve()}:7 and {clashing.resolve()}:2"


def test_a_mapping_in_the_paths_is_a_layer_above_the_files(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(BASE_TEXT)
    surface = load_surface([str(base), {"extra": {"a": 5}}], parse_sets(["extra.b=7"]))
    assert surface.problems == []
    assert surface.data["extra"] == {"a": 5, "b": 7}
    assert surface.paths == [str(base), "<mapping>"]
    assert surface.overrides == [(("extra", "b"), Source("--set", 1), Source(str(base.resolve()), 7))]


def test_parse_sets_reads_dotted_paths_and_params_as_yaml():
    assert parse_sets(["training.epochs=5", "record=runs/x", "training.stop=[]", "device=cpu"], ["lr=1e-4"]) == [
        ("training.epochs", 5), ("record", "runs/x"), ("training.stop", []), ("device", "cpu"),
        ("params.lr", 0.0001)]
    assert parse_sets() == []
    assert parse_set("data.batch={size: 8, drop_last: true}") == ("data.batch", {"size": 8, "drop_last": True})
    assert parse_param("seed=3") == ("params.seed", 3)


def test_set_overrides_nested_paths_and_params_on_the_reference(workdir):
    sets = parse_sets(["training.epochs=5", "data.batch.size=32", "training.stop=[]"], ["lr=1e-4", "seed=3"])
    surface = load_surface([REFERENCE], sets)
    assert surface.problems == []
    assert surface.data["training"]["epochs"] == 5
    assert surface.data["training"]["stop"] == []
    assert surface.data["data"]["batch"] == {"size": 32, "eval_size": 256, "drop_last": True}
    assert surface.data["optimizers"]["main"]["params"]["lr"] == 0.0001
    assert surface.data["seed"] == 3
    assert surface.data["data"]["split"] == {"ratios": [0.7, 0.15, 0.15], "seed": 3}
    assert surface.source(("training", "epochs")) == Source("--set", 1)
    assert surface.source(("data", "batch", "size")) == Source("--set", 2)
    assert surface.overrides == [
        (("training", "epochs"), Source("--set", 1), Source(REFERENCE, 166)),
        (("training", "stop"), Source("--set", 3), Source(REFERENCE, 179)),
        (("data", "batch", "size"), Source("--set", 2), Source(REFERENCE, 27)),
        (("params", "lr"), Source("--set", 4), Source(REFERENCE, 6)),
        (("params", "seed"), Source("--set", 5), Source(REFERENCE, 5)),
    ]
    assert surface.layers_text() == "\n".join([
        "layers, bottom to top:",
        "  1. base.yaml (included by tabular.yaml)",
        "  2. tabular.yaml (included by reference.yaml)",
        "  3. reference.yaml",
        "  4. --set",
        f"       overrides training.epochs ({REFERENCE}:166)",
        f"       overrides training.stop ({REFERENCE}:179)",
        f"       overrides data.batch.size ({REFERENCE}:27)",
        f"       overrides params.lr ({REFERENCE}:6)",
        f"       overrides params.seed ({REFERENCE}:5)",
    ])


def test_parse_set_refuses_a_bare_name_that_is_no_top_level_key():
    with pytest.raises(ValueError) as caught:
        parse_set("foo=1")
    assert str(caught.value) == ("--set 'foo=1': 'foo' is not a top level key; write a dotted path from the root "
                                 "(--set training.epochs=5) or -p foo=1 for params.foo")


@pytest.mark.parametrize("text", ["training", "=3", ""])
def test_parse_set_refuses_a_malformed_assignment(text):
    with pytest.raises(ValueError) as caught:
        parse_set(text)
    assert str(caught.value) == f"--set expects PATH=VALUE, got {text!r}"


@pytest.mark.parametrize("text", ["lr", "=3"])
def test_parse_param_refuses_a_malformed_assignment(text):
    with pytest.raises(ValueError) as caught:
        parse_param(text)
    assert str(caught.value) == f"-p expects NAME=VALUE, got {text!r}"


def test_plugins_list_imports_a_module_next_to_the_config(workdir):
    plugin = write_plugin(workdir / "plugins", "surface_plug")
    config = load_config("reference")
    config["plugins"] = ["surface_plug"]
    config["losses"]["z"] = {"uri": "zero_loss", "output": "y_hat"}
    surface = load_surface([write_config(workdir / "plugged.yaml", config)], parse_sets([]))
    assert surface.problems == []
    assert sys.modules["surface_plug"].__file__ == str(plugin)
    assert registry.lookup("/criterion/surface_plug/zero").description == "always zero"
    assert registry.facts("/criterion/surface_plug/zero").kind == "criterion"
    assert plugin_aliases() == {"zero_loss": "/criterion/surface_plug/zero"}
    assert surface.aliases["zero_loss"] == "/criterion/surface_plug/zero"
    assert surface.data["losses"]["z"] == {"uri": "/criterion/surface_plug/zero", "output": "y_hat"}
    assert surface.data["plugins"] == ["surface_plug"]


def test_plugin_that_cannot_be_imported_is_a_problem(workdir):
    config = load_config("reference")
    config["plugins"] = ["zzz_not_a_module", 5]
    surface = load_surface([write_config(workdir / "plugged.yaml", config)], parse_sets([]))
    assert surface.problems == [
        Problem("error", "plugin_import_failed",
                "cannot import plugin zzz_not_a_module: No module named 'zzz_not_a_module'"),
        Problem("error", "plugin_import_failed", "plugin entries must be strings, got 5"),
    ]
    assert surface.errors == surface.problems


def test_import_plugins_reads_the_plugins_of_a_config_and_module_names(workdir):
    plugin = write_plugin(workdir, "import_plug")
    config = load_config("reference")
    config["plugins"] = ["import_plug"]
    path = write_config(workdir / "plugged.yaml", config)
    assert import_plugins([path], []) == []
    assert sys.modules["import_plug"].__file__ == str(plugin)
    problems = import_plugins([], ["zzz_not_a_module"])
    assert [(problem.kind, problem.message) for problem in problems] == [
        ("plugin_import_failed", "cannot import plugin zzz_not_a_module: No module named 'zzz_not_a_module'")]
    assert module_name("plugins/other_plug.py") == "other_plug"
    assert module_name("named_plug") == "named_plug"
    assert sys.path[0] == str(workdir)


def test_alias_section_adds_names_and_chains_resolve_to_the_pack(workdir):
    config = load_config("reference")
    config["alias"] = {"my_mse": "mse", "my_loss": "my_mse"}
    config["losses"]["a"] = {"uri": "my_loss", "output": "y_hat"}
    surface = load_surface([write_config(workdir / "aliased.yaml", config)], parse_sets([]))
    assert surface.problems == []
    assert surface.aliases["my_mse"] == "mse"
    assert surface.aliases["my_loss"] == "my_mse"
    assert surface.data["losses"]["a"]["uri"] == "/criterion/kalfa/mse"
    assert "alias" in surface.data
    assert "alias" not in written_config(surface.data)
    assert sorted(written_config(surface.data)) == sorted(set(config) - {"include", "alias"})


def test_alias_cycle_and_unknown_name_are_unknown_alias_problems(workdir):
    config = load_config("reference")
    config["alias"] = {"a": "b", "b": "a"}
    config["losses"]["a"] = {"uri": "a", "output": "y_hat"}
    config["losses"]["b"] = {"uri": "nope_alias", "output": "y_hat"}
    path = write_config(workdir / "cyclic.yaml", config)
    surface = load_surface([path], parse_sets([]))
    assert [(problem.kind, problem.message, problem.hint) for problem in surface.problems] == [
        ("unknown_alias", "'a' is not a known alias and does not start with /",
         "include an alias pack such as /alias/kalfa/tabular or write the full URI"),
        ("unknown_alias", "'nope_alias' is not a known alias and does not start with /",
         "include an alias pack such as /alias/kalfa/tabular or write the full URI"),
    ]
    assert surface.problems[0].file == path
    assert surface.data["losses"]["a"]["uri"] == "a"
    assert surface.data["losses"]["b"]["uri"] == "nope_alias"


def test_resolve_alias_follows_chains_and_refuses_cycles():
    assert resolve_alias("/criterion/kalfa/mse", {}) == "/criterion/kalfa/mse"
    assert resolve_alias("a", {"a": "b", "b": "/c/d/e"}) == "/c/d/e"
    assert resolve_alias("a", {"a": "b", "b": "a"}) is None
    assert resolve_alias("a", {"a": "a"}) is None
    assert resolve_alias("zz", {}) is None


def test_param_substitution_keeps_the_type_of_a_whole_value(workdir):
    surface = load_surface([REFERENCE], parse_sets([]))
    assert surface.data["training"]["epochs"] == 3
    assert surface.data["seed"] == 11
    assert surface.data["optimizers"]["main"]["params"]["lr"] == 0.001
    constraints = surface.raw["params"]["constraints"]
    assert surface.data["losses"]["total"]["params"]["constraints"] == constraints
    assert surface.data["losses"]["total"]["params"]["constraints"] is not constraints
    assert surface.data["model"]["models"]["lambdas"]["nodes"][0]["params"]["names"] == constraints
    assert surface.data["params"] == surface.raw["params"]


def test_param_substitution_inside_text_and_dotted_fields(workdir):
    config = load_config("reference")
    config["params"]["tag"] = "run"
    config["params"]["batch"] = {"size": 16}
    config["record"] = "runs/$tag$_$datetime$_$$"
    config["data"]["batch"]["size"] = "$batch.size$"
    config["data"]["batch"]["eval_size"] = "eval_$batch.size$"
    surface = load_surface([write_config(workdir / "textual.yaml", config)], parse_sets([]))
    assert surface.problems == []
    assert surface.data["record"] == "runs/run_$datetime$_$"
    assert surface.data["data"]["batch"]["size"] == 16
    assert surface.data["data"]["batch"]["eval_size"] == "eval_16"


def test_unknown_variable_and_unknown_field_are_problems(workdir):
    config = load_config("reference")
    config["training"]["epochs"] = "$nope$"
    config["data"]["batch"]["size"] = "$constraints.nope$"
    path = write_config(workdir / "unknown.yaml", config)
    surface = load_surface([path], parse_sets([]))
    assert [(problem.kind, problem.message, problem.hint, problem.file) for problem in surface.problems] == [
        ("unknown_variable", "param 'constraints' has no field 'nope'", None, path),
        ("unknown_variable", "unknown variable $nope$", "define it under params", path),
    ]
    assert surface.data["training"]["epochs"] == "$nope$"
    assert surface.data["data"]["batch"]["size"] == "$constraints.nope$"


def test_builtin_datetime_variable_is_kept_for_the_run(workdir):
    surface = load_surface([REFERENCE], parse_sets([]))
    assert surface.data["record"] == "runs/ref_$datetime$"
    assert surface.raw["record"] == "runs/ref_$datetime$"


def test_short_calls_turn_into_uris(workdir):
    config = load_config("reference")
    config["rng"] = "derived"
    config["training"]["turn"] = "alternating"
    config["training"]["checkpoint"] = "last"
    config["training"]["report"] = "last"
    config["sweep"] = {"strategy": "grid", "space": {"lr": [0.001, 0.01]},
                       "objective": {"monitor": "val/rmse_lin"}, "record": "runs/sweep"}
    surface = load_surface([write_config(workdir / "short.yaml", config)], parse_sets([]))
    assert surface.problems == []
    assert surface.data["training"]["turn"] == "/turn/kalfa/alternating"
    assert surface.data["training"]["checkpoint"] == "/checkpoint/kalfa/last"
    assert surface.data["data"]["feed"] == "/feed/kalfa/table"
    assert surface.data["device"] == "/device/kalfa/cpu"
    assert surface.data["rng"] == "/rng/kalfa/derived"
    assert surface.data["sweep"]["strategy"] == "/strategy/kalfa/grid"
    assert surface.data["training"]["report"] == "last"


def test_lego_calls_resolve_their_uri_everywhere(workdir):
    surface = load_surface([REFERENCE], parse_sets([]))
    assert surface.data["training"]["turn"] == {
        "uri": "/turn/kalfa/alternating",
        "params": {"order": ["main", "aux"], "steps": {"main": 1, "aux": 2}, "fresh_batch": False}}
    assert surface.data["training"]["checkpoint"] == {"uri": "/checkpoint/kalfa/best",
                                                      "params": {"monitor": "val/rmse_lin", "mode": "min"}}
    assert surface.data["data"]["source"] == {"uri": "/source/kalfa/parquet", "params": {"path": "reference.parquet"}}
    assert surface.data["data"]["transform"][1] == "num_1 > -2.5"
    assert surface.data["model"]["models"]["tower"]["init"] == {
        "weights": {"uri": "/init/torch/xavier"},
        "bias": {"uri": "/init/torch/zeros"},
        "scale": {"uri": "/init/torch/normal", "params": {"std": 0.05, "mean": 1.0}},
        "patterns": [{"match": "h_*", "weights": {"uri": "/init/torch/normal", "params": {"std": 0.02}}}]}
    assert surface.data["metrics"]["recon_lin"] == {"uri": "/metric/kalfa/recon_error", "output": "y_hat"}


def test_template_variables_stay_as_placeholders(workdir):
    surface = load_surface([REFERENCE], parse_sets([]))
    template = surface.data["model"]["templates"]["res_block"]
    assert template["nodes"]["f"] == {"uri": "/layer/kalfa/linear", "params": {"out_features": "$width$"},
                                      "inputs": ["u"]}
    assert template["nodes"]["d"] == {"uri": "/layer/torch/dropout", "params": {"p": "$drop$"}, "inputs": ["a"]}
    assert surface.data["model"]["models"]["tower"]["nodes"]["h"] == {
        "template": "res_block", "params": {"width": 16}, "repeat": 2, "inputs": ["stem"]}


def test_reference_params_resolve_lego_refs_inside_params(workdir):
    config = load_config("reference")
    config["losses"]["v"] = {"uri": "/objective/kalfa/vae",
                             "params": {"encoder": "tower", "decoder": "head_lin", "recon": "mse"}}
    surface = load_surface([write_config(workdir / "refs.yaml", config)], parse_sets([]))
    assert surface.problems == []
    assert surface.data["losses"]["v"]["params"] == {"encoder": "tower", "decoder": "head_lin",
                                                     "recon": "/criterion/kalfa/mse"}
    assert surface.data["losses"]["total"]["params"]["primary"] == "ws"
    assert surface.data["losses"]["total"]["params"]["multipliers"] == "lambdas"


def test_resolve_rule_sets_resolves_a_lego_valued_rule_effect(workdir):
    config = load_config("reference")
    config["losses"]["v"] = {"uri": "/objective/kalfa/vae",
                             "params": {"encoder": "tower", "decoder": "head_lin", "recon": "mse"}}
    config["training"]["rules"].append({"name": "to_huber", "when": {"uri": "after_epoch", "params": {"at": 2}},
                                        "set": {"v.recon": "huber", "ws.terms.mse_lin": 3.0}})
    surface = load_surface([write_config(workdir / "rules.yaml", config)], parse_sets([]))
    assert surface.problems == []
    assert surface.data["training"]["rules"][-1]["set"] == {"v.recon": "/criterion/kalfa/huber",
                                                            "ws.terms.mse_lin": 3.0}
    assert surface.data["training"]["rules"][2]["set"] == {"main.stem.lr": {"times": 0.5}, "ws.terms.mse_lin": 2.0}


def test_resolve_rule_sets_touches_only_lego_typed_loss_params():
    data = {"losses": {"v": {"uri": "/objective/kalfa/vae"}, "plain": "/criterion/kalfa/mse"},
            "training": {"rules": [
                {"name": "a", "set": {"v.recon": "huber", "v.w_rec": "huber", "main.lr": "huber", "loss": "huber",
                                      "v.recon.deep": "huber", "plain.x": "huber", "nope.recon": "huber"}},
                "not a rule", {"name": "b", "set": "not a mapping"}]}}
    resolve_rule_sets(data, {"huber": "/criterion/kalfa/huber"})
    assert data["training"]["rules"][0]["set"] == {
        "v.recon": "/criterion/kalfa/huber", "v.w_rec": "huber", "main.lr": "huber", "loss": "huber",
        "v.recon.deep": "huber", "plain.x": "huber", "nope.recon": "huber"}


def test_pack_tables_lists_every_alias_pack_with_its_included_names():
    tables = pack_tables()
    assert list(tables) == ["/alias/kalfa/base", "/alias/kalfa/lazy_tabular", "/alias/kalfa/tabular",
                            "/alias/kalfa/text", "/alias/kalfa/vision"]
    assert tables["/alias/kalfa/base"]["mse"] == "/criterion/kalfa/mse"
    assert tables["/alias/kalfa/tabular"]["parquet"] == "/source/kalfa/parquet"
    assert tables["/alias/kalfa/lazy_tabular"]["parquet"] == "/source/kalfa/parquet_stream"
    assert set(tables["/alias/kalfa/base"].items()) <= set(tables["/alias/kalfa/tabular"].items())


def test_plugin_aliases_lists_only_names_outside_the_std_set():
    assert plugin_aliases() == {}
    lego("/criterion/aliased_plug/zero", zero_criterion, alias="zero_loss", partial=True)
    assert plugin_aliases() == {"zero_loss": "/criterion/aliased_plug/zero"}
