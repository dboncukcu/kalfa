"""kalfa's own check rules: one small invalid config per problem kind."""

import pytest

import kalfa  # noqa: F401
from helpers import delete, minimal, write_config
from kalfa.api import check
from kalfa.check import sets_text
from kalfa.config import load_surface, parse_sets


def kinds_of(tmp_path, config, name="cfg.yaml", sets=None):
    path = write_config(tmp_path / name, config)
    prepared = check([path], sets)
    return prepared, [problem.kind for problem in prepared.problems]


def errors_of(prepared):
    return [problem.kind for problem in prepared.errors]


def test_minimal_config_checks_clean(workdir):
    prepared, kinds = kinds_of(workdir, minimal())
    assert kinds == []
    assert prepared.sizes == {"train": 1400, "valid": 300, "test": 300}
    assert sets_text(prepared.sizes) == "sets (before filters, from the file header): train 1400, valid 300, test 300"
    assert ("training.epochs.body.turn", "device", "device") in prepared.implicit


def test_unknown_key(workdir):
    config = minimal()
    config["data"]["batches"] = 3
    config["trainng"] = {}
    prepared, kinds = kinds_of(workdir, config)
    assert kinds.count("unknown_key") == 2
    assert any("did you mean" in (problem.hint or "") for problem in prepared.problems)


def test_missing_key(workdir):
    prepared, kinds = kinds_of(workdir, delete(minimal(), "training", "report"))
    assert "missing_key" in kinds
    prepared, kinds = kinds_of(workdir, delete(minimal(), "losses"))
    assert "missing_key" in kinds


def test_kind_mismatch(workdir):
    config = minimal()
    config["losses"]["r"] = {"uri": "rmse"}
    config["metrics"]["o"] = {"uri": "/objective/kalfa/fake"}
    config["data"]["feed"] = "parquet"
    prepared, kinds = kinds_of(workdir, config)
    assert kinds.count("kind_mismatch") == 2
    assert "unknown_uri" in kinds


def test_unresolved_ref(workdir):
    config = minimal()
    config["training"]["loss"] = "nope"
    config["data"]["fields"]["price"] = {"target": True, "preprocessors": ["ghost"]}
    config["training"]["predicts"] = "other"
    config["training"]["rules"] = [{"name": "a", "when": {"uri": "after_epoch", "params": {"at": 1}},
                                    "set": {"loss": "mse"}, "after": "zzz"}]
    prepared, kinds = kinds_of(workdir, config)
    assert kinds.count("unresolved_ref") == 4


def test_unknown_alias_without_the_pack(workdir):
    config = minimal()
    config["include"] = []
    prepared, kinds = kinds_of(workdir, config)
    assert "unknown_alias" in kinds


def test_set_target_and_set_value(workdir):
    config = minimal()
    config["training"]["rules"] = [
        {"name": "a", "when": {"uri": "after_epoch", "params": {"at": 1}}, "set": {"nothing": 1}},
        {"name": "b", "when": {"uri": "after_epoch", "params": {"at": 1}}, "set": {"loss": "ghost"}},
        {"name": "c", "when": {"uri": "after_epoch", "params": {"at": 1}}, "set": {"mse.delta": 1.0}},
        {"name": "d", "when": {"uri": "after_epoch", "params": {"at": 1}}, "set": {"model.trainable": "yes"}},
        {"name": "e", "when": {"uri": "after_epoch", "params": {"at": 1}}, "set": {"ghost.loss": "mse"}},
        {"name": "f", "when": {"uri": "after_epoch", "params": {"at": 1}}, "set": {"model.lr": 0.1}},
    ]
    prepared, kinds = kinds_of(workdir, config)
    assert kinds.count("set_target") == 2
    assert kinds.count("set_value") == 3


def test_glob_ambiguous_and_column_missing(workdir):
    config = minimal()
    config["data"]["fields"] = {"x?": {}, "?1": {}, "price": {"target": True}, "zzz*": {}}
    prepared, kinds = kinds_of(workdir, config)
    assert "glob_ambiguous" in kinds and "column_missing" in kinds


def test_set_missing_and_test_monitor(workdir):
    config = minimal()
    config["data"]["split"] = {"ratios": [0.8, 0.0, 0.2], "seed": 1}
    config["training"]["stop"] = [{"uri": "plateau", "params": {"monitor": "val/rmse", "patience": 2}}]
    config["training"]["checkpoint"] = {"uri": "best", "params": {"monitor": "test/rmse"}}
    config["training"]["report"] = "best"
    prepared, kinds = kinds_of(workdir, config)
    assert "set_missing" in kinds and "test_monitor" in kinds
    assert prepared.sizes == {"train": 1600, "valid": 0, "test": 400}
    assert sets_text(prepared.sizes) == "sets (before filters, from the file header): train 1600, valid none, test 400"


def test_unused_optimizer(workdir):
    config = minimal()
    config["optimizers"] = {"spare": {"uri": "adam", "loss": "mse"}}
    prepared, kinds = kinds_of(workdir, config)
    assert "unused_optimizer" in kinds


def test_predicts_required(workdir):
    config = minimal()
    config["model"] = {"models": {
        "a": {"optimizer": "main", "inputs": ["x"], "outputs": ["y"],
              "nodes": [{"uri": "linear", "params": {"out_features": 1}}]},
        "b": {"optimizer": "main", "inputs": ["x"], "outputs": ["y"],
              "nodes": [{"uri": "linear", "params": {"out_features": 1}}]}}}
    config["optimizers"] = {"main": {"uri": "adam", "loss": "mse"}}
    del config["training"]["loss"]
    prepared, kinds = kinds_of(workdir, config)
    assert "predicts_required" in kinds


def test_report_mismatch(workdir):
    config = minimal()
    config["training"]["report"] = "best"
    prepared, kinds = kinds_of(workdir, config)
    assert "report_mismatch" in kinds


def test_needs_grad_set(workdir):
    import cirak

    kalfa.lego("/objective/test/grad", lambda models, batch: 0.0, partial=True, needs_grad=True)
    config = minimal()
    config["losses"]["g"] = {"uri": "/objective/test/grad"}
    prepared, kinds = kinds_of(workdir, config)
    assert "needs_grad_set" in kinds
    config["losses"]["g"] = {"uri": "/objective/test/grad", "sets": ["train"]}
    prepared, kinds = kinds_of(workdir, config)
    assert "needs_grad_set" not in kinds


def test_model_name_dot_and_reserved(workdir):
    config = minimal()
    shortcut = config.pop("model")
    config["model"] = {"models": {"a.b": shortcut, "training": dict(shortcut)}}
    config["training"]["predicts"] = "training"
    prepared, kinds = kinds_of(workdir, config)
    assert "model_name_dot" in kinds and "model_name_reserved" in kinds


def test_turn_order(workdir):
    config = minimal()
    config["training"]["turn"] = {"uri": "alternating", "params": {"order": ["model", "ghost"]}}
    prepared, kinds = kinds_of(workdir, config)
    assert "turn_order" in kinds
    config = minimal()
    config["model"] = {"models": {
        "a": {"optimizer": "g", "inputs": ["x"], "outputs": ["y"],
              "nodes": [{"uri": "linear", "params": {"out_features": 1}}]},
        "b": {"optimizer": "d", "inputs": ["x"], "outputs": ["y"],
              "nodes": [{"uri": "linear", "params": {"out_features": 1}}]}}}
    config["optimizers"] = {"g": {"uri": "adam", "loss": "mse"}, "d": {"uri": "adam", "loss": "mse"}}
    config["training"]["predicts"] = "a"
    del config["training"]["loss"]
    prepared, kinds = kinds_of(workdir, config)
    assert "turn_order" in kinds


def test_turn_params_must_fit_the_signature(workdir):
    config = minimal()
    config["training"]["grad_clip"] = 1.0
    config["training"]["momentum"] = 0.9
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == ["signature_mismatch"]


def test_warnings_no_seed_unused_preprocessor_untrained_model(workdir):
    config = minimal()
    del config["seed"]
    config["data"]["preprocessors"]["spare"] = {"uri": "standard_scaler"}
    config["model"] = {"models": {
        "model": {"optimizer": "main", "inputs": ["x"], "outputs": ["y"],
                  "nodes": [{"uri": "linear", "params": {"out_features": 1}}]},
        "idle": {"inputs": ["x"], "outputs": ["y"], "nodes": [{"uri": "linear", "params": {"out_features": 1}}]}}}
    config["optimizers"] = {"main": {"uri": "adam", "loss": "mse"}}
    config["training"]["predicts"] = "model"
    del config["training"]["loss"]
    prepared, kinds = kinds_of(workdir, config)
    assert sorted(kinds) == ["no_seed", "untrained_model", "unused_preprocessor"]
    assert all(problem.severity == "warning" for problem in prepared.problems)


def test_source_missing_and_dtype_unsupported(tmp_path, monkeypatch):
    import pandas

    monkeypatch.chdir(tmp_path)
    prepared, kinds = kinds_of(tmp_path, minimal())
    assert kinds == ["source_missing"]
    assert prepared.dump() is not None
    pandas.DataFrame({"x0": [1.0, 2.0], "price": [1.0, 2.0], "kind": ["a", "b"]}).to_parquet("housing.parquet")
    config = minimal()
    config["data"]["fields"]["kind"] = {}
    prepared, kinds = kinds_of(tmp_path, config)
    assert "dtype_unsupported" in kinds


def test_empty_plots_and_metrics_are_empty_groups(workdir):
    config = minimal()
    config["plots"] = {}
    config["metrics"] = {}
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == [] and prepared.pipeline is not None
    del config["plots"]
    del config["metrics"]
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == []


def test_reserved_field_and_duplicate_name(workdir):
    config = minimal()
    config["data"]["fields"]["input"] = {}
    config["metrics"]["mse"] = {"uri": "mse"}
    prepared, kinds = kinds_of(workdir, config)
    assert "reserved_field" in kinds and "duplicate_name" in kinds


def test_definition_keys_are_accepted_and_plot_inputs_are_typed(workdir):
    import cirak

    config = minimal()
    config["data"]["preprocessors"]["scale"] = {"uri": "standard_scaler", "sets": ["train"]}
    config["losses"]["mse"] = {"uri": "mse", "every": 2, "sets": ["train", "valid"], "output": "y", "target": "price"}
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == []
    document = prepared.document
    assert document["flow"]["training"]["params"]["losses_keys"]["mse"] == {"sets": ["train", "valid"], "every": 2,
                                                                            "output": "y", "target": "price"}
    assert document["flow"]["data"]["params"]["preprocessors_keys"] == {"scale": {"sets": ["train"]}}
    kalfa.lego("/plot/test/curve", lambda predictions, history, models, record, series=None: None,
               partial=True, refs={"series": "history"})
    config["plots"]["curve"] = {"uri": "/plot/test/curve", "inputs": {"series": "val/rmse"}}
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == []
    config["plots"]["curve"] = {"uri": "/plot/test/curve", "inputs": {"series": "val/ghost", "nope": "x"}}
    prepared, kinds = kinds_of(workdir, config)
    assert sorted(kinds) == ["signature_mismatch", "unresolved_ref"]


def test_set_override_rule(workdir):
    config = minimal()
    config["params"] = {"lr": 0.1}
    config["model"]["optimizer"]["params"]["lr"] = "$lr$"
    path = write_config(workdir / "cfg.yaml", config)
    surface = load_surface([path], parse_sets(["training.epochs=7", "device=cpu"], ["lr=0.5"]))
    assert surface.data["model"]["optimizer"]["params"]["lr"] == 0.5
    assert surface.data["training"]["epochs"] == 7 and surface.data["device"] == "/device/kalfa/cpu"
    assert any(path[0] == "training" for path, _, _ in surface.overrides)
    assert "overrides" in surface.layers_text()
    with pytest.raises(ValueError, match="-p lr=0.5"):
        parse_sets(["lr=0.5"])
    with pytest.raises(ValueError):
        parse_sets(["novalue"])
    assert parse_sets(params=["a.b=1"]) == [("params.a.b", 1)]


def test_turn_without_an_extras_declaration_takes_no_extra_keys(workdir):
    import cirak

    def turn(models, optimizers, emas, counters, composites, effects, loader, params, extra, losses, metrics,
             losses_keys, metrics_keys, predicts, steps):
        return {}

    kalfa.lego("/turn/test/plain", turn, returns=["models", "optimizers", "emas", "counters", "metrics"])
    config = minimal()
    config["training"]["turn"] = "/turn/test/plain"
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == []
    config["training"]["grad_clip"] = 1.0
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == ["signature_mismatch"]
    assert "no extras fact" in prepared.problems[0].hint


def two_targets():
    """The minimal config with two target fields (x0 and price) and two output wires (y and y2)."""
    config = minimal()
    config["data"]["fields"] = {"x0": {"target": True, "preprocessors": ["scale"]},
                                "x*": {"preprocessors": ["scale"]},
                                "price": {"target": True}}
    config["model"]["outputs"] = ["y", "y2"]
    config["model"]["nodes"] = {"y": {"uri": "linear", "params": {"out_features": 1}, "inputs": ["x"]},
                                "y2": {"uri": "linear", "params": {"out_features": 1}, "inputs": ["x"]}}
    config["metrics"] = {"rmse": {"uri": "rmse", "output": "y"}}
    config["losses"] = {"mse": {"uri": "mse", "output": "y"}}
    return config


def test_several_targets_and_wires_need_the_targets_table(workdir):
    prepared, kinds = kinds_of(workdir, two_targets())
    assert "targets_missing" in kinds
    assert "write training.targets" in prepared.errors[0].message


def test_the_targets_table_binds_the_wires_and_the_definitions_inherit(workdir):
    config = two_targets()
    config["training"]["targets"] = {"y": "price", "y2": "x0"}
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == []
    keys = prepared.document["flow"]["training"]["params"]
    assert keys["losses_keys"]["mse"] == {"output": "y", "target": "price"}
    assert keys["metrics_keys"]["rmse"] == {"output": "y", "target": "price"}
    assert prepared.document["flow"]["after"]["params"]["targets"] == {"y": "price", "y2": "x0"}


def test_the_targets_table_is_checked_against_the_wires_and_the_fields(workdir):
    config = two_targets()
    config["training"]["targets"] = {"ghost": "price", "y2": "x0"}
    prepared, kinds = kinds_of(workdir, config)
    assert "targets_not_a_wire" in kinds
    config["training"]["targets"] = {"y": "nowhere_*", "y2": "x0"}
    prepared, kinds = kinds_of(workdir, config)
    assert "target_not_a_field" in kinds
    config["training"]["targets"] = {"y": ["price", "x0"], "y2": 3}
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == ["invalid_value"]


def test_a_definition_target_names_a_field(workdir):
    config = two_targets()
    config["training"]["targets"] = {"y": "price", "y2": "x0"}
    config["losses"]["mse"]["target"] = "ghost"
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == ["target_not_a_field"]
    config["losses"]["mse"]["target"] = ["price", "x0"]
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == []


def test_a_definition_without_a_target_is_caught(workdir):
    config = two_targets()
    config["training"]["targets"] = {"y2": "x0"}
    prepared, kinds = kinds_of(workdir, config)
    assert kinds == ["target_missing", "target_missing"]      # the loss and the metric that name the wire y
    assert "training.targets" in prepared.errors[0].message


def test_unresolved_lego_reference_in_params(workdir):
    import myexample  # noqa: F401

    config = minimal()
    config["losses"]["adv"] = {"uri": "/objective/myexample/alad_generator", "params": {"criterion": "ghost",
                                                                                     "latent_dim": 2}}
    prepared, kinds = kinds_of(workdir, config)
    assert "unresolved_ref" in kinds


def test_unknown_variable(workdir):
    config = minimal()
    config["training"]["epochs"] = "$missing$"
    prepared, kinds = kinds_of(workdir, config)
    assert "unknown_variable" in kinds


def test_plot_name_clash_for_nameless_plot_legos(workdir):
    import cirak

    @kalfa.lego("/plot/test/nameless", partial=True)
    def nameless(predictions, history, models, record):
        return None

    @kalfa.lego("/plot/test/named", partial=True)
    def named(predictions, history, models, record, name=None):
        return None

    config = minimal()
    config["plots"] = {"a": {"uri": "/plot/test/nameless"}, "b": {"uri": "/plot/test/nameless"},
                       "c": {"uri": "/plot/test/named"}, "d": {"uri": "/plot/test/named"}}
    problems = check([str(write_config(workdir / "cfg.yaml", config))], parse_sets([])).problems
    assert [problem.kind for problem in problems] == ["plot_name_clash"]
    assert "plots.b and plots.a" in problems[0].message
