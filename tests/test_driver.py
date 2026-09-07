"""The driver: 01 and tidy against the generated recipe files, and one small config per reshaping."""

import json
import sys
import warnings

import pytest
from cirak.registry import registry
from ruamel.yaml import YAML

import kalfa  # noqa: F401
from conftest import ROOT

TIDY = ROOT / "examples" / "alad" / "config.yaml"
from kalfa.config import load_surface
from kalfa.driver import (blocks_of, component_of, data_params, ema_items, is_composite, keys_of, models_of,
                          optimizers_of, predicts_of, recipe, resolve_refs, set_values, trained_and_composites,
                          triggers_of)

sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from regenerate_recipes import TARGETS, render  # noqa: E402

DUMPS = ROOT / "configs" / "dumps"


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, default=repr))


@pytest.mark.parametrize("config, name", TARGETS)
def test_recipe_matches_the_generated_file(config, name):
    surface = load_surface([str(ROOT / config)])
    assert surface.errors == []
    document = recipe(surface.data, registry, surface.aliases)
    expected = YAML(typ="safe").load((DUMPS / name).read_text())
    assert expected.pop("include") == ["../../src/kalfa/templates/kalfa.yaml"]
    assert plain(document) == plain(expected)
    assert list(document) == ["losses", "metrics", "triggers", "plots", "progress", "blocks", "flow"]
    assert list(document["flow"]) == ["outputs", "data", "models", "optimizers", "training", "after"]


@pytest.mark.parametrize("config, name", TARGETS)
def test_recipe_text_is_current(config, name):
    assert (DUMPS / name).read_text() == render(config)


def test_tidy_recipe_details():
    surface = load_surface([str(TIDY)])
    document = recipe(surface.data, registry, surface.aliases)
    assert document["losses"]["adv_d"]["params"]["criterion"] == {"uri": "/criterion/kalfa/bce_logits"}
    assert document["metrics"]["auroc"]["uri"] == "/adapter/kalfa/metric"
    assert document["triggers"] == {}
    params = document["flow"]["models"]["params"]
    assert [item["name"] for item in params["trained_items"]] == ["encoder", "generator", "dxz", "dxx", "dzz"]
    assert params["composite_items"] == [{"name": "anomaly_score"}]
    assert document["blocks"]["mlp"]["graph"]["h"] == {"block": "mlp_hidden", "params": {"width": "$width$"},
                                                        "inputs": ["a0"], "repeat": "$depth$"}
    assert document["blocks"]["encoder"]["spec"][0] == {"block": "mlp", "params": {"in_features": 6, "out_features": 64,
                                                                                   "width": 64, "depth": 2}}
    optimizers = document["flow"]["optimizers"]["params"]["optimizer_items"]
    assert [item["name"] for item in optimizers] == ["g", "d"]
    assert optimizers[0]["models"] == {"encoder": "encoder", "generator": "generator"}
    data = document["flow"]["data"]["params"]
    assert data["filter_pre"] == ["(is_anomaly > -4) & (is_anomaly < 4)"]
    assert data["filter_set"] == [{"query": "is_anomaly == 0", "sets": ["train"]}]
    assert data["drop"] == ["xx6", "xx7"]
    assert document["flow"]["training"]["params"]["checkpoint"] is None


def test_single_model_shortcut_becomes_models_model():
    templates, models = models_of({"inputs": ["x"], "outputs": ["y"], "nodes": [], "optimizer": "main"})
    assert templates == {} and list(models) == ["model"]
    templates, models = models_of({"templates": {"t": {"nodes": []}}, "models": {"net": {"nodes": []}}})
    assert list(templates) == ["t"] and list(models) == ["net"]


def test_inline_optimizer_goes_to_the_table_under_the_model_name_with_training_loss():
    config = {"training": {"loss": "mse"}, "optimizers": {}}
    models = {"net": {"optimizer": {"uri": "/optimizer/torch/adam", "params": {"lr": 0.1}}, "nodes": []}}
    items = optimizers_of(config, models)
    assert items == [{"name": "net", "uri": "/optimizer/torch/adam", "params": {"lr": 0.1}, "loss": "mse",
                      "schedule": None, "models": {"net": "net"}}]


def test_named_optimizers_gather_their_models_in_writing_order():
    config = {"training": {}, "optimizers": {"g": {"uri": "/optimizer/torch/adam", "loss": "adv_g"},
                                             "d": {"uri": "/optimizer/torch/adam", "loss": "adv_d",
                                                   "schedule": {"uri": "/schedule/kalfa/step_decay"}}}}
    models = {"enc": {"optimizer": "g", "nodes": []}, "dis": {"optimizer": "d", "nodes": []},
              "gen": {"optimizer": "g", "nodes": []}, "score": {"nodes": {"z": {"model": "enc"}}}}
    items = optimizers_of(config, models)
    assert [item["name"] for item in items] == ["g", "d"]
    assert items[0]["models"] == {"enc": "enc", "gen": "gen"}
    assert items[1]["models"] == {"dis": "dis"} and items[1]["schedule"] == {"uri": "/schedule/kalfa/step_decay"}


def test_trained_and_composite_split_and_index_only_trained_models():
    models = {"enc": {"nodes": [], "init": {"weights": {"uri": "/init/torch/normal"}}, "trainable": False},
              "score": {"nodes": {"z": {"model": "enc", "inputs": ["x"]}}},
              "gen": {"nodes": [{"uri": "/layer/kalfa/linear", "params": {"out_features": 2}}]}}
    trained, composites = trained_and_composites(models)
    assert [item["name"] for item in trained] == ["enc", "gen"]
    assert [item["index"] for item in trained] == [0, 1]
    assert trained[0]["trainable"] is False and trained[0]["init"] == {"weights": {"uri": "/init/torch/normal"}}
    assert trained[1]["trainable"] is True and trained[1]["init"] is None and trained[1]["weights"] is None
    assert composites == [{"name": "score"}]
    assert is_composite(models["score"]) and not is_composite(models["gen"])


def test_ema_items_and_refs():
    models = {"unet": {"nodes": [], "ema": {"decay": 0.999}}, "other": {"nodes": []}}
    assert ema_items(models) == [{"name": "unet", "decay": 0.999}]
    surface_like = {"model": {"models": models}, "losses": {}, "training": {"turn": "/turn/kalfa/alternating",
                                                                             "report": "last", "epochs": 1},
                    "data": {"source": {"uri": "/source/kalfa/parquet", "params": {}},
                             "split": {"ratios": [1, 0, 0]}, "batch": 1, "fields": {}, "feed": "/feed/kalfa/table"}}
    document = recipe(surface_like, registry)
    params = document["flow"]["models"]["params"]
    assert params["ema_items"] == [{"name": "unet", "decay": 0.999}]
    assert params["ema_refs"] == {"unet": "unet_ema"}
    assert params["trained_refs"] == {"unet": "unet", "other": "other"}


def test_split_and_batch_short_forms_and_filter_kinds():
    data = {"source": "/source/kalfa/parquet", "split": {"ratios": [0.8, 0.1, 0.1], "seed": 3}, "batch": 32,
            "filter": ["a > 0", {"query": "b == 1", "sets": ["train"]}], "fields": {}, "feed": "/feed/kalfa/table",
            "preprocessors": {"s": {"uri": "/pre/sklearn/standard_scaler", "sets": ["train"]}}}
    params = data_params(data)
    assert params["split"] == {"uri": "/split/kalfa/random", "params": {"ratios": [0.8, 0.1, 0.1], "seed": 3}}
    assert params["batch"] == {"size": 32}
    assert params["filter_pre"] == ["a > 0"] and params["filter_set"] == [{"query": "b == 1", "sets": ["train"]}]
    assert params["source"] == {"uri": "/source/kalfa/parquet", "params": {}}
    assert params["feed"] == {"uri": "/feed/kalfa/table", "params": {}}
    assert params["preprocessors"] == {"s": {"uri": "/pre/sklearn/standard_scaler"}}
    long = data_params({**data, "split": {"uri": "/split/kalfa/kfold", "params": {"k": 5}},
                        "batch": {"size": 8, "eval_size": 16}})
    assert long["split"] == {"uri": "/split/kalfa/kfold", "params": {"k": 5}}
    assert long["batch"] == {"size": 8, "eval_size": 16}


def test_templates_and_models_become_blocks_with_spec_or_graph():
    templates = {"mlp": {"variables": {"width": {"required": True}}, "inputs": ["x"], "outputs": ["y"],
                         "nodes": [{"uri": "/layer/torch/linear", "params": {"in_features": 2, "out_features": 2}}]}}
    models = {"net": {"inputs": ["x"], "outputs": ["y"],
                      "nodes": {"h": {"template": "mlp", "params": {"width": 4}, "inputs": ["x"], "repeat": 2},
                                "y": {"uri": "/layer/kalfa/linear", "params": {"out_features": 1}, "inputs": ["h"],
                                      "init": {"weights": {"uri": "/init/torch/zeros"}}}}},
              "score": {"inputs": ["x"], "outputs": ["s"], "nodes": {"s": {"model": "net", "inputs": ["x"]}}}}
    blocks = blocks_of(templates, models)
    assert list(blocks) == ["mlp", "net", "score"]
    assert blocks["mlp"]["variables"] == {"width": {"required": True}} and "spec" in blocks["mlp"]
    assert blocks["net"]["graph"]["h"] == {"block": "mlp", "params": {"width": 4}, "inputs": ["x"], "repeat": 2}
    assert blocks["net"]["graph"]["y"]["init"] == {"weights": {"uri": "/init/torch/zeros"}}
    assert blocks["score"]["graph"]["s"] == {"model": "net", "inputs": ["x"]}


def test_criteria_and_metrics_are_wrapped_by_kind_and_objectives_stay_direct():
    criterion = component_of({"uri": "/criterion/kalfa/huber", "params": {"delta": 2.0}, "target": "input"},
                             registry)
    assert criterion == {"uri": "/adapter/kalfa/criterion",
                         "params": {"criterion": {"uri": "/criterion/kalfa/huber", "params": {"delta": 2.0}}}}
    metric = component_of({"uri": "/metric/kalfa/rmse", "every": 2, "sets": ["valid"]}, registry)
    assert metric == {"uri": "/adapter/kalfa/metric", "params": {"metric": {"uri": "/metric/kalfa/rmse"}}}
    unknown = component_of({"uri": "/objective/proj/custom", "params": {"w": 1}, "sets": ["train"]}, registry)
    assert unknown == {"uri": "/objective/proj/custom", "params": {"w": 1}}


def test_definition_keys_go_to_the_parallel_table():
    section = {"a": {"uri": "/criterion/kalfa/mse", "every": 2, "sets": ["valid"], "output": "y", "target": "price"},
               "b": {"uri": "/criterion/kalfa/mae"}, "c": "/criterion/kalfa/mae"}
    assert keys_of(section) == {"a": {"sets": ["valid"], "every": 2, "output": "y", "target": "price"}, "b": {},
                                "c": {}}
    data = {"source": "/source/kalfa/parquet", "split": {"ratios": [1, 0, 0]}, "batch": 1, "fields": {},
            "feed": "/feed/kalfa/table",
            "preprocessors": {"s": {"uri": "/pre/sklearn/standard_scaler", "sets": ["train"]}, "t": {"uri": "/pre/kalfa/abs"}}}
    params = data_params(data)
    assert params["preprocessors"] == {"s": {"uri": "/pre/sklearn/standard_scaler"}, "t": {"uri": "/pre/kalfa/abs"}}
    assert params["preprocessors_keys"] == {"s": {"sets": ["train"]}, "t": {}}
    surface = load_surface([str(ROOT / "configs" / "01_mlp_regression.yaml")])
    document = recipe(surface.data, registry, surface.aliases)
    training = document["flow"]["training"]["params"]
    assert training["losses_keys"] == {"loss_mse": {}, "loss_huber": {}, "loss_mae": {}, "loss_logcosh": {}}
    assert training["metrics_keys"] == {"rmse": {}, "mae": {}}
    assert document["flow"]["after"]["params"]["plots_keys"] == {"loss_curve": {}, "pred_vs_true": {}}
    assert document["flow"]["data"]["params"]["preprocessors_keys"] == {"std_scaler": {}, "target_std": {}}


def test_refs_turn_lego_names_into_inline_components_and_keep_model_names():
    aliases = {"bce_logits": "/criterion/kalfa/bce_logits"}
    params = resolve_refs("/objective/myexample/alad_generator", {"criterion": "bce_logits", "latent_dim": 4}, aliases,
                          registry)
    assert params == {"criterion": {"uri": "/criterion/kalfa/bce_logits"}, "latent_dim": 4}
    untouched = resolve_refs("/optimizer/torch/adam", {"loss": "mse", "schedule": None}, aliases, registry)
    assert untouched == {"loss": "mse", "schedule": None}
    assert resolve_refs("/objective/myexample/alad_generator", {"criterion": "ghost"}, aliases, registry) == {
        "criterion": {"uri": "ghost"}}


def test_predicts_is_derived_from_a_single_trained_model_only():
    assert predicts_of({}, [{"name": "net"}], []) == "net"
    assert predicts_of({}, [{"name": "a"}, {"name": "b"}], []) is None
    assert predicts_of({"predicts": "score"}, [{"name": "a"}, {"name": "b"}], [{"name": "score"}]) == "score"


def test_triggers_are_named_after_rules_and_stop_positions():
    training = {"rules": [{"name": "r", "when": "/trigger/kalfa/after_turn"}],
                "stop": [{"uri": "/trigger/kalfa/plateau", "params": {"monitor": "val/x", "patience": 2}}]}
    assert triggers_of(training) == {"r": {"uri": "/trigger/kalfa/after_turn"},
                                     "stop_0": {"uri": "/trigger/kalfa/plateau",
                                                "params": {"monitor": "val/x", "patience": 2}}}


def test_set_values_keep_loss_names_and_build_lego_references():
    losses = {"vae": {"uri": "/objective/proj/vae"}, "adv": {"uri": "/objective/myexample/alad_generator"}}
    out = set_values({"loss": "mse", "vae.w": 0.5, "adv.criterion": "bce_logits"}, losses,
                     {"bce_logits": "/criterion/kalfa/bce_logits"}, registry)
    assert out == {"loss": "mse", "vae.w": 0.5, "adv.criterion": {"uri": "/criterion/kalfa/bce_logits"}}


def test_template_variables_stay_literal_and_win_inside_the_template(tmp_path):
    surface = load_surface([str(TIDY)])
    assert surface.problems == []
    node = surface.data["model"]["templates"]["mlp"]["nodes"]["h0"]
    assert node["params"] == {"in_features": "$in_features$", "out_features": "$width$"}
    text = TIDY.read_text().replace("  latent_dim: 4\n", "  latent_dim: 4\n  width: 3\n")
    path = tmp_path / "shadow.yaml"
    path.write_text(text)
    (tmp_path / "myexample.py").write_text((TIDY.parent / "myexample.py").read_text())
    surface = load_surface([str(path)])
    assert surface.problems == []
    assert surface.data["model"]["templates"]["mlp"]["nodes"]["h0"]["params"]["out_features"] == "$width$"
    surface = load_surface([str(ROOT / "configs" / "07_wgan_gp.yaml")])
    assert surface.problems == []
    generator = surface.data["model"]["models"]["generator"]["nodes"]["y"]
    assert generator["params"] == {"n_classes": 10}
