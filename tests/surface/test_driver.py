import pytest
from cirak.registry import registry

from helpers import config_path, load_config
from kalfa import lego
from kalfa.config import load_surface, parse_sets
from kalfa.contract import Contract
from kalfa.driver import (
    block_of,
    blocks_of,
    call,
    call_resolved,
    call_with_params,
    chain_graph,
    column_refs,
    component_of,
    components_of,
    data_params,
    ema_items,
    eval_sets,
    is_composite,
    is_shortcut,
    keys_of,
    loaders_of,
    models_of,
    node_of,
    optimizers_of,
    plots_keys_of,
    predicts_of,
    recipe,
    set_values,
    spectators_of,
    split_sets,
    trained_and_composites,
    transforms_of,
    triggers_of,
)


REFERENCE = config_path("reference")

CONSTRAINTS = {"mae_lin": {"epsilon": 0.5, "lmbda_init": 0.0, "scale": 1.0, "damping": 1.0},
               "ws.huber_hv": {"epsilon": 0.3}}

TOWER_INIT = {"weights": {"uri": "/init/torch/xavier"}, "bias": {"uri": "/init/torch/zeros"},
              "scale": {"uri": "/init/torch/normal", "params": {"std": 0.05, "mean": 1.0}},
              "patterns": [{"match": "h_*", "weights": {"uri": "/init/torch/normal", "params": {"std": 0.02}}}]}

OPTIMIZER_ITEMS = [
    {"name": "main", "uri": "/optimizer/torch/adam",
     "params": {"lr": 0.001, "groups": [{"name": "lambdas", "match": "lambdas.*", "lr": -0.01},
                                        {"name": "stem", "match": "tower.nodes.stem.*", "lr": 0.0005}]},
     "loss": "total", "schedule": None,
     "models": {"tower": "tower", "head_lin": "head_lin", "head_aux": "head_aux", "lambdas": "lambdas"}},
    {"name": "aux", "uri": "/optimizer/torch/sgd", "params": {"lr": 0.05, "momentum": 0.9}, "loss": "bce",
     "schedule": {"uri": "/schedule/kalfa/warmup_cosine", "params": {"warmup": 4, "total": 200}},
     "models": {"tail_stem": "tail_stem", "tail_head": "tail_head"}},
]

TRAINED_ITEMS = [
    {"name": "tower", "index": 0, "init": TOWER_INIT, "trainable": True, "weights": None},
    {"name": "head_lin", "index": 1, "init": None, "trainable": True, "weights": None},
    {"name": "head_aux", "index": 2, "init": None, "trainable": True, "weights": None},
    {"name": "tail_stem", "index": 3, "init": None, "trainable": False, "weights": None},
    {"name": "tail_head", "index": 4, "init": None, "trainable": True, "weights": None},
    {"name": "lambdas", "index": 5, "init": None, "trainable": True, "weights": None},
]

RULES = [
    {"name": "unfreeze", "when": "@triggers.unfreeze", "set": {"tail_stem.trainable": True}, "after": None,
     "sticky": True},
    {"name": "swap_aux", "when": "@triggers.swap_aux", "set": {"aux.loss": "bce_pos"}, "after": None,
     "sticky": True},
    {"name": "cool_stem", "when": "@triggers.cool_stem",
     "set": {"main.stem.lr": {"times": 0.5}, "ws.terms.mse_lin": 2.0}, "after": "swap_aux", "sticky": False},
    {"name": "reweight", "when": "@triggers.reweight", "set": {"total.constraints.mae_lin.scale": 2.0},
     "after": "cool_stem", "sticky": True},
]

TRIGGERS = {
    "unfreeze": {"uri": "/trigger/kalfa/after_turn", "params": {"at": 1}},
    "swap_aux": {"uri": "/trigger/kalfa/after_turn", "params": {"at": 1}},
    "cool_stem": {"uri": "/trigger/kalfa/metric_below", "params": {"monitor": "val/rmse_lin", "value": 1.0e9}},
    "reweight": {"uri": "/trigger/kalfa/plateau", "params": {"monitor": "val/mae_orig", "patience": 0}},
    "stop_0": {"uri": "/trigger/kalfa/time_budget", "params": {"minutes": 1000}},
}

LOSSES_KEYS = {
    "mse_lin": {"output": "y_hat", "target": ["y_lin", "y_quad"]},
    "huber_hv": {"output": "aux_hat", "target": ["y_heavy", "y_frac"]},
    "mae_lin": {"output": "y_hat", "target": ["y_lin", "y_quad"]},
    "bce": {"output": "tail_logit", "target": "is_hot"},
    "bce_pos": {"output": "tail_logit", "target": "is_hot"},
    "probe_heavy": {"sets": ["valid", "test"], "every": 2, "output": "aux_hat", "target": ["y_heavy", "y_frac"]},
    "ws": {},
    "total": {},
}

METRICS_KEYS = {
    "rmse_lin": {"output": "y_hat", "target": ["y_lin", "y_quad"]},
    "mae_orig": {"output": "y_hat", "target": ["y_lin", "y_quad"]},
    "recon_lin": {"output": "y_hat", "target": ["y_lin", "y_quad"]},
    "auroc": {"output": "tail_logit", "target": "is_hot"},
    "acc": {"output": "tail_logit", "target": "is_hot"},
    "ap_every": {"sets": ["valid", "test"], "every": 2, "output": "tail_logit", "target": "is_hot"},
}

BLOCKS = {
    "res_block": {
        "variables": {"width": {"required": True}, "drop": {"default": 0.1}},
        "inputs": ["u"], "outputs": ["v"],
        "graph": {
            "f": {"uri": "/layer/kalfa/linear", "params": {"out_features": "$width$"}, "inputs": ["u"]},
            "n": {"uri": "/layer/torch/batch_norm", "inputs": ["f"]},
            "a": {"uri": "/layer/torch/hardtanh", "params": {"min_val": -3.0, "max_val": 3.0}, "inputs": ["n"]},
            "d": {"uri": "/layer/torch/dropout", "params": {"p": "$drop$"}, "inputs": ["a"]},
            "v": {"uri": "/layer/kalfa/add", "inputs": ["u", "d"]}}},
    "tower": {
        "inputs": ["x"], "outputs": ["h"],
        "graph": {
            "norm": {"uri": "/layer/torch/layer_norm",
                     "params": {"normalized_shape": {"uri": "/data/kalfa/feature_width"}}, "inputs": ["x"]},
            "stem": {"uri": "/layer/kalfa/linear", "params": {"out_features": 16}, "inputs": ["norm"]},
            "h": {"block": "res_block", "params": {"width": 16}, "inputs": ["stem"], "repeat": 2}}},
    "head_lin": {
        "inputs": ["h", "x"], "outputs": ["y_hat"],
        "graph": {
            "base": {"uri": "/layer/kalfa/select",
                     "params": {"index": {"uri": "/data/kalfa/feature_index",
                                          "params": {"columns": ["num_0", "num_1"]}}},
                     "inputs": ["x"]},
            "delta": {"uri": "/layer/kalfa/mlp", "params": {"widths": [16], "dropout": 0.1, "out_features": 2},
                      "inputs": ["h"]},
            "y_hat": {"uri": "/layer/kalfa/add", "inputs": ["base", "delta"]}}},
    "head_aux": {"inputs": ["h"], "outputs": ["aux_hat"],
                 "spec": [{"uri": "/layer/kalfa/linear", "params": {"out_features": 2}}]},
    "tail_stem": {"inputs": ["x"], "outputs": ["s"],
                  "spec": [{"uri": "/layer/kalfa/linear_relu", "params": {"out_features": 8}}]},
    "tail_head": {"inputs": ["s"], "outputs": ["tail_logit"],
                  "spec": [{"uri": "/layer/kalfa/linear", "params": {"out_features": 1}}]},
    "lambdas": {"inputs": ["x"], "outputs": ["lmbda"],
                "spec": [{"uri": "/layer/kalfa/multipliers", "params": {"names": CONSTRAINTS}}]},
    "full": {
        "inputs": ["x"], "outputs": ["y_hat", "aux_hat", "tail_logit"],
        "graph": {
            "h": {"model": "tower", "inputs": ["x"]},
            "y_hat": {"model": "head_lin", "inputs": ["h", "x"]},
            "aux_hat": {"model": "head_aux", "inputs": ["h"]},
            "s": {"model": "tail_stem", "inputs": ["x"]},
            "tail_logit": {"model": "tail_head", "inputs": ["s"]}}},
}

TRANSFORM_PRE = [
    {"uri": "/transform/kalfa/rename", "params": {"pattern": "^raw_(\\d)$", "to": "num_\\1"}},
    {"uri": "/transform/kalfa/filter", "params": {"query": "num_1 > -2.5"}},
    {"uri": "/transform/kalfa/derive", "params": {"column": "inter", "expr": "num_0 * num_1"}},
    {"uri": "/transform/kalfa/astype", "params": {"columns": {"count": "float64"}}},
    {"uri": "/transform/kalfa/drop", "params": {"columns": ["junk"]}},
]

TRAIN_FILTER = {"uri": "/transform/kalfa/filter", "params": {"query": "y_heavy > -6 and y_heavy < 6"}}

PREPROCESSORS = {
    "std": {"uri": "/pre/sklearn/standard_scaler"},
    "target_std": {"uri": "/pre/sklearn/standard_scaler"},
    "robust": {"uri": "/pre/sklearn/robust_scaler"},
    "onehot": {"uri": "/pre/kalfa/one_hot"},
    "ordinal": {"uri": "/pre/kalfa/label_encoder"},
    "squash": {"uri": "/pre/kalfa/tanh", "params": {"scale": 3.0}},
    "to_logit": {"uri": "/pre/kalfa/logit"},
    "impute": {"uri": "/pre/kalfa/simple_imputer", "params": {"strategy": "median", "indicator": True}},
    "fill0": {"uri": "/pre/kalfa/fill", "params": {"value": 0.0}},
    "abs_train": {"uri": "/pre/kalfa/abs"},
    "to_float": {"uri": "/pre/kalfa/cast", "params": {"dtype": "float32"}},
}

PREPROCESSORS_KEYS = {"std": {}, "target_std": {}, "robust": {}, "onehot": {}, "ordinal": {}, "squash": {},
                      "to_logit": {}, "impute": {}, "fill0": {}, "abs_train": {"sets": ["train"]}, "to_float": {}}

PLOT_URIS = {
    "loss_curve": "/plot/kalfa/loss_curve", "steps": "/plot/kalfa/loss_curve",
    "curves_rates": "/plot/kalfa/loss_curve", "pred_vs_true": "/plot/kalfa/pred_vs_true",
    "pred_histogram": "/plot/kalfa/pred_histogram", "residuals": "/plot/kalfa/residuals",
    "error_map": "/plot/kalfa/error_map", "correlation_heatmap": "/plot/kalfa/correlation_heatmap",
    "feature_distributions": "/plot/kalfa/feature_distributions",
    "target_correlation": "/plot/kalfa/target_correlation", "target_vs_features": "/plot/kalfa/target_vs_features",
    "data_pipeline": "/plot/kalfa/data_pipeline", "architecture": "/plot/kalfa/architecture",
    "architecture_text": "/plot/kalfa/architecture_text", "class_histogram": "/plot/kalfa/class_histogram",
    "permutation_importance": "/plot/kalfa/permutation_importance", "binary_roc": "/plot/torchmetrics/binary_roc",
    "binary_precision_recall_curve": "/plot/torchmetrics/binary_precision_recall_curve",
}


def calib_split(df):
    return {}


@pytest.fixture
def surface():
    found = load_surface([REFERENCE], parse_sets([]))
    assert found.problems == []
    return found


@pytest.fixture
def document(surface):
    return recipe(surface.data, registry, surface.aliases, Contract.load())


@pytest.fixture
def contract():
    return Contract.load()


def test_recipe_has_the_component_tables_the_blocks_and_the_flow(document):
    assert list(document) == ["losses", "metrics", "triggers", "plots", "calibrate", "checkpoint", "blocks", "flow"]
    assert list(document["flow"]) == ["outputs", "data", "models", "optimizers", "training", "after"]
    assert document["flow"]["outputs"] == ["history", "predictions"]
    assert {name: entry["block"] for name, entry in document["flow"].items() if name != "outputs"} == {
        "data": "data", "models": "models", "optimizers": "optimizers", "training": "training", "after": "after"}


def test_optimizer_items_name_the_models_loss_and_schedule(document):
    params = document["flow"]["optimizers"]["params"]
    assert params["optimizer_items"] == OPTIMIZER_ITEMS
    assert params["optimizer_refs"] == {"main": "opt_main", "aux": "opt_aux"}


def test_trained_items_composites_and_emas(document):
    params = document["flow"]["models"]["params"]
    assert params["trained_items"] == TRAINED_ITEMS
    assert params["composite_items"] == [{"name": "full"}]
    assert params["ema_items"] == [{"name": "tower", "decay": 0.9}]
    assert params["trained_refs"] == {name: name for name in
                                      ("tower", "head_lin", "head_aux", "tail_stem", "tail_head", "lambdas")}
    assert params["composite_refs"] == {"full": "full"}
    assert params["ema_refs"] == {"tower": "tower_ema"}
    assert params["seed"] == 11
    assert params["rng"] == {"uri": "/rng/kalfa/derived", "params": {}}
    assert params["builder"] == "/builder/kalfa/module"


def test_rules_reference_their_triggers_and_wrap_lego_values(document):
    params = document["flow"]["training"]["params"]
    assert params["rules"] == RULES
    assert params["stop"] == ["@triggers.stop_0"]
    assert document["triggers"] == TRIGGERS


def test_training_params_carry_the_turn_the_keys_and_the_sets(document):
    params = document["flow"]["training"]["params"]
    assert params["turn"] == {"uri": "/turn/kalfa/alternating",
                              "params": {"order": ["main", "aux"], "steps": {"main": 1, "aux": 2},
                                         "fresh_batch": False}}
    assert params["turn_params"] == {"accumulate": 2, "grad_clip": 5.0}
    assert params["losses_keys"] == LOSSES_KEYS
    assert params["metrics_keys"] == METRICS_KEYS
    assert params["predicts"] == "full"
    assert params["epochs"] == 3
    assert params["steps"] is None
    assert params["sets"] == ["valid", "test"]
    assert params["history_prefix"] == {"train": "train", "valid": "val", "test": "test"}


def test_components_are_wrapped_by_kind(document):
    assert document["losses"]["mse_lin"] == {"uri": "/adapter/kalfa/criterion",
                                             "params": {"criterion": {"uri": "/criterion/kalfa/mse"}}}
    assert document["losses"]["huber_hv"] == {
        "uri": "/adapter/kalfa/criterion",
        "params": {"criterion": {"uri": "/criterion/kalfa/huber", "params": {"delta": 1.0}}}}
    assert document["losses"]["ws"] == {
        "uri": "/adapter/kalfa/objective",
        "params": {"objective": {"uri": "/objective/kalfa/weighted_sum",
                                 "params": {"terms": {"mse_lin": 1.0, "huber_hv": 0.5}}}}}
    assert document["losses"]["total"] == {
        "uri": "/adapter/kalfa/objective",
        "params": {"objective": {"uri": "/objective/kalfa/mdmm",
                                 "params": {"primary": "ws", "multipliers": "lambdas", "constraints": CONSTRAINTS}}}}
    assert document["metrics"]["rmse_lin"] == {"uri": "/adapter/kalfa/metric",
                                               "params": {"metric": {"uri": "/metric/kalfa/rmse"}}}
    assert document["metrics"]["mae_orig"] == {"uri": "/adapter/kalfa/criterion",
                                               "params": {"criterion": {"uri": "/criterion/kalfa/mae"}}}
    assert document["metrics"]["auroc"] == {"uri": "/adapter/kalfa/metric",
                                            "params": {"metric": {"uri": "/metric/torchmetrics/binary_auroc"}}}
    assert list(document["losses"]) == ["mse_lin", "huber_hv", "mae_lin", "bce", "bce_pos", "probe_heavy", "ws",
                                        "total"]
    assert list(document["metrics"]) == ["rmse_lin", "mae_orig", "recon_lin", "auroc", "acc", "ap_every"]


def test_component_of_resolves_lego_refs_and_the_generate_section():
    assert component_of({"uri": "/lego/kalfa/const", "params": {"value": 1}}, registry) == {
        "uri": "/lego/kalfa/const", "params": {"value": 1}}
    assert component_of({"uri": "/objective/kalfa/vae",
                         "params": {"encoder": "e", "decoder": "d", "recon": "/criterion/kalfa/mse"}}, registry) == {
        "uri": "/adapter/kalfa/objective",
        "params": {"objective": {"uri": "/objective/kalfa/vae",
                                 "params": {"encoder": "e", "decoder": "d", "recon": {"uri": "/criterion/kalfa/mse"}}}}}
    sampler = {"uri": "/generate/kalfa/lm_sampler", "params": {"model": "gpt", "prompt": "x"}}
    assert component_of({"uri": "/metric/kalfa/sample_writer", "params": {"sampler": "generate", "n": 4}}, registry,
                        generate=sampler) == {
        "uri": "/adapter/kalfa/metric",
        "params": {"metric": {"uri": "/metric/kalfa/sample_writer", "params": {"sampler": sampler, "n": 4}}}}
    with pytest.raises(ValueError) as caught:
        component_of({"uri": "/metric/kalfa/sample_writer", "params": {"sampler": "generate"}}, registry)
    assert str(caught.value) == ("/metric/kalfa/sample_writer: sampler names the generate section, which the config "
                                 "does not write")
    assert components_of(None, registry) == {}
    assert components_of({"a": "/criterion/kalfa/mae"}, registry, {"mae": "/criterion/kalfa/mae"}) == {
        "a": {"uri": "/adapter/kalfa/criterion", "params": {"criterion": {"uri": "/criterion/kalfa/mae"}}}}


def test_blocks_keep_template_variables_and_node_shapes(document):
    assert document["blocks"] == BLOCKS
    assert list(document["blocks"]) == list(BLOCKS)


def test_block_of_turns_a_chain_into_a_spec_or_a_graph():
    assert block_of({"inputs": ["x"], "outputs": ["y"], "nodes": [{"uri": "/a/b/c", "repeat": 2, "bogus": 1}]}) == {
        "inputs": ["x"], "outputs": ["y"], "spec": [{"uri": "/a/b/c", "repeat": 2}]}
    assert block_of({"variables": {"w": {"required": True}}, "nodes": [{"template": "t", "params": {"w": 1}}]},
                    template=True) == {"variables": {"w": {"required": True}},
                                       "spec": [{"block": "t", "params": {"w": 1}}]}
    assert block_of({"variables": {}, "nodes": []}, template=True) == {"spec": []}
    assert block_of({"inputs": ["a", "b"], "outputs": ["y"], "nodes": [{"uri": "/a/b/c"}, {"uri": "/a/b/d"}]}) == {
        "inputs": ["a", "b"], "outputs": ["y"],
        "graph": {"s0": {"uri": "/a/b/c", "inputs": ["a", "b"]}, "y": {"uri": "/a/b/d", "inputs": ["s0"]}}}
    assert chain_graph([{"uri": "/a/b/c"}, {"uri": "/a/b/d"}], ["a"], ["y", "z"]) == {
        "s0": {"uri": "/a/b/c", "inputs": ["a"]}, "s1": {"uri": "/a/b/d", "inputs": ["s0"], "outputs": ["y", "z"]}}
    assert node_of({"uri": "/a/b/c", "template": "t", "model": "m", "params": {}, "inputs": [], "outputs": [],
                    "init": {}, "repeat": 1, "bogus": 1}) == {
        "uri": "/a/b/c", "block": "t", "model": "m", "params": {}, "inputs": [], "outputs": [], "init": {},
        "repeat": 1}


def test_single_model_shortcut_is_the_model_named_model():
    section = {"optimizer": {"uri": "/optimizer/torch/adam"}, "inputs": ["x"], "outputs": ["y"],
               "nodes": [{"uri": "/layer/kalfa/linear", "params": {"out_features": 1}}]}
    assert is_shortcut(section) is True
    assert is_shortcut({"models": {}}) is False
    assert models_of(section) == ({}, {"model": section})
    assert models_of({"templates": {"t": {}}, "models": {"m": section}}) == ({"t": {}}, {"m": section})
    assert blocks_of(*models_of(section)) == {"model": {
        "inputs": ["x"], "outputs": ["y"], "spec": [{"uri": "/layer/kalfa/linear", "params": {"out_features": 1}}]}}
    assert trained_and_composites({"model": section}) == (
        [{"name": "model", "index": 0, "init": None, "trainable": True, "weights": None}], [])
    assert optimizers_of({"training": {"loss": "mse"}}, {"model": section}) == [
        {"name": "model", "uri": "/optimizer/torch/adam", "params": {}, "loss": "mse", "schedule": None,
         "models": {"model": "model"}}]
    assert predicts_of({}, [{"name": "model"}], []) == "model"


def test_composites_are_the_models_with_model_nodes():
    assert is_composite({"nodes": {"z": {"model": "encoder", "inputs": ["x"]}}}) is True
    assert is_composite({"nodes": [{"model": "encoder"}]}) is True
    assert is_composite({"nodes": [{"uri": "/layer/kalfa/linear"}]}) is False
    assert is_composite({}) is False
    trained, composites = trained_and_composites({
        "a": {"nodes": [{"uri": "/x/y/z"}], "trainable": False, "init": {"bias": {"uri": "/init/torch/zeros"}}},
        "ab": {"nodes": [{"model": "a"}]},
        "b": {"nodes": [{"uri": "/x/y/z"}], "weights": {"run": "r", "model": "a", "which": "best"}}})
    assert trained == [
        {"name": "a", "index": 0, "init": {"bias": {"uri": "/init/torch/zeros"}}, "trainable": False,
         "weights": None},
        {"name": "b", "index": 1, "init": None, "trainable": True,
         "weights": {"run": "r", "model": "a", "which": "best"}}]
    assert composites == [{"name": "ab"}]
    assert ema_items({"a": {"ema": {"decay": 0.5}}, "b": {"ema": "yes"}, "c": {}}) == [{"name": "a", "decay": 0.5}]


def test_predicts_of_follows_the_training_key_then_the_single_trained_model():
    assert predicts_of({"predicts": "full"}, [{"name": "a"}], []) == "full"
    assert predicts_of({"predicts": "tower.ema"}, [{"name": "tower"}, {"name": "b"}], []) == "tower.ema"
    assert predicts_of({}, [{"name": "only"}], [{"name": "c"}]) == "only"
    assert predicts_of({}, [{"name": "a"}, {"name": "b"}], []) is None
    assert predicts_of({}, [], [{"name": "c"}]) is None


def test_calls_normalize_the_short_and_long_forms():
    assert call("/a/b/c") == {"uri": "/a/b/c"}
    assert call({"uri": "/a/b/c", "params": {}}) == {"uri": "/a/b/c"}
    assert call({"uri": "/a/b/c", "params": {"k": 1}, "sets": ["train"]}) == {"uri": "/a/b/c", "params": {"k": 1}}
    assert call_with_params("/a/b/c") == {"uri": "/a/b/c", "params": {}}
    assert call_with_params({"uri": "/a/b/c"}) == {"uri": "/a/b/c", "params": {}}
    given = {"uri": "/a/b/c", "params": {"k": 1}}
    assert call_with_params(given) == given
    assert call_with_params(given)["params"] is not given["params"]
    assert call_resolved("/criterion/kalfa/mse", {}, registry) == {"uri": "/criterion/kalfa/mse"}
    assert call_resolved({"uri": "/pre/kalfa/two_views", "params": {"transform": "aug"}}, {}, registry,
                         {"aug": {"uri": "/pre/kalfa/simclr_aug", "params": {"size": 96}}}) == {
        "uri": "/pre/kalfa/two_views",
        "params": {"transform": {"uri": "/pre/kalfa/simclr_aug", "params": {"size": 96}}}}


def test_keys_of_inherits_the_target_from_the_targets_table():
    section = {"a": {"uri": "x", "output": "w", "every": 2}, "b": {"uri": "x"}, "c": {"uri": "x", "target": "t"},
               "d": "x", "e": {"uri": "x", "output": "v", "sets": ["valid"], "width": 7}}
    assert keys_of(section, {"w": ["p", "q"]}) == {
        "a": {"every": 2, "output": "w", "target": ["p", "q"]}, "b": {"target": ["p", "q"]}, "c": {"target": "t"},
        "d": {"target": ["p", "q"]}, "e": {"output": "v", "sets": ["valid"], "width": 7}}
    assert keys_of(section, {"w": "p", "v": "q"}) == {
        "a": {"every": 2, "output": "w", "target": "p"}, "b": {}, "c": {"target": "t"}, "d": {},
        "e": {"output": "v", "sets": ["valid"], "width": 7, "target": "q"}}
    assert keys_of(section) == {"a": {"every": 2, "output": "w"}, "b": {}, "c": {"target": "t"}, "d": {},
                                "e": {"output": "v", "sets": ["valid"], "width": 7}}
    assert keys_of(None) == {}


def test_plots_keys_carry_the_lego_beside_the_definition_keys(document):
    assert plots_keys_of({"a": {"uri": "/plot/kalfa/loss_curve", "sets": ["valid"], "width": 7, "params": {}},
                          "b": "/plot/kalfa/pred_vs_true"}) == {
        "a": {"sets": ["valid"], "width": 7, "lego": "/plot/kalfa/loss_curve"},
        "b": {"lego": "/plot/kalfa/pred_vs_true"}}
    assert document["flow"]["after"]["params"]["plots_keys"] == {name: {"lego": uri} for name, uri in PLOT_URIS.items()}
    assert document["plots"] == {name: {"uri": uri, **({"params": load_config("reference")["plots"][name]["params"]}
                                                       if "params" in load_config("reference")["plots"][name] else {})}
                                 for name, uri in PLOT_URIS.items()}


def test_after_params_name_the_report_figures_targets_and_bus(document, contract):
    params = document["flow"]["after"]["params"]
    assert params["report"] == "best"
    assert params["predicts"] == "full"
    assert params["figures"] == {"uri": "/lego/kalfa/figures",
                                 "params": {"format": "png", "width": 5.0, "height": 3.5, "dpi": 72}}
    assert params["targets"] == {"y_hat": ["y_lin", "y_quad"], "aux_hat": ["y_heavy", "y_frac"],
                                 "tail_logit": "is_hot"}
    assert params["generate"] is None
    assert params["plot_bus"] == contract.plot_bus
    assert params["loader_refs"] == {"train": "train_loader", "valid": "valid_loader", "test": "test_loader"}
    assert params["extra_sets"] == []
    assert params["losses_keys"] == LOSSES_KEYS
    assert document["checkpoint"] == {"uri": "/checkpoint/kalfa/best", "params": {"monitor": "val/rmse_lin",
                                                                                  "mode": "min"}}
    assert document["calibrate"] == {"hot_cut": {"uri": "/calibrate/kalfa/threshold",
                                                 "params": {"set": "valid", "quantile": 0.9, "output": "tail_logit"}}}


def test_data_params_place_the_transforms_before_and_after_the_split(document, surface):
    params = document["flow"]["data"]["params"]
    assert params["source"] == {"uri": "/source/kalfa/parquet", "params": {"path": "reference.parquet"}}
    assert params["sets"] == ["train", "valid", "test"]
    assert params["transform_pre"] == TRANSFORM_PRE
    assert params["set_transforms"] == {"train": {"set": "train", "transforms": [TRAIN_FILTER]},
                                        "valid": {"set": "valid", "transforms": []},
                                        "test": {"set": "test", "transforms": []}}
    assert params["transform_notes"] == [{"lego": item["uri"], "with": item["params"]} for item in TRANSFORM_PRE]
    assert params["set_transform_notes"] == {"train": [{"lego": "/transform/kalfa/filter",
                                                        "with": {"query": "y_heavy > -6 and y_heavy < 6"}}]}
    assert params["split"] == {"uri": "/split/kalfa/random", "params": {"ratios": [0.7, 0.15, 0.15], "seed": 11}}
    assert params["split_outputs"] == {"train": "train_df_0", "valid": "valid_df_0", "test": "test_df_0"}
    assert params["frame_refs"] == {"train": "train_frame", "valid": "valid_frame", "test": "test_frame"}
    assert params["mask"] == "num_0 > 1.5"
    assert params["feed"] == {"uri": "/feed/kalfa/table", "params": {}}
    assert params["loaders"] == {name: {"uri": "/loader/kalfa/torch", "set": name,
                                        "params": {"set": name, "size": 64, "eval_size": 256, "drop_last": True}}
                                 for name in ("train", "valid", "test")}
    assert params["frames"] == {
        "uri": "/lego/kalfa/fit_frames",
        "params": {"frames": [
            {"uri": "/frame/kalfa/group_statistic", "params": {"by": "site", "column": "num_0", "statistic": "median"}},
            {"uri": "/frame/kalfa/target_encoding",
             "params": {"column": "site", "target": "y_lin", "smoothing": 2.0}}]},
        "inputs": {"df": "train_df_1"}}
    assert params["prep"] == {
        "uri": "/lego/kalfa/fit",
        "params": {"fields": surface.data["data"]["fields"], "preprocessors": PREPROCESSORS, "drop": ["noise_id"],
                   "keys": PREPROCESSORS_KEYS, "spectators": ["sample_id", "site"]},
        "inputs": {"df": "train_df"}}
    assert params["preprocessors_keys"] == PREPROCESSORS_KEYS


def test_data_params_of_a_record_or_a_prepared_directory_read_instead_of_fitting(surface, contract):
    data = surface.data["data"]
    recorded = data_params(data, surface.aliases, registry, contract, record="runs/x")
    assert recorded["prep"] == {"uri": "/lego/kalfa/read_prep", "params": {"record": "runs/x"}, "inputs": {}}
    assert recorded["frames"] == {"uri": "/lego/kalfa/read_frames", "params": {"record": "runs/x"}, "inputs": {}}
    prepared = data_params(data, surface.aliases, registry, contract, prepared="prep")
    assert prepared["source"] == {"uri": "/source/kalfa/prepared", "params": {"path": "prep"}}
    assert prepared["split"] == {"uri": "/split/kalfa/prepared", "params": {"path": "prep"}}
    assert prepared["transform_pre"] == []
    assert prepared["set_transforms"] == {name: {"set": name, "transforms": []} for name in ("train", "valid", "test")}
    assert prepared["transform_notes"] == []
    assert prepared["set_transform_notes"] == {}
    assert prepared["frames"] == {"uri": "/lego/kalfa/const", "params": {"value": []}, "inputs": {}}
    assert prepared["prep"] == {"uri": "/lego/kalfa/read_prep", "params": {"record": "prep"}, "inputs": {}}
    assert prepared["loaders"] == recorded["loaders"]


def test_preprocessor_refs_are_resolved_to_their_definition(contract):
    data = {"source": {"uri": "/source/kalfa/image_folder", "params": {"path": "d"}},
            "split": {"ratios": [0.8, 0.2, 0.0], "seed": 1}, "batch": 8,
            "preprocessors": {"aug": {"uri": "/pre/kalfa/simclr_aug", "params": {"size": 96}},
                              "two": {"uri": "/pre/kalfa/two_views", "params": {"transform": "aug"},
                                      "sets": ["train"]}},
            "fields": {"image": {"preprocessors": ["two"]}}, "feed": "/feed/kalfa/table"}
    params = data_params(data, {}, registry, contract)
    assert params["prep"]["params"]["preprocessors"] == {
        "aug": {"uri": "/pre/kalfa/simclr_aug", "params": {"size": 96}},
        "two": {"uri": "/pre/kalfa/two_views",
                "params": {"transform": {"uri": "/pre/kalfa/simclr_aug", "params": {"size": 96}}}}}
    assert params["preprocessors_keys"] == {"aug": {}, "two": {"sets": ["train"]}}
    assert params["prep"]["params"]["spectators"] == []
    assert params["prep"]["params"]["drop"] == []
    assert params["mask"] is None
    assert params["frames"] == (
        {"uri": "/lego/kalfa/fit_frames", "params": {"frames": []}, "inputs": {"df": "train_df_1"}})
    assert params["loaders"]["train"] == {"uri": "/loader/kalfa/torch", "set": "train",
                                          "params": {"set": "train", "size": 8}}


def test_spectators_include_the_columns_legos_reference(surface):
    data = surface.data["data"]
    assert column_refs(data, registry) == [("site", "frame.0.params.by"), ("site", "frame.1.params.column")]
    assert spectators_of(data, registry) == ["sample_id", "site"]
    assert spectators_of({"spectators": ["site", "id"], "frame": data["frame"]}, registry) == ["site", "id"]
    assert spectators_of({}, registry) == []


def test_split_sets_follow_the_returns_fact_of_the_split(contract):
    assert split_sets({}, contract, registry) == ["train", "valid", "test"]
    assert split_sets({"split": {"uri": "/split/nope/x"}}, contract, registry) == ["train", "valid", "test"]
    lego("/split/driver_test/calib", calib_split, returns=["train", "calib"])
    assert split_sets({"split": {"uri": "/split/driver_test/calib"}}, contract, registry) == ["train", "calib"]
    assert eval_sets(["train", "calib"], contract) == ["calib"]
    assert eval_sets(["train", "valid", "test", "calib"], contract) == ["valid", "test", "calib"]
    lego("/split/driver_test/notrain", calib_split, returns=["valid"])
    with pytest.raises(ValueError) as caught:
        split_sets({"split": {"uri": "/split/driver_test/notrain"}}, contract, registry)
    assert str(caught.value) == (
        "the split /split/driver_test/notrain returns ['valid']; a split must return a train set")


def test_loaders_take_the_batch_section_as_params(contract):
    assert loaders_of(64, contract, ["train", "calib"]) == {
        "train": {"uri": "/loader/kalfa/torch", "set": "train", "params": {"set": "train", "size": 64}},
        "calib": {"uri": "/loader/kalfa/torch", "set": "calib", "params": {"set": "calib", "size": 64}}}
    assert loaders_of(None, contract, ["train"]) == {
        "train": {"uri": "/loader/kalfa/torch", "set": "train", "params": {"set": "train", "size": None}}}
    assert loaders_of({"size": 8, "workers": 2}, contract, ["train"])["train"]["params"] == {
        "set": "train", "size": 8, "workers": 2}


def test_transforms_split_on_the_presence_of_sets(contract):
    pre, per_set = transforms_of({"transform": ["a > 1", {"uri": "/transform/kalfa/drop", "params": {"columns": ["z"]}},
                                                {"uri": "/transform/kalfa/filter", "params": {"query": "b > 2"},
                                                 "sets": ["valid", "test"]}]}, contract, ["train", "valid", "test"])
    assert pre == [{"uri": "/transform/kalfa/filter", "params": {"query": "a > 1"}},
                   {"uri": "/transform/kalfa/drop", "params": {"columns": ["z"]}}]
    assert per_set == {"train": {"set": "train", "transforms": []},
                       "valid": {"set": "valid", "transforms": [{"uri": "/transform/kalfa/filter",
                                                                 "params": {"query": "b > 2"}}]},
                       "test": {"set": "test", "transforms": [{"uri": "/transform/kalfa/filter",
                                                               "params": {"query": "b > 2"}}]}}
    assert transforms_of({}, contract, ["train"]) == ([], {"train": {"set": "train", "transforms": []}})


def test_set_values_wrap_lego_typed_effects_only():
    losses = {"v": {"uri": "/objective/kalfa/vae"}, "plain": "/criterion/kalfa/mse"}
    aliases = {"huber": "/criterion/kalfa/huber"}
    assert set_values({"v.recon": "huber", "v.recon.deep": "huber", "v.w_rec": "huber", "aux.loss": "bce",
                       "loss": "bce", "plain.x": "huber", "nope.recon": "huber", "main.lr": {"times": 0.5},
                       "v.recon2": "/criterion/kalfa/huber"}, losses, aliases, registry) == {
        "v.recon": {"uri": "/criterion/kalfa/huber"}, "v.recon.deep": {"uri": "/criterion/kalfa/huber"},
        "v.w_rec": "huber", "aux.loss": "bce", "loss": "bce", "plain.x": "huber", "nope.recon": "huber",
        "main.lr": {"times": 0.5}, "v.recon2": "/criterion/kalfa/huber"}
    assert set_values(None, losses, aliases, registry) == {}


def test_triggers_of_names_rules_and_stop_positions():
    training = {"rules": [{"name": "a", "when": "/trigger/kalfa/time_budget"},
                          {"name": "b", "when": {"uri": "/trigger/kalfa/after_turn", "params": {"at": 3}}}],
                "stop": [{"uri": "/trigger/kalfa/plateau", "params": {"monitor": "val/x", "patience": 2}},
                         "/trigger/kalfa/time_budget"]}
    assert triggers_of(training) == {
        "a": {"uri": "/trigger/kalfa/time_budget"},
        "b": {"uri": "/trigger/kalfa/after_turn", "params": {"at": 3}},
        "stop_0": {"uri": "/trigger/kalfa/plateau", "params": {"monitor": "val/x", "patience": 2}},
        "stop_1": {"uri": "/trigger/kalfa/time_budget"}}
    assert triggers_of({}) == {}


def test_recipe_of_a_record_reads_the_fitted_state(surface, contract):
    document = recipe(surface.data, registry, surface.aliases, contract, record="runs/x")
    params = document["flow"]["data"]["params"]
    assert params["prep"] == {"uri": "/lego/kalfa/read_prep", "params": {"record": "runs/x"}, "inputs": {}}
    assert params["frames"] == {"uri": "/lego/kalfa/read_frames", "params": {"record": "runs/x"}, "inputs": {}}
    assert params["transform_pre"] == TRANSFORM_PRE
    assert document["blocks"] == BLOCKS
