import pytest
from cirak.api import Analysis
from ruamel.yaml import YAML

from helpers import config_path
from kalfa.api import check
from kalfa.config import parse_sets
from kalfa.contract import Contract
from kalfa.recipe import DRIVER_LABEL, analyze, compile, dump, implicit_bindings, recipe_text


REFERENCE = config_path("reference")

DATA_NODES = ["source", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "split", "set_transform_train",
              "set_transform_valid", "set_transform_test", "frames", "framed_train", "framed_valid", "framed_test",
              "prep", "frame_train", "frame_valid", "frame_test", "feed_train", "feed_valid", "feed_test",
              "loader_train", "loader_valid", "loader_test", "report"]

MODEL_NODES = ["build_tower", "build_head_lin", "build_head_aux", "build_tail_stem", "build_tail_head",
               "build_lambdas", "ema_tower", "models", "emas", "compose_full", "composites"]

TRAINING_BODY = ["effects", "turn", "evaluate_valid", "evaluate_test", "metrics", "rules_0", "rule_1", "rule_2",
                 "rule_3", "rule_4", "rules_ruled", "stop", "checkpoint", "log"]

LOG_BINDINGS = [("training.epochs.body.log", name, name) for name in
                ("monitor", "metrics", "turn_index", "counters_next", "optimizers_next", "rules_next", "effects",
                 "record")]


@pytest.fixture
def prepared(workdir):
    found = check([REFERENCE], parse_sets([]))
    assert found.problems == []
    return found


def parsed(text):
    return YAML(typ="safe").load(text)


def test_driver_label_names_the_document_layer(prepared):
    assert DRIVER_LABEL == "kalfa driver"
    analysis = prepared.analysis
    assert isinstance(analysis, Analysis)
    assert analysis.layer.label == DRIVER_LABEL
    assert [loaded.file for loaded in analysis.layer.walk()] == [str(prepared.contract.path), DRIVER_LABEL]
    assert analysis.problems == []
    assert analysis.overrides == []


def test_analyze_keeps_the_document_and_expands_the_flow(prepared):
    analysis = analyze(prepared.document, prepared.contract)
    assert analysis.problems == []
    assert analysis.data["losses"] == prepared.document["losses"]
    assert analysis.data["blocks"]["res_block"] == prepared.document["blocks"]["res_block"]
    assert list(analysis.flow) == ["outputs", "data", "models", "optimizers", "training", "after"]
    assert analysis.flow["outputs"] == ["history", "predictions"]
    assert list(analysis.flow["data"]) == DATA_NODES
    assert list(analysis.flow["models"]) == MODEL_NODES
    assert list(analysis.flow["optimizers"]) == ["build_opt_main", "build_opt_aux", "optimizers"]
    assert list(analysis.flow["training"]) == ["counters", "rules", "stream", "init", "epochs"]
    assert list(analysis.flow["training"]["epochs"]["loop"]["body"]) == TRAINING_BODY
    assert list(analysis.flow["after"]) == ["final", "report", "calibrate", "predict", "figures", "losses",
                                            "losses_keys", "architecture", "plots", "generate"]


def test_expanded_model_nodes_carry_the_builder_and_the_block(prepared):
    flow = prepared.analysis.flow
    build = flow["models"]["build_tower"]
    assert build["block"] == "tower"
    assert build["builder"] == "/builder/kalfa/module"
    assert build["outputs"] == ["tower"]
    assert build["params"]["name"] == "tower"
    assert build["params"]["index"] == 0
    assert build["params"]["seed"] == 11
    assert build["params"]["trainable"] is True
    assert build["params"]["rng"] == {"uri": "/rng/kalfa/derived", "params": {}}
    assert flow["models"]["build_tail_stem"]["params"]["trainable"] is False
    assert flow["models"]["ema_tower"] == {"uri": "/lego/kalfa/clone", "params": {"decay": 0.9},
                                           "inputs": {"model": "tower"}, "outputs": ["tower_ema"]}
    assert flow["models"]["compose_full"] == {"block": "full", "builder": "/builder/kalfa/module",
                                              "inputs": {"models": "models"}, "outputs": ["full"]}
    assert flow["optimizers"]["build_opt_aux"]["outputs"] == ["opt_aux"]
    assert flow["optimizers"]["build_opt_aux"]["inputs"] == {"models": {"tail_stem": "tail_stem",
                                                                        "tail_head": "tail_head"}}
    assert flow["data"]["loader_valid"]["outputs"] == ["valid_loader"]
    assert flow["data"]["pre_2"] == {"uri": "/transform/kalfa/filter", "params": {"query": "num_1 > -2.5"},
                                     "inputs": {"df": "df_1"}, "outputs": ["df_2"]}


def test_compile_gives_a_pipeline_without_problems(prepared):
    pipeline, problems = compile(prepared.analysis, prepared.contract.run_inputs, dry=True)
    assert problems == []
    assert pipeline is not None
    assert pipeline.resolved is not None
    assert prepared.pipeline is not None


def test_analyze_reports_a_document_that_misses_a_block_variable(prepared):
    document = {**prepared.document, "flow": {"outputs": [], "data": {"block": "data", "params": {}}}}
    analysis = analyze(document, prepared.contract)
    errors = [problem for problem in analysis.problems if problem.severity == "error"]
    assert [problem.kind for problem in errors] == ["missing_variable"]
    assert errors[0].message == "block 'data' requires variable 'source' (used at flow.data)"
    assert {problem.kind for problem in analysis.problems} == {"missing_variable", "unused_component"}


def test_dump_parses_as_the_three_part_document(prepared):
    text = prepared.dump()
    document = parsed(text)
    assert list(document) == ["components", "blocks", "flow"]
    assert list(document["components"]) == ["losses", "metrics", "triggers", "plots", "calibrate", "checkpoint"]
    assert document["components"]["checkpoint"] == prepared.document["checkpoint"]
    assert document["components"]["losses"] == prepared.document["losses"]
    assert list(document["blocks"]) == ["res_block", "tower", "head_lin", "head_aux", "tail_stem", "tail_head",
                                        "lambdas", "full"]
    assert document["blocks"] == prepared.document["blocks"]
    assert list(document["flow"]) == ["outputs", "data", "models", "optimizers", "training", "after"]
    assert list(document["flow"]["models"]) == MODEL_NODES
    assert document["flow"]["models"]["ema_tower"]["outputs"] == ["tower_ema"]
    assert "# implicit: device" in text
    assert parsed(dump(prepared.analysis, None))["flow"]["data"].keys() == document["flow"]["data"].keys()


def test_recipe_text_round_trips_the_driver_document(prepared):
    text = recipe_text(prepared.document)
    assert text.startswith("losses:\n  mse_lin: {uri: /adapter/kalfa/criterion, params: {criterion: "
                           "{uri: /criterion/kalfa/mse}}}\n")
    assert parsed(text) == prepared.document
    assert list(parsed(text)) == ["losses", "metrics", "triggers", "plots", "calibrate", "checkpoint", "blocks",
                                  "flow"]


def test_implicit_bindings_list_the_wiring_of_every_step(prepared):
    bindings = list(implicit_bindings(prepared.pipeline.resolved))
    assert bindings == prepared.implicit
    assert {key for _, _, key in bindings} == {"device", "record", "monitor", "prep", "train_loader", "metrics",
                                               "turn_index", "counters_next", "optimizers_next", "rules_next",
                                               "effects"}
    assert all(param == key for _, param, key in bindings)
    assert sorted(path for path, _, key in bindings if key == "monitor") == ["training.epochs.body.log",
                                                                             "training.epochs.body.turn"]
    assert [binding for binding in bindings if binding[0] == "training.epochs.body.log"] == LOG_BINDINGS
    assert [binding for binding in bindings if binding[0].startswith("models.build_tower")] == [
        ("models.build_tower", "prep", "prep"), ("models.build_tower", "train_loader", "train_loader")]
    assert [path for path, _, key in bindings if key == "device"] == [
        "data.loader_train", "data.loader_valid", "data.loader_test", "training.init",
        "training.epochs.body.turn", "training.epochs.body.evaluate_valid", "training.epochs.body.evaluate_test",
        "after.calibrate", "after.predict", "after.architecture"]
    assert bindings[0] == ("data.frames", "record", "record")
    assert bindings[-1] == ("after.generate", "record", "record")


def test_implicit_bindings_of_nothing_is_empty():
    assert list(implicit_bindings(None)) == []


def test_default_contract_is_used_when_none_is_given(prepared):
    assert analyze(prepared.document).flow == analyze(prepared.document, Contract.load()).flow
