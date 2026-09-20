import json
from pathlib import Path

import pytest

import kalfa
from data import reference_frame
from helpers import reference_sets
from kalfa.api import check
from kalfa.config import parse_sets
from kalfa.contract import Contract
from kalfa.record import read_resolved
from kalfa.std.split.kalfa.splits import cuts_of


pytestmark = pytest.mark.slow

TOP = ["architecture.json", "contract.yaml", "data.json", "device.json", "events.jsonl", "flow.yaml", "git.json",
       "history.jsonl", "host.json", "manifest.json", "predictions.parquet", "resolved.yaml", "run.json",
       "stderr.txt", "stdout.txt", "steps.jsonl"]
PREPROCESSORS = ["abs_train", "fill0", "impute", "onehot", "ordinal", "robust", "squash", "std", "target_std",
                 "to_float", "to_logit"]
MODELS = ["tower", "head_lin", "head_aux", "tail_stem", "tail_head", "lambdas"]
PLOTS = ["loss_curve", "steps", "curves_rates", "pred_vs_true", "pred_histogram", "residuals", "error_map",
         "correlation_heatmap", "feature_distributions", "target_correlation", "target_vs_features", "data_pipeline",
         "class_histogram", "permutation_importance", "binary_roc", "binary_precision_recall_curve",
         *(f"architecture_{name}" for name in [*MODELS, "full"])]
FIELDS = [("num_0", ["std"], False, ["num_0"], []), ("num_1", ["std"], False, ["num_1"], []),
          ("num_2", ["std"], False, ["num_2"], []), ("num_3", ["std"], False, ["num_3"], []),
          ("inter", ["std"], False, ["inter"], []), ("heavy", ["squash", "robust"], False, ["heavy"], []),
          ("frac", ["to_logit", "std"], False, ["frac"], []), ("late", ["impute", "std"], False, ["late"],
                                                               ["late_missing"]),
          ("gap", ["fill0", "abs_train", "std"], False, ["gap"], []), ("count", ["robust"], False, ["count"], []),
          ("region", ["onehot"], False, ["region_east", "region_north", "region_south", "region_west"], []),
          ("tier", ["ordinal"], False, ["tier"], []),
          ("num_0_median_by_site", ["std"], False, ["num_0_median_by_site"], []),
          ("site_target", ["std"], False, ["site_target"], []),
          ("y_lin", ["target_std"], True, ["y_lin"], []), ("y_quad", ["target_std"], True, ["y_quad"], []),
          ("y_heavy", ["squash"], True, ["y_heavy"], []), ("y_frac", ["to_logit"], True, ["y_frac"], []),
          ("is_hot", ["to_float"], True, ["is_hot"], [])]


def tree(record):
    return sorted(str(path.relative_to(record)) for path in Path(record).rglob("*") if path.is_file())


def note(record, name):
    return json.loads((Path(record) / name).read_text())


def test_the_record_holds_exactly_the_expected_files(reference):
    expected = {*TOP, "checkpoints/best.pt", "checkpoints/last.pt", "final/state.pt", "fitted/frames/frames.pkl",
                "fitted/calibrate/calibrate.json", "fitted/calibrate/calibrations.pkl",
                "fitted/preprocessors/plan.json", *(f"fitted/preprocessors/{name}.pkl" for name in PREPROCESSORS),
                *(f"plots/{name}.png" for name in PLOTS), "plots/architecture_text.txt"}
    assert set(tree(reference.record)) == expected
    assert reference.record == "runs/ref_fixed"


def test_the_manifest_names_the_run(reference):
    manifest = note(reference.record, "manifest.json")
    assert list(manifest) == ["kind", "started", "version", "name", "config", "params", "contract", "turn", "prepared"]
    assert manifest["kind"] == "run" and manifest["turn"] == "epoch" and manifest["prepared"] is None
    assert manifest["version"] == kalfa.__version__
    assert manifest["contract"] == Contract.load().digest()
    assert manifest["params"]["constraints"] == {"mae_lin": {"epsilon": 0.5, "lmbda_init": 0.0, "scale": 1.0,
                                                             "damping": 1.0}, "ws.huber_hv": {"epsilon": 0.3}}
    assert manifest["params"]["epochs"] == 3 and manifest["params"]["seed"] == 11


def test_the_notes_describe_the_host_device_and_git_state(reference):
    assert set(note(reference.record, "host.json")) == {"hostname", "pid", "cwd"}
    assert note(reference.record, "device.json") == {"device": "cpu", "uri": "/device/kalfa/cpu", "params": {}}
    assert set(note(reference.record, "git.json")) == {"commit", "dirty"}


def test_the_resolved_config_checks_clean_and_keeps_the_record_token(reference):
    resolved = read_resolved(reference.record)
    assert "include" not in resolved and resolved["record"] == "runs/ref_$datetime$"
    assert resolved["data"]["source"]["uri"] == "/source/kalfa/parquet" and resolved["seed"] == 11
    assert resolved["training"]["turn"]["uri"] == "/turn/kalfa/alternating"
    prepared = check([str(Path(reference.record) / "resolved.yaml")], parse_sets([]))
    assert prepared.problems == []
    assert check([reference.record], parse_sets([])).problems == []


def test_the_contract_copy_and_the_flow_dump_are_written(reference):
    assert (Path(reference.record) / "contract.yaml").read_text() == Contract.load().text()
    flow = (Path(reference.record) / "flow.yaml").read_text()
    assert flow.startswith("components:")
    assert "build_tower:" in flow and "compose_full:" in flow and "ema_tower:" in flow


def test_the_data_report_follows_the_pipeline_stage_by_stage(reference):
    data = note(reference.record, "data.json")
    source = reference_frame()
    kept = int((source["raw_1"] > -2.5).sum())
    assert list(data) == ["stages", "set_transforms", "split", "after_set_transforms", "frames", "fit", "sets",
                          "loaders"]
    stages = [(stage["stage"], stage["rows"], stage["columns"]) for stage in data["stages"]]
    assert stages == [("df_0", 2000, 20), ("df_1", 2000, 20), ("df_2", kept, 20), ("df_3", kept, 21),
                      ("df_4", kept, 21), ("df_5", kept, 20)]
    assert data["stages"][1]["added"] == ["num_0", "num_1", "num_2", "num_3"]
    assert data["stages"][1]["removed"] == ["raw_0", "raw_1", "raw_2", "raw_3"]
    assert data["stages"][3]["added"] == ["inter"] and data["stages"][5]["removed"] == ["junk"]
    assert data["set_transforms"] == {"train": [{"uri": "/transform/kalfa/filter",
                                                 "params": {"query": "y_heavy > -6 and y_heavy < 6"}}]}
    first, second = cuts_of(kept, [0.7, 0.15, 0.15])
    assert data["split"] == {"test": kept - second, "train": first, "valid": second - first}
    assert data["after_set_transforms"]["train"] < data["split"]["train"]
    assert data["after_set_transforms"]["valid"] == data["split"]["valid"]
    assert data["after_set_transforms"]["test"] == data["split"]["test"]
    assert data["frames"] == ["GroupStatistic", "TargetEncoding"]
    assert data["fit"] == {"fields": 19, "features": 18, "targets": ["y_lin", "y_quad", "y_heavy", "y_frac", "is_hot"],
                           "preprocessors": {"squash": 2, "to_logit": 2, "impute": 1, "fill0": 1, "abs_train": 1,
                                             "onehot": 1, "ordinal": 1, "to_float": 1, "std": 10, "robust": 2,
                                             "target_std": 2},
                           "extras": ["late_missing"]}
    for name in ("train", "valid", "test"):
        assert data["sets"][name] == {"rows": data["after_set_transforms"][name], "features": 18}
    train = reference_sets()["train"]
    train = train[(train["y_heavy"] > -6) & (train["y_heavy"] < 6)]
    assert data["loaders"]["train"] == {"batches": int((train["num_0"] <= 1.5).sum()) // 64, "size": 64}
    assert data["loaders"]["valid"] == {"batches": 2, "size": 256} and data["loaders"]["test"]["size"] == 256


def test_the_fitted_plan_lists_every_field_chain_and_the_carried_columns(reference):
    plan = note(reference.record, "fitted/preprocessors/plan.json")
    assert list(plan) == ["fields", "sets", "dtypes", "drop", "spectators"]
    fields = [(field["name"], field["chain"], field["target"], field["columns"], field["extras"])
              for field in plan["fields"]]
    assert fields == FIELDS
    assert plan["sets"] == {"abs_train": ["train"]} and plan["drop"] == ["noise_id"]
    assert plan["spectators"] == ["sample_id", "site"]


def test_the_calibration_note_records_the_fitted_threshold(reference):
    calibrate = note(reference.record, "fitted/calibrate/calibrate.json")
    assert list(calibrate) == ["hot_cut"]
    assert {key: value for key, value in calibrate["hot_cut"].items() if key != "threshold"} == {
        "set": "valid", "quantile": 0.9, "output": "tail_logit"}
    assert isinstance(calibrate["hot_cut"]["threshold"], float)


def test_the_architecture_note_covers_every_model_and_the_composite(reference):
    architecture = note(reference.record, "architecture.json")
    assert list(architecture) == ["models", "features", "targets"]
    assert list(architecture["models"]) == [*MODELS, "full"]
    assert architecture["features"] == [name for _, _, target, columns, extras in FIELDS if not target
                                        for name in [*columns, *extras]]
    assert architecture["targets"] == {name: [name] for name in ("y_lin", "y_quad", "y_heavy", "y_frac", "is_hot")}
    for entry in architecture["models"].values():
        assert list(entry) == ["boxes", "arrows", "widths", "parameters", "trainable"]
    for name in MODELS:
        entry = architecture["models"][name]
        assert entry["trainable"] == entry["parameters"] > 0, name
    assert architecture["models"]["lambdas"]["parameters"] == 2
    parts = [name for name in MODELS if name != "lambdas"]
    for counted in ("parameters", "trainable"):
        assert architecture["models"]["full"][counted] == sum(architecture["models"][name][counted]
                                                              for name in parts)
    assert {box["name"] for box in architecture["models"]["head_lin"]["boxes"]} >= {"base", "delta", "y_hat"}
