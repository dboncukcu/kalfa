import copy
import json
import re

import pytest
from cirak.errors import CirakWarning, ConfigError, error, render_problems, warning

from helpers import ROOT, config_path, load_config, write_config
from kalfa import lego
from kalfa.api import Prepared, check, gate
from kalfa.check import sets_text
from kalfa.config import parse_sets


REFERENCE = config_path("reference")

CHECK_SOURCES = [*sorted((ROOT / "src" / "kalfa" / "check").glob("*.py")), ROOT / "src" / "kalfa" / "config.py",
                 ROOT / "src" / "kalfa" / "api.py"]

FIELD_SOURCE = ROOT / "src" / "kalfa" / "std" / "pre" / "base.py"

KIND_CALL = re.compile(r'(?:error|warning)\("([a-z_]+)"')

KIND_TUPLE = re.compile(r'\(\("([a-z_]+)",\s*f"')

HEADER_SIZES = {"train": 1400, "valid": 300, "test": 300}

MEASURED_SIZES = {"train": 1250, "valid": 273, "test": 278}


def put(config, path, value):
    holder = config
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value


def drop(config, path):
    holder = config
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]


def gp_objective(models, batch):
    return None


def plain_turn(models, optimizers, emas, counters, composites, effects, loader, params, extra, losses, losses_keys,
               metrics_keys, predicts, steps, stream=None):
    return None


def nameless_plot(predictions, history, models, record):
    return None


def needy_plot(predictions, history, models, record, name=None):
    return None


def teacher_run(config, workdir, best=True):
    root = workdir / "runs" / "teacher"
    (root / "checkpoints").mkdir(parents=True)
    write_config(root / "resolved.yaml", {"model": copy.deepcopy(config["model"])})
    if best:
        (root / "checkpoints" / "best.pt").write_bytes(b"")


def teacher_weights(config, run="runs/teacher", model="tail_head", which="best"):
    put(config, ("model", "models", "tail_head", "weights"), {"run": run, "model": model, "which": which})


def lazy(config, workdir):
    config["include"] = ["/alias/kalfa/lazy_tabular"]
    put(config, ("data", "batch", "balanced"), True)
    put(config, ("data", "feed"), {"uri": "window", "params": {"size": 4, "horizon": 1}})
    put(config, ("losses", "cw"), {"uri": "cross_entropy", "params": {"weight": {"uri": "class_weights"}},
                                   "output": "tail_logit"})


def dataset_source(config, workdir):
    (workdir / "lines.txt").write_text("a b c\nd e f\n")
    put(config, ("data", "source"), {"uri": "/source/kalfa/text_lines", "params": {"path": "lines.txt"}})


def amp_without_scaler(config, workdir):
    lego("/objective/check_test/gp", gp_objective, needs_grad=True, partial=True)
    put(config, ("losses", "gp"), {"uri": "/objective/check_test/gp", "sets": ["train"]})
    put(config, ("training", "amp"), True)


def unforeseen_column(config, workdir):
    config["data"]["transform"].append({"uri": "derive", "params": {"column": "z", "expr": "nope * 2"}})


def opposite_grouped_orders(config, workdir):
    put(config, ("data", "fields", "heavy"), {"preprocessors": ["robust", "std"]})
    put(config, ("data", "fields", "count"), {"preprocessors": ["std", "robust"]})


def plot_needing_a_library(config, workdir):
    lego("/plot/check_test/needy", needy_plot, partial=True, requires="zzz_never_installed")
    put(config, ("plots", "needy"), {"uri": "/plot/check_test/needy"})


def dotted_model_name(config, workdir):
    config["model"]["models"]["a.b"] = config["model"]["models"].pop("tail_head")


def template_named_like_a_block(config, workdir):
    config["model"]["templates"]["after"] = config["model"]["templates"]["res_block"]


def nameless_plot_twice(config, workdir):
    lego("/plot/check_test/nameless", nameless_plot, partial=True)
    put(config, ("plots", "a"), {"uri": "/plot/check_test/nameless"})
    put(config, ("plots", "b"), {"uri": "/plot/check_test/nameless"})


def prepared_from_another_data_section(config, workdir):
    (workdir / "prepared").mkdir()
    manifest = {"kind": "data", "hash": "other", "sets": ["train", "valid", "test"],
                "sizes": {"train": 10, "valid": 2, "test": 2},
                "header": {"columns": ["num_0"], "dtypes": {"num_0": "float64"}, "rows": 14}}
    (workdir / "prepared" / "manifest.json").write_text(json.dumps(manifest))


def unreadable_source(config, workdir):
    (workdir / "bad.parquet").write_bytes(b"not a parquet file")
    put(config, ("data", "source", "params", "path"), "bad.parquet")


def turn_taking_no_metrics(config, workdir):
    lego("/turn/check_test/plain", plain_turn, partial=True,
         returns=["models", "optimizers", "emas", "counters", "stream", "metrics"])
    put(config, ("training", "turn"), {"uri": "/turn/check_test/plain", "params": {"order": ["main", "aux"]}})
    drop(config, ("training", "accumulate"))
    drop(config, ("training", "grad_clip"))


def weights_of_another_architecture(config, workdir):
    teacher_run(config, workdir)
    teacher_weights(config)
    put(config, ("model", "models", "tail_head", "nodes"), [{"uri": "linear", "params": {"out_features": 3}}])


def weights_without_the_checkpoint(config, workdir):
    teacher_run(config, workdir, best=False)
    teacher_weights(config)


CASES = {
    "ambiguous_model": (lambda c, d: put(c, ("model", "inputs"), ["x"]), "error",
                        "model mixes the single model shortcut with templates or models"),
    "amp_scaler": (amp_without_scaler, "error", "losses.gp needs gradients under amp, so its signature must take "
                                                "scaler"),
    "column_in_fields": (lambda c, d: put(c, ("data", "fields", "site"), {"preprocessors": ["ordinal"]}), "error",
                         "column 'site' is referenced by a lego; it is not a field and does not enter the model"),
    "column_unused": (lambda c, d: put(c, ("data", "drop"), []), "warning",
                      "columns ['noise_id'] match no field and no spectator; they are read and discarded"),
    "columns_unforeseen": (unforeseen_column, "warning",
                           "the columns after the transforms cannot be foreseen on an empty table: "
                           "UndefinedVariableError"),
    "composite_key": (lambda c, d: put(c, ("model", "models", "full", "optimizer"), "main"), "error",
                      "composite model 'full' cannot write optimizer"),
    "driver_failed": (lambda c, d: put(c, ("data", "spectators"), 5), "error",
                      "the driver cannot shape this config: TypeError"),
    "drop_missing": (lambda c, d: put(c, ("data", "drop"), ["noise_id", "nope"]), "warning",
                     "drop names 'nope', which is no column"),
    "dropped_column_ref": (lambda c, d: put(c, ("data", "drop"), ["noise_id", "site"]), "error",
                           "column 'site' is referenced by a lego and cannot be dropped"),
    "dropped_spectator": (lambda c, d: put(c, ("data", "spectators"), ["sample_id", "noise_id"]), "error",
                          "spectator 'noise_id' matches only dropped columns; drop and spectators cannot name the "
                          "same"),
    "dtype_unsupported": (lambda c, d: put(c, ("data", "fields", "region"), {}), "error",
                          "column 'region' has dtype object; it needs a preprocessor (cast, one_hot, label_encoder)"),
    "duplicate_name": (lambda c, d: put(c, ("metrics", "mse_lin"), {"uri": "mse", "output": "y_hat"}), "error",
                       "'mse_lin' is defined under both losses and metrics"),
    "frame_needs_table": (dataset_source, "error",
                          "a Dataset source has no frame to transform; data.frame needs a table"),
    "generate_missing": (lambda c, d: put(c, ("metrics", "sw"), {"uri": "/metric/kalfa/sample_writer",
                                                                 "params": {"sampler": "generate"}}), "error",
                         "sampler: generate names the generate section, which the config does not write"),
    "grouped_order": (opposite_grouped_orders, "error",
                      "preprocessors 'std' and 'robust' both fit over all their columns and are written in both "
                      "orders"),
    "invalid_call": (lambda c, d: put(c, ("model", "models", "tower", "init", "bias"), "zeros"), "error",
                     "init.bias must be {uri, params}; a short name is not built here"),
    "invalid_params": (lambda c, d: put(c, ("losses", "mse_lin", "params"), 5), "error",
                       "params of losses.mse_lin must be a mapping"),
    "invalid_section": (lambda c, d: put(c, ("metrics",), 5), "error", "metrics must be a mapping"),
    "invalid_value": (lambda c, d: put(c, ("training", "epochs"), -1), "error",
                      "training.epochs must be a non negative integer"),
    "kind_mismatch": (lambda c, d: put(c, ("losses", "mse_lin"), {"uri": "rmse", "output": "y_hat"}), "error",
                      "/metric/kalfa/rmse is a metric lego, losses.mse_lin needs criterion or objective"),
    "lazy_batch": (lazy, "error", "a stream source cannot be counted for the balanced sampler"),
    "lazy_data": (lazy, "error", "losses.cw builds /data/kalfa/class_weights, which counts the train set"),
    "lazy_feed": (lazy, "error", "the feed needs the table in memory"),
    "lazy_fit": (lazy, "warning", "preprocessor 'onehot' (/pre/kalfa/one_hot) has no partial_fit; on a stream its "
                                  "column is collected in memory to fit"),
    "lazy_frame": (lazy, "error", "a stream source cannot fit a frame transform; it needs the table in memory"),
    "lazy_mask": (lazy, "error", "a stream source cannot carry a mask; it needs the table in memory"),
    "lazy_spectators": (lazy, "error", "a stream source carries only the fields it reads; data.spectators needs the "
                                       "table in memory"),
    "lazy_split": (lazy, "error", "a stream source cannot be shuffled or folded"),
    "lazy_transform": (lazy, "error", "transform 0 (/transform/kalfa/rename) needs the table in memory; a stream "
                                      "takes filter only"),
    "library_missing": (plot_needing_a_library, "warning",
                        "plots.needy (/plot/check_test/needy) wants zzz_never_installed, which is not installed"),
    "mask_needs_table": (dataset_source, "error", "a Dataset source has no frame to mask; data.mask needs a table"),
    "missing_key": (lambda c, d: drop(c, ("record",)), "error", "record is required"),
    "model_name_dot": (dotted_model_name, "error", "model name 'a.b' contains a dot"),
    "model_name_reserved": (template_named_like_a_block, "error",
                            "template name 'after' is reserved for a template block"),
    "needs_grad_set": (lambda c, d: put(c, ("losses", "gp"), {"uri": "/objective/kalfa/wgan_gp_d",
                                                              "params": {"generator": "tower", "critic": "head_lin",
                                                                         "latent": 4}}), "error",
                       "losses.gp needs gradients; write sets: [train]"),
    "no_seed": (lambda c, d: drop(c, ("seed",)), "warning",
                "seed is not written; torch runs unseeded and two runs differ"),
    "plot_name_clash": (nameless_plot_twice, "error",
                        "plots.b and plots.a both use /plot/check_test/nameless, which takes no name and writes one "
                        "file"),
    "plugin_import_failed": (lambda c, d: put(c, ("plugins",), ["zzz_not_a_module"]), "error",
                             "cannot import plugin zzz_not_a_module: No module named 'zzz_not_a_module'"),
    "predicts_required": (lambda c, d: drop(c, ("training", "predicts")), "error",
                          "losses or metrics need the predicts model and there is not exactly one trained model; "
                          "write training.predicts"),
    "prepared_mismatch": (prepared_from_another_data_section, "error",
                          "prepared was prepared from another data section; prepare it again from this config"),
    "report_mismatch": (lambda c, d: put(c, ("training", "checkpoint"), "last"), "error",
                        "training.report must be last or a checkpoint the policy writes; 'best' is not among ['last']"),
    "reserved_field": (lambda c, d: put(c, ("data", "fields", "input"), {"preprocessors": ["std"]}), "error",
                       "'input' is a reserved field name"),
    "set_missing": (lambda c, d: put(c, ("data", "split", "ratios"), [0.85, 0.0, 0.15]), "error",
                    "monitor 'val/rmse_lin' watches the valid set, which the split does not produce"),
    "set_target": (lambda c, d: put(c, ("training", "rules", 0, "set"), {"nope.lr": 1.0}), "error",
                   "set target 'nope.lr' names neither an optimizer, a loss nor a trained model"),
    "set_value": (lambda c, d: put(c, ("training", "rules", 1, "set"), {"aux.loss": "nope"}), "error",
                  "set aux.loss names 'nope', which losses does not define"),
    "signature_mismatch": (lambda c, d: put(c, ("training", "bogus"), 1), "error",
                           "training.bogus goes to the turn, but /turn/kalfa/alternating does not accept 'bogus'"),
    "source_missing": (lambda c, d: put(c, ("data", "source", "params", "path"), "nope.parquet"), "error",
                       "data source 'nope.parquet' does not exist"),
    "source_unreadable": (unreadable_source, "error", "cannot read the header of 'bad.parquet'"),
    "spectator_in_fields": (lambda c, d: put(c, ("data", "spectators"), ["sample_id", "heavy"]), "error",
                            "column 'heavy' is a field ('heavy') and a spectator; a column is one or the other"),
    "spectator_missing": (lambda c, d: put(c, ("data", "spectators"), ["sample_id", "nope"]), "warning",
                          "spectator 'nope' matches no column; the columns are ['sample_id', 'num_0'"),
    "spectators_need_table": (dataset_source, "error",
                              "a Dataset source has no frame to carry columns in; data.spectators needs a table"),
    "sweep_space": (lambda c, d: put(c, ("sweep",), {"strategy": "grid", "space": {"lr": {"low": 1.0, "high": 0.5}},
                                                     "objective": {"monitor": "val/rmse_lin"}, "record": "runs/sweep"}),
                    "error", "sweep.space.lr: low must be a number below high"),
    "target_missing": (lambda c, d: put(c, ("losses", "extra"), {"uri": "mse", "output": "nope"}), "error",
                       "losses.extra compares against a target and the data has 5 target fields; write target on the "
                       "definition, or training.targets for the output wire it names"),
    "target_not_a_field": (lambda c, d: put(c, ("training", "targets", "y_hat"), "zz_*"), "error",
                           "training.targets.y_hat names 'zz_*', which matches no target field; the target fields are "
                           "['y_lin', 'y_quad', 'y_heavy', 'y_frac', 'is_hot']"),
    "targets_missing": (lambda c, d: drop(c, ("training", "targets")), "error",
                        "the predicts model writes 3 output wires ['y_hat', 'aux_hat', 'tail_logit'] and the data has "
                        "5 target fields; write training.targets to say which wire predicts which fields"),
    "targets_not_a_wire": (lambda c, d: put(c, ("training", "targets", "nope"), "y_lin"), "error",
                           "training.targets names 'nope', which is no output wire of the predicts model; the wires "
                           "are ['y_hat', 'aux_hat', 'tail_logit']"),
    "test_monitor": (lambda c, d: put(c, ("training", "checkpoint", "params", "monitor"), "test/rmse_lin"), "error",
                     "'test/rmse_lin': stop and checkpoint cannot watch the test set"),
    "turn_order": (lambda c, d: put(c, ("training", "turn"), "supervised"), "error",
                   "supervised is the single optimizer turn; write alternating with order"),
    "turn_without_metrics": (turn_taking_no_metrics, "warning",
                             "turn /turn/check_test/plain takes no metrics; train/ values are not computed"),
    "unknown_alias": (lambda c, d: put(c, ("losses", "mse_lin", "uri"), "nope_alias"), "error",
                      "'nope_alias' is not a known alias and does not start with /"),
    "unknown_key": (lambda c, d: put(c, ("data", "bogus"), 1), "error", "unknown key 'bogus' under data"),
    "unknown_uri": (lambda c, d: put(c, ("losses", "mse_lin", "uri"), "/criterion/kalfa/nope"), "error",
                    "/criterion/kalfa/nope is not registered"),
    "unknown_variable": (lambda c, d: put(c, ("training", "epochs"), "$nope$"), "error", "unknown variable $nope$"),
    "unresolved_ref": (lambda c, d: put(c, ("optimizers", "main", "loss"), "nope"), "error",
                       "optimizers.main.loss names 'nope', which losses does not define"),
    "untrained_model": (lambda c, d: drop(c, ("model", "models", "head_aux", "optimizer")), "warning",
                        "model 'head_aux' writes no optimizer and is never trained"),
    "unused_optimizer": (lambda c, d: put(c, ("optimizers", "spare"), {"uri": "sgd", "params": {"lr": 0.1},
                                                                       "loss": "bce"}), "error",
                         "optimizer 'spare' is defined but no model uses it"),
    "unused_preprocessor": (lambda c, d: put(c, ("data", "preprocessors", "spare"), {"uri": "standard_scaler"}),
                            "warning", "preprocessor 'spare' is defined but no field uses it"),
    "weights_init": (lambda c, d: put(c, ("model", "models", "tower", "weights"), {"run": "runs/teacher",
                                                                                   "model": "tower", "which": "best"}),
                     "error", "model 'tower' writes both weights and init"),
    "weights_mismatch": (weights_of_another_architecture, "error",
                         "weights of model 'tail_head': nodes differ from model 'tail_head' of run 'runs/teacher'"),
    "weights_missing": (weights_without_the_checkpoint, "error",
                        "weights of model 'tail_head': runs/teacher/checkpoints/best.pt does not exist"),
    "weights_run_missing": (lambda c, d: teacher_weights(c, run="runs/nope"), "error",
                            "weights of model 'tail_head': run 'runs/nope' has no resolved.yaml"),
}

UNREACHABLE = {}

CHECK_KWARGS = {"prepared_mismatch": {"prepared": "prepared"}}

FIELD_CASES = {
    "column_missing": (lambda c, d: put(c, ("data", "fields", "zzz"), {"preprocessors": ["std"]}),
                       "field 'zzz' matches no column; the columns are ['sample_id', 'num_0'"),
    "glob_ambiguous": (lambda c, d: (put(c, ("data", "fields", "num_?"), {"preprocessors": ["std"]}),
                                     put(c, ("data", "fields", "?um_0"), {"preprocessors": ["std"]})),
                       "column 'num_0' matches 'num_?' and '?um_0' with equal specificity"),
}


def mutated(kind, workdir, table=CASES):
    config = load_config("reference")
    table[kind][0](config, workdir)
    return check([write_config(workdir / "mutated.yaml", config)], parse_sets([]), **CHECK_KWARGS.get(kind, {}))


def test_cases_cover_every_kind_the_checks_emit():
    kinds = {match for path in CHECK_SOURCES for match in KIND_CALL.findall(path.read_text())}
    assert set(CASES) | set(UNREACHABLE) == kinds
    assert not set(CASES) & set(UNREACHABLE)
    assert set(FIELD_CASES) == set(KIND_TUPLE.findall(FIELD_SOURCE.read_text()))


@pytest.mark.parametrize("kind", sorted(CASES))
def test_every_problem_kind_is_reachable_from_a_config(kind, workdir):
    _, severity, fragment = CASES[kind]
    prepared = mutated(kind, workdir)
    found = [problem for problem in prepared.problems if problem.kind == kind]
    assert found, [(problem.kind, problem.message) for problem in prepared.problems]
    assert any(fragment in problem.message for problem in found), [problem.message for problem in found]
    assert {problem.severity for problem in found} == {severity}
    assert isinstance(prepared, Prepared)


@pytest.mark.parametrize("kind", sorted(FIELD_CASES))
def test_field_assignment_kinds_are_reported_as_errors(kind, workdir):
    prepared = mutated(kind, workdir, FIELD_CASES)
    found = [problem for problem in prepared.problems if problem.kind == kind]
    assert [problem.severity for problem in found] == ["error"]
    assert FIELD_CASES[kind][1] in found[0].message


def test_reference_config_checks_clean_with_the_header_sizes(workdir):
    prepared = check([REFERENCE], parse_sets([]))
    assert prepared.problems == []
    assert prepared.errors == []
    assert prepared.warnings == []
    assert prepared.sizes == HEADER_SIZES
    assert prepared.sets == ["train", "valid", "test"]
    assert prepared.measured is None
    assert prepared.header["rows"] == 2000
    assert prepared.header["columns"] == [
        "sample_id", "num_0", "num_1", "num_2", "num_3", "heavy", "frac", "late", "gap", "count", "region", "tier",
        "site", "noise_id", "y_lin", "y_quad", "y_heavy", "y_frac", "is_hot", "inter", "num_0_median_by_site",
        "site_target"]
    assert prepared.header["dtypes"]["count"] == "float64"
    assert prepared.header["dtypes"]["region"] == "object"
    assert prepared.document is not None
    assert prepared.analysis is not None
    assert prepared.pipeline is not None
    assert prepared.aliasing == []
    assert prepared.surface.paths == [REFERENCE]


def test_measure_counts_the_sets_after_the_filters(workdir):
    prepared = check([REFERENCE], parse_sets([]), measure=True)
    assert prepared.problems == []
    assert prepared.sizes == HEADER_SIZES
    assert prepared.measured == MEASURED_SIZES
    assert all(prepared.measured[name] < prepared.sizes[name] for name in prepared.sets)
    assert sum(prepared.measured.values()) == 1801


def test_measure_is_skipped_on_a_config_with_errors(workdir):
    config = load_config("reference")
    put(config, ("training", "epochs"), -1)
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]), measure=True)
    assert [problem.kind for problem in prepared.errors] == ["invalid_value"]
    assert prepared.measured is None


def test_prepared_splits_errors_from_warnings(workdir):
    config = load_config("reference")
    drop(config, ("seed",))
    put(config, ("training", "epochs"), -1)
    put(config, ("data", "drop"), ["noise_id", "nope"])
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [problem.kind for problem in prepared.errors] == ["invalid_value"]
    assert [problem.kind for problem in prepared.warnings] == ["no_seed", "drop_missing"]
    assert prepared.problems == [*prepared.warnings[:1], *prepared.errors, *prepared.warnings[1:]]
    assert prepared.errors[0].file == str(workdir / "mutated.yaml")
    written = (workdir / "mutated.yaml").read_text().splitlines()
    assert prepared.errors[0].line == written.index("  epochs: -1") + 1
    assert prepared.warnings[0].file is None and prepared.warnings[0].line is None
    assert prepared.warnings[1].file == str(workdir / "mutated.yaml")
    assert prepared.warnings[1].line == next(number for number, line in enumerate(written, 1)
                                             if line.strip().startswith("drop:"))


def test_structural_errors_stop_before_the_driver(workdir):
    config = load_config("reference")
    put(config, ("metrics",), 5)
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [problem.kind for problem in prepared.errors] == ["invalid_section", "unresolved_ref", "unresolved_ref",
                                                             "unresolved_ref"]
    assert prepared.document is None
    assert prepared.analysis is None
    assert prepared.pipeline is None
    assert prepared.dump() is None
    assert prepared.implicit == []
    assert prepared.sizes == HEADER_SIZES


def test_header_phase_errors_still_shape_the_document(workdir):
    config = load_config("reference")
    put(config, ("data", "spectators"), ["sample_id", "heavy"])
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [problem.kind for problem in prepared.errors] == ["spectator_in_fields"]
    assert prepared.document is not None
    assert prepared.pipeline is not None
    assert prepared.dump() is not None


def test_surface_problems_block_the_checker(workdir):
    config = load_config("reference")
    put(config, ("plugins",), ["zzz_not_a_module"])
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [problem.kind for problem in prepared.problems] == ["plugin_import_failed"]
    assert prepared.header is None
    assert prepared.sizes is None
    assert prepared.document is None


def test_an_init_role_written_as_a_short_name_is_invalid_call(workdir):
    config = load_config("reference")
    put(config, ("model", "models", "tower", "init", "bias"), "zeros")
    put(config, ("model", "models", "tower", "init", "patterns"), [{"match": "h_*", "weights": "normal"}])
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [(problem.severity, problem.kind, problem.message, problem.hint) for problem in prepared.problems] == [
        ("error", "invalid_call", "init.bias must be {uri, params}; a short name is not built here", None),
        ("error", "invalid_call", "init pattern weights must be {uri, params}; a short name is not built here", None),
    ]


def test_init_on_a_node_is_invalid_value(workdir):
    config = load_config("reference")
    put(config, ("model", "models", "tower", "nodes", "stem", "init"), {"weights": {"uri": "zeros"}})
    put(config, ("model", "templates", "res_block", "nodes", "f", "init"), {"weights": {"uri": "zeros"}})
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    message = ("init on a node is not applied; write a pattern of the model's init, {match: '<node>.*', "
               "weights | bias | scale}")
    assert [(problem.severity, problem.kind, problem.message) for problem in prepared.problems] == [
        ("error", "invalid_value", message), ("error", "invalid_value", message)]


def test_init_written_as_a_string_or_a_bare_pattern_is_invalid_value(workdir):
    config = load_config("reference")
    put(config, ("model", "models", "tower", "init"), "xavier")
    put(config, ("model", "models", "head_lin", "init"), {"patterns": ["h_*"]})
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [(problem.kind, problem.message) for problem in prepared.problems] == [
        ("invalid_value", "init is a role mapping: {weights, bias, scale, patterns}"),
        ("invalid_value", "an init pattern is {match, weights | bias | scale}"),
    ]


def test_a_positive_absolute_rate_into_every_group_is_a_set_value_warning(workdir):
    config = load_config("reference")
    config["training"]["rules"].append({"name": "climb", "when": {"uri": "after_epoch", "params": {"at": 2}},
                                        "set": {"main.*.lr": 0.01}})
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [(problem.severity, problem.kind, problem.message, problem.hint) for problem in prepared.problems] == [
        ("warning", "set_value", "set main.*.lr writes a positive rate into every group of 'main', a group with a "
                                 "negative rate included, which turns its climb into a descent",
         "write the group's own target, or a relative value ({times: 0.5}) that keeps the sign")]
    assert prepared.document is not None


def test_a_target_name_the_data_does_not_carry_is_a_target_not_a_field_warning(workdir):
    config = load_config("reference")
    put(config, ("training", "targets", "tail_logit"), ["is_hot", "extra"])
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert (
        [(problem.severity, problem.kind) for problem in prepared.problems] == [("warning", "target_not_a_field")] * 6)
    assert prepared.problems[0].message == ("training.targets.tail_logit names ['extra'], which the data does not "
                                            "carry as target fields; only a feed that writes them puts them in the "
                                            "batch")
    assert [problem.message.split(" ")[0] for problem in prepared.problems] == [
        "training.targets.tail_logit", "losses.bce.target", "losses.bce_pos.target", "metrics.auroc.target",
        "metrics.acc.target", "metrics.ap_every.target"]


def test_set_effects_are_checked_against_the_definition(workdir):
    config = load_config("reference")
    config["training"]["rules"][2]["set"] = {"ws.terms.nope": 2.0, "ws.terms.mse_lin": "two", "ws.nope": 1.0,
                                             "main.lr": {"times": "half"}, "ws.terms": {"times": 0.5},
                                             "main.nope.lr": 0.1, "loss": "bce"}
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [(problem.kind, problem.message) for problem in prepared.problems] == [
        ("set_value", "set ws.terms.nope: the definition writes no 'nope' under ws.terms"),
        ("set_value", "set ws.terms.mse_lin: 'two' is a text, the definition holds a number"),
        ("set_value", "set ws.nope: /objective/kalfa/weighted_sum has no parameter 'nope'"),
        ("set_value", "set main.lr: a relative effect is {times: x} or {plus: x} with a number"),
        ("set_value", "set ws.terms: a relative effect ({times}, {plus}) changes an optimizer param only"),
        ("set_target", "set target 'main.nope.lr': optimizer 'main' has no group named 'nope'; the named groups are "
                       "['lambdas', 'stem']"),
        ("set_target", "set: {loss: ...} needs exactly one optimizer; write <opt>.loss"),
    ]


def test_monitors_are_checked_for_their_set_and_their_name(workdir):
    config = load_config("reference")
    config["training"]["stop"] = [{"uri": "plateau", "params": {"monitor": "nope/rmse_lin", "patience": 1}},
                                  {"uri": "plateau", "params": {"monitor": "val/nope", "patience": 1}},
                                  {"uri": "plateau", "params": {"monitor": "rmse_lin", "patience": 1}}]
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [(problem.kind, problem.message) for problem in prepared.problems] == [
        ("invalid_value", "monitor 'nope/rmse_lin' must start with one of train/, val/, test/"),
        ("unresolved_ref", "monitor 'val/nope' names 'nope', which neither losses nor metrics define"),
        ("invalid_value", "monitor must be <set>/<name>, got 'rmse_lin'"),
    ]


def test_unknown_keys_come_with_a_close_match_hint(workdir):
    config = load_config("reference")
    put(config, ("data", "spectator"), ["sample_id"])
    put(config, ("bogus",), 1)
    prepared = check([write_config(workdir / "mutated.yaml", config)], parse_sets([]))
    assert [(problem.kind, problem.message, problem.hint) for problem in prepared.problems] == [
        ("unknown_key", "unknown key 'bogus' under the config", None),
        ("unknown_key", "unknown key 'spectator' under data", "did you mean 'spectators'?"),
    ]


def test_gate_raises_a_config_error_on_errors():
    problems = [error("missing_key", "record is required"), warning("no_seed", "seed is not written")]
    with pytest.raises(ConfigError) as raised, pytest.warns(CirakWarning) as caught:
        gate(problems)
    assert raised.value.problems == [problems[0]]
    assert str(raised.value) == "1 problem found:\n  1. [missing_key] record is required"
    assert str(caught[0].message) == "1 problem found:\n  1. [no_seed] seed is not written"


def test_gate_only_warns_on_warnings():
    problems = [warning("no_seed", "seed is not written", hint="write seed"),
                warning("drop_missing", "drop names 'nope', which is no column")]
    with pytest.warns(CirakWarning) as caught:
        assert gate(problems) is None
    assert len(caught) == 1
    assert str(caught[0].message) == render_problems(problems)
    assert str(caught[0].message) == ("2 problems found:\n  1. [no_seed] seed is not written; write seed\n"
                                      "  2. [drop_missing] drop names 'nope', which is no column")


def test_gate_is_silent_without_problems(recwarn):
    assert gate([]) is None
    assert len(recwarn) == 0


def test_sets_text_writes_the_set_table():
    assert sets_text(HEADER_SIZES) == "sets (before filters, from the file header): train 1400, valid 300, test 300"
    assert sets_text(MEASURED_SIZES, measured=True) == "sets (measured): train 1250, valid 273, test 278"
    assert sets_text(None) == "sets (before filters, from the file header): unknown (the data header could not be read)"
    assert sets_text(None, measured=True) == "sets (measured): unknown (the data header could not be read)"
    assert sets_text({"train": 10, "valid": None, "test": 0}) == (
        "sets (before filters, from the file header): train 10, valid ?, test none")
    assert sets_text({"train": 5, "calib": 2}, sets=("train", "calib")) == (
        "sets (before filters, from the file header): train 5, calib 2")
    assert sets_text({}) == "sets (before filters, from the file header): train ?, valid ?, test ?"
