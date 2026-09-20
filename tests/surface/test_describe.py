import pytest

from helpers import config_path, load_config, write_config
from kalfa import __version__
from kalfa.api import check, probe
from kalfa.config import parse_sets
from kalfa.describe import ALL_SECTIONS, DEFAULT_SECTIONS, Plain, visible
from kalfa.describe.render import measure_text, render, report
from kalfa.describe.text import ARROW, DOT, head, wide
from kalfa.style import Style


REFERENCE = config_path("reference")

FIELD_ROWS = [
    "  field        columns                     preprocessors                           role",
    "  ──────────────────────────────────────────────────────────────────────────────────────────",
    "  num_*        num_0, num_1, num_2, num_3  std                                     feature",
    "  inter        inter                       std                                     feature",
    "  heavy        heavy                       squash ─→ robust                        feature",
    "  frac         frac                        to_logit ─→ std                         feature",
    "  late         late                        impute ─→ std                           feature",
    "  gap          gap                         fill0 ─→ abs_train (train only) ─→ std  feature",
    "  count        count                       robust                                  feature",
    "  region       region                      onehot                                  feature",
    "  tier         tier                        ordinal                                 feature",
    "  *_by_site    num_0_median_by_site        std                                     feature",
    "  site_target  site_target                 std                                     feature",
    "  y_*          y_lin, y_quad               target_std                              target",
    "  y_heavy      y_heavy                     squash                                  target",
    "  y_frac       y_frac                      to_logit                                target",
    "  is_hot       is_hot                      to_float                                target",
    "  sample_id    sample_id                   —                                       spectator",
    "  site         site                        —                                       spectator",
]

MAIN_LEGO = ("adam lr=0.001 groups=[{'name': 'lambdas', 'match': 'lambdas.*', 'lr': -0.01}, "
             "{'name': 'stem', 'match': 'tower.nodes.stem.*', 'lr': 0.0005}]")
MAIN_TRAINS = "tower, head_lin, head_aux, lambdas"
OPTIMIZER_ROWS = [
    f"  main       {MAIN_LEGO}  ─→ {MAIN_TRAINS}  total  no schedule",
    f"  aux        {'sgd lr=0.05 momentum=0.9'.ljust(len(MAIN_LEGO))}  "
    f"─→ {'tail_stem, tail_head'.ljust(len(MAIN_TRAINS))}  {'bce'.ljust(5)}  warmup_cosine warmup=4 total=200",
]

RULE_ROWS = [
    "  rules (evaluated at the end of a turn, effective in the next)",
    "    turn ≥ 1                ─→  unfreeze   tail_stem.trainable := True",
    "    turn ≥ 1                ─→  swap_aux   aux.loss := bce_pos",
    "    val/rmse_lin < 1e+09    ─→  cool_stem  main.stem.lr := ×0.5, ws.terms.mse_lin := 2.0  after swap_aux, every "
    "turn",
    "    val/mae_orig plateau 0  ─→  reweight   total.constraints.mae_lin.scale := 2.0         after cool_stem",
]

MEASURE_LINES = [
    "measured the data block:  reference.parquet 2 000 rows ─→ 6 transforms",
    "  ─→ split random  0.7 / 0.15 / 0.15  seed=11",
    "  ─→ fitted std, target_std, robust, onehot, ordinal, squash, to_logit, impute, fill0, abs_train, to_float on "
    "train",
    "  ─→ table feed ─→ 3 loaders, batch 64",
    "sets after filters: train 1 250  ·  valid 273  ·  test 278",
]


@pytest.fixture
def prepared(workdir):
    found = check([REFERENCE], parse_sets([]), measure=True)
    assert found.problems == []
    return found


@pytest.fixture
def text(prepared):
    return report(prepared, Plain())


def headers(text):
    return [line[3:].split(" ")[0] for line in text.splitlines() if line.startswith("── ")]


def test_sections_are_the_documented_ones():
    assert DEFAULT_SECTIONS == ("summary", "data", "model", "training", "after", "columns")
    assert ALL_SECTIONS == (*DEFAULT_SECTIONS, "wiring")


def test_report_prints_the_default_sections_in_order(text):
    assert headers(text) == ["DATA", "MODEL", "TRAINING", "AFTER", "COLUMNS"]
    assert text.endswith("\n")
    assert "── WIRING" not in text


def test_wiring_section_lists_the_implicit_bindings(prepared):
    text = report(prepared, Plain(), sections=list(ALL_SECTIONS))
    assert headers(text) == ["DATA", "MODEL", "TRAINING", "AFTER", "COLUMNS", "WIRING"]
    wiring = text[text.index("── WIRING"):].splitlines()[1:]
    assert wiring == [f"  {path}: {param} ← {key}" for path, param, key in prepared.implicit]
    assert wiring[0] == "  data.frames: record ← record"
    only = report(prepared, Plain(), sections=["wiring"])
    assert headers(only) == ["WIRING"]
    assert only.splitlines()[2:] == wiring


def test_summary_names_the_layers_params_seed_and_device(text):
    lines = text.splitlines()
    assert lines[0].startswith("reference.yaml")
    assert lines[0].endswith(f"kalfa {__version__}")
    assert lines[1:12] == [
        "  layers      reference.yaml, /alias/kalfa/base, /alias/kalfa/tabular",
        "  params",
        "    epochs       3",
        "    seed         11",
        "    lr           0.001",
        "    constraints  {'mae_lin': {'epsilon': 0.5, 'lmbda_init': 0.0, 'scale': 1.0, 'damping': 1.0}, "
        "'ws.huber_hv': {'epsilon': 0.3}}",
        "  seed        11",
        "  device      cpu",
        "  rng         derived (no rng key)",
        "  record      runs/ref_$datetime$",
        "",
    ]


def test_data_section_shows_the_source_split_sizes_batch_and_transforms(text):
    lines = text.splitlines()
    assert "  source      parquet  reference.parquet                  2 000 rows · 22 columns" in lines
    assert "  split       random  0.7 / 0.15 / 0.15  seed=11          train 1 400  ·  valid 300  ·  test 300" in lines
    assert "  batch       64                                          feed  table" in lines
    assert ("  transforms  rename pattern=^raw_(\\d)$ to=num_\\1, filter query=num_1 > -2.5, derive column=inter "
            "expr=num_0 * num_1, astype columns={count=float64}, drop columns=[junk]  ·  after the split: filter "
            "query=y_heavy > -6 and y_heavy < 6 (train)  the sizes above are from the header, before these; "
            "--measure counts them") in lines
    assert ("  frames      group_statistic by=site column=num_0 statistic=median, target_encoding column=site "
            "target=y_lin smoothing=2  fitted on train") in lines
    assert "  drop        noise_id" in lines
    assert "  mask        num_0 > 1.5  kept in the frame, not scored" in lines


def test_fields_table_gives_every_field_its_role(text):
    lines = text.splitlines()
    start = lines.index(FIELD_ROWS[0])
    assert lines[start:start + len(FIELD_ROWS)] == FIELD_ROWS


def test_data_tree_walks_every_set_through_the_preprocessors(text):
    lines = text.splitlines()
    start = lines.index("  reference.parquet ─→ split random")
    fitted = "fit std, target_std, robust, onehot, ordinal, squash, to_logit, impute, fill0, abs_train, to_float"
    assert lines[start:start + 4] == [
        "  reference.parquet ─→ split random",
        f"    ├─ train    1 400  ─→ {fitted} ─→ table",
        f"    ├─ valid      300  ─→ {'apply'.ljust(len(fitted))} ─→ table",
        f"    └─ test       300  ─→ {'apply'.ljust(len(fitted))} ─→ table",
    ]


def test_model_section_lists_trained_models_then_composites(text):
    lines = text.splitlines()
    assert ("  tower   trained  ·  optimizer main  ·  init weights=xavier bias=zeros scale=normal std=0.05 mean=1 "
            "patterns=[{'match': 'h_*', 'weights': {'uri': '/init/torch/normal', 'params': {'std': 0.02}}}]  ·  "
            "ema decay 0.9") in lines
    start = lines.index("    inputs x   outputs h")
    assert lines[start:start + 4] == [
        "    inputs x   outputs h",
        "    x     ─→  layer_norm normalized_shape=feature_width  ─→  norm",
        "    norm  ─→  linear 16                                  ─→  stem",
        "    stem  ─→  res_block(width=16)  ×2                    ─→  h",
    ]
    assert "  head_aux   trained  ·  optimizer main" in lines
    assert "    h ─→ linear 2 ─→ aux_hat" in lines
    assert "  tail_stem   frozen (eval mode)  ·  optimizer aux" in lines
    assert "    x ─→ linear_relu 8 ─→ s" in lines
    assert ("    x ─→ multipliers names={mae_lin={epsilon=0.5 lmbda_init=0 scale=1 damping=1} "
            "ws.huber_hv={epsilon=0.3}} ─→ lmbda") in lines
    assert "  the parameter counts need --measure" in lines
    start = lines.index("  full   composite")
    assert lines[start:start + 7] == [
        "  full   composite",
        "    inputs x   outputs y_hat, aux_hat, tail_logit",
        "    x     ─→  tower      ─→  h",
        "    h, x  ─→  head_lin   ─→  y_hat",
        "    h     ─→  head_aux   ─→  aux_hat",
        "    x     ─→  tail_stem  ─→  s",
        "    s     ─→  tail_head  ─→  tail_logit",
    ]
    assert lines.index("  full   composite") > lines.index("  lambdas   trained  ·  optimizer main")


def test_training_section_shows_turn_targets_and_the_optimizer_table(text):
    lines = text.splitlines()
    assert "  turn        alternating         3 epochs        predicts full" in lines
    assert "              order=[main, aux] steps={main=1 aux=2} fresh_batch=false accumulate=2 grad_clip=5" in lines
    start = lines.index("  output wire      predicts the target fields")
    assert lines[start + 2:start + 5] == [
        "  y_hat        ─→  y_lin, y_quad",
        "  aux_hat      ─→  y_heavy, y_frac",
        "  tail_logit   ─→  is_hot",
    ]
    start = lines.index(OPTIMIZER_ROWS[0])
    assert lines[start - 2].split() == ["optimizer", "lego", "trains", "loss", "schedule"]
    assert lines[start:start + 2] == OPTIMIZER_ROWS


def test_training_section_lists_losses_and_metrics_with_their_sets(text):
    lines = text.splitlines()
    losses = lines[lines.index(next(line for line in lines if line.startswith("  loss         lego"))) + 2:]
    assert [line.split() for line in losses[:8]] == [
        ["mse_lin", "mse", "y_hat", "─→", "y_lin,", "y_quad", "train,", "valid,", "test", "held", "for", "the",
         "rules"],
        ["huber_hv", "huber", "delta=1", "aux_hat", "─→", "y_heavy,", "y_frac", "train,", "valid,", "test", "held",
         "for", "the", "rules"],
        ["mae_lin", "mae", "y_hat", "─→", "y_lin,", "y_quad", "train,", "valid,", "test", "held", "for", "the",
         "rules"],
        ["bce", "bce_logits", "tail_logit", "─→", "is_hot", "train,", "valid,", "test", "active", "at", "turn",
         "1"],
        ["bce_pos", "bce_logits", "pos_weight=3", "tail_logit", "─→", "is_hot", "train,", "valid,", "test",
         "held", "for", "the", "rules"],
        ["probe_heavy", "mae", "aux_hat", "─→", "y_heavy,", "y_frac", "valid,", "test", "every", "2", "held",
         "for", "the", "rules"],
        ["ws", "weighted_sum", "terms={mse_lin=1", "huber_hv=0.5}", "train,", "valid,", "test", "held", "for",
         "the", "rules"],
        ["total", "mdmm", "primary=ws", "multipliers=lambdas", "constraints={mae_lin={epsilon=0.5", "lmbda_init=0",
         "scale=1", "damping=1}", "ws.huber_hv={epsilon=0.3}}", "train,", "valid,", "test", "active", "at",
         "turn", "1"],
    ]
    start = lines.index("  metric     lego                      compares                reported on")
    assert lines[start + 2:start + 8] == [
        "  rmse_lin   rmse                      y_hat ─→ y_lin, y_quad  train, valid, test",
        "  mae_orig   mae                       y_hat ─→ y_lin, y_quad  train, valid, test",
        "  recon_lin  recon_error               y_hat ─→ y_lin, y_quad  train, valid, test",
        "  auroc      binary_auroc              tail_logit ─→ is_hot    train, valid, test",
        "  acc        accuracy                  tail_logit ─→ is_hot    train, valid, test",
        "  ap_every   binary_average_precision  tail_logit ─→ is_hot    valid, test  every 2",
    ]


def test_training_section_shows_checkpoint_stop_and_the_rules_block(text):
    lines = text.splitlines()
    assert "  checkpoint  best monitor=val/rmse_lin mode=min          report best" in lines
    assert "  stop        after 1000 minutes" in lines
    start = lines.index(RULE_ROWS[0])
    assert lines[start:start + len(RULE_ROWS)] == RULE_ROWS


def test_after_section_names_report_plots_figures_and_calibrations(text):
    lines = text.splitlines()
    start = lines.index("  report      best        predict full ─→ predictions.parquet")
    assert lines[start:start + 5] == [
        "  report      best        predict full ─→ predictions.parquet",
        "  plots       loss_curve, steps (loss_curve x=step), curves_rates (loss_curve series=[train/total, "
        "val/rmse_lin, lr/aux] rates=true), pred_vs_true, pred_histogram, residuals, error_map, correlation_heatmap, "
        "feature_distributions, target_correlation, target_vs_features, data_pipeline, architecture, "
        "architecture_text, class_histogram, permutation_importance, binary_roc, binary_precision_recall_curve  "
        "─→ plots/",
        "  figures     format png, width 5.0, height 3.5, dpi 72",
        "  calibrate   hot_cut (threshold set=valid quantile=0.9 output=tail_logit)  ─→ fitted/calibrate/",
        "  record      runs/ref_$datetime$",
    ]


def test_columns_section_gives_every_source_column_a_role(text):
    lines = text.splitlines()
    start = lines.index("  column                dtype    field        preprocessors              role                 "
                        "       tensor")
    rows = lines[start + 2:start + 24]
    assert rows[0] == (
        "  sample_id             int64    —            —                          spectator                   —")
    assert rows[1] == (
        "  num_0                 float64  num_*        std                        feature                     x")
    assert rows[5] == (
        "  heavy                 float64  heavy        squash ─→ robust           feature                     x")
    assert rows[8] == (
        "  gap                   float64  gap          fill0 ─→ abs_train ─→ std  feature                     x")
    assert rows[12] == (
        "  site                  object   —            —                          spectator, read by frame.1  —")
    assert rows[13] == (
        "  noise_id              int64    —            —                          dropped                     —")
    assert rows[14] == (
        "  y_lin                 float64  y_*          target_std                 target                      y_hat[0]")
    assert rows[15] == (
        "  y_quad                float64  y_*          target_std                 target                      y_hat[1]")
    assert rows[16] == ("  y_heavy               float64  y_heavy      squash                     target"
                        "                      aux_hat[0]")
    assert rows[18] == ("  is_hot                int64    is_hot       to_float                   target"
                        "                      tail_logit[0]")
    assert rows[20] == (
        "  num_0_median_by_site  float64  *_by_site    std                        feature                     x")
    assert [row.split()[0] for row in rows] == prepared_columns()
    assert lines[start + 24] == "  the produced widths and the tensor slots need --measure"


def prepared_columns():
    return ["sample_id", "num_0", "num_1", "num_2", "num_3", "heavy", "frac", "late", "gap", "count", "region",
            "tier", "site", "noise_id", "y_lin", "y_quad", "y_heavy", "y_frac", "is_hot", "inter",
            "num_0_median_by_site", "site_target"]


def test_report_is_independent_of_the_span(prepared):
    style = Plain()
    assert report(prepared, style, span=1000) == report(prepared, style)
    assert report(prepared, style, sections=["model"], span=500) == report(prepared, style, sections=["model"])
    assert max(wide(line) for line in report(prepared, style).splitlines()) == wide(
        report(prepared, style).splitlines()[0])


def test_render_fits_the_headers_to_the_width(prepared):
    text = render(prepared, Plain(), sections=["data", "model"], width=150)
    assert headers(text) == ["DATA", "MODEL"]
    assert all(wide(line) == 150 for line in text.splitlines() if line.startswith("── "))
    assert head("DATA", 150, Plain()) == "── DATA " + "─" * 142
    assert text == render(prepared, Plain(), sections=["model", "data"], width=150)


def test_styled_text_reads_the_same_when_the_colors_are_stripped(prepared):
    plain = report(prepared, Plain())
    styled = report(prepared, Style(True))
    assert styled != plain
    assert visible(styled) == plain
    assert report(prepared, Style(False)) == plain


def test_measure_text_describes_the_data_block_it_ran(prepared, monkeypatch):
    monkeypatch.setenv("COLUMNS", "96")
    assert prepared.measured == {"train": 1250, "valid": 273, "test": 278}
    assert measure_text(prepared, Plain()) == "\n".join(MEASURE_LINES)
    assert measure_text(prepared, Plain(), sizes={"train": 3, "valid": None, "test": 0}).splitlines()[-1] == (
        f"sets after filters: train 3  {DOT}  valid ?  {DOT}  test 0")


def test_probe_measures_sizes_features_parameters_and_shapes(prepared):
    found = probe(prepared.document, prepared.contract)
    assert found.sizes == {"train": 1250, "valid": 273, "test": 278}
    assert found.features == 18
    assert found.prep.features == ["num_0", "num_1", "num_2", "num_3", "inter", "heavy", "frac", "late",
                                   "late_missing", "gap", "count", "region_east", "region_north", "region_south",
                                   "region_west", "tier", "num_0_median_by_site", "site_target"]
    assert [item.name for item in found.prep.fields if item.target] == ["y_lin", "y_quad", "y_heavy", "y_frac",
                                                                        "is_hot"]
    assert found.parameters == {"tower": (948, 948), "tail_stem": (152, 0), "lambdas": (2, 2), "full": (1449, 1297)}
    assert found.shapes == {"tower": {"h": (64, 16)}, "tail_stem": {"s": (64, 8)}, "lambdas": {"lmbda": (2,)},
                            "full": {"y_hat": (64, 2), "aux_hat": (64, 2), "tail_logit": (64, 1)}}
    assert sorted(found.notes) == ["head_aux", "head_lin", "tail_head"]
    assert found.notes["head_lin"].startswith("not built on the batch: KeyError \"model input 'h' is not a batch field")


def test_report_with_the_probe_fills_in_the_measured_values(prepared):
    found = probe(prepared.document, prepared.contract)
    lines = report(prepared, Plain(), probe=found).splitlines()
    assert "  split       random  0.7 / 0.15 / 0.15  seed=11          train 1 250  ·  valid 273  ·  test 278" in lines
    assert ("    ├─ train    1 250  ─→ fit std, target_std, robust, onehot, ordinal, squash, to_logit, impute, fill0, "
            "abs_train, to_float ─→ table ─→ x [64, 18]") in lines
    assert any(line.endswith("ema decay 0.9  ·  948 parameters") for line in lines)
    assert "  tail_stem   frozen (eval mode)  ·  optimizer aux  ·  152 parameters (0 trainable)" in lines
    assert "  head_lin   trained  ·  optimizer main  ·  parameters after the first batch" in lines
    assert "  lambdas   trained  ·  optimizer main  ·  2 parameters" in lines
    assert "  the parameter counts need --measure" not in lines
    assert "  the produced widths and the tensor slots need --measure" not in lines
    assert ("  num_0                 float64  num_*        std                         feature"
            "                     x[0]") in lines
    assert ("  region                object   region       onehot  (4 columns)         feature"
            "                     x[11 … 14]") in lines
    assert ("  late_missing          bool     late         a side output of the chain  feature"
            "                     x[8]") in lines
    assert not any("the sizes above are from the header" in line for line in lines)


def test_sections_that_need_a_shaped_config_say_so(workdir):
    config = load_config("reference")
    config["metrics"] = 5
    prepared = check([write_config(workdir / "broken.yaml", config)], parse_sets([]))
    assert [problem.kind for problem in prepared.errors][0] == "invalid_section"
    assert prepared.document is None
    lines = report(prepared, Plain()).splitlines()
    assert lines[0].startswith("broken.yaml")
    assert "  the config could not be shaped, no data analysis" in lines
    assert "  the config could not be shaped, no model analysis" in lines
    assert "  the config could not be shaped, no training analysis" in lines
    assert "  the config could not be shaped, no after analysis" in lines
    assert lines.count("  the config could not be shaped, no data analysis") == 1
    assert "  no implicit bindings" in report(prepared, Plain(), sections=["wiring"]).splitlines()
    assert ARROW == "─→"
