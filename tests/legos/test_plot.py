import json
import logging
import shutil

import numpy
import pandas
import pytest
import torch
from cirak.build import Graph, GraphNode
from cirak.registry import registry
from torch import nn

from helpers import build, frame, needs, tiny_model
from kalfa.registration import lego
from kalfa.std import STD_URIS
from kalfa.std.builder.kalfa.module import Module
from kalfa.std.common.device import Device
from kalfa.std.common.figure import Figure
from kalfa.std.feed.base import Dataset
from kalfa.std.lego.kalfa.figures import Figures
from kalfa.std.pre.base import Field, Prep, TableFrame


PLOTS = sorted(uri for uri in STD_URIS if uri.startswith("/plot/"))

KALFA_PLOTS = ["/plot/kalfa/architecture", "/plot/kalfa/architecture_text", "/plot/kalfa/class_histogram",
               "/plot/kalfa/confusion_matrix", "/plot/kalfa/correlation_heatmap", "/plot/kalfa/data_pipeline",
               "/plot/kalfa/error_map", "/plot/kalfa/feature_distributions", "/plot/kalfa/forecast_samples",
               "/plot/kalfa/image_grid", "/plot/kalfa/image_pairs", "/plot/kalfa/loss_curve",
               "/plot/kalfa/permutation_importance", "/plot/kalfa/pred_histogram", "/plot/kalfa/pred_vs_true",
               "/plot/kalfa/residuals", "/plot/kalfa/samples_gif", "/plot/kalfa/samples_matrix",
               "/plot/kalfa/target_correlation", "/plot/kalfa/target_vs_features"]

OTHER_PLOTS = ["/plot/seaborn/kde", "/plot/seaborn/pairplot", "/plot/seaborn/violin",
               "/plot/torchmetrics/binary_precision_recall_curve", "/plot/torchmetrics/binary_roc",
               "/plot/torchview/architecture"]


class Images(Dataset):
    def __init__(self, count=6):
        self.images = torch.rand(count, 1, 4, 4, generator=torch.Generator().manual_seed(0))
        self.inputs = ["image"]
        self.targets = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"image": self.images[index]}


class ImageMaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = ["z"]
        self.outputs = ["image"]

    def forward(self, noise):
        return torch.rand(len(noise), 1, 2, 2, generator=torch.Generator().manual_seed(len(noise)))


class Monitor:
    def __init__(self):
        self.lines = []

    def elapsed(self):
        return 1.5

    def turn(self, line):
        self.lines.append(line)


def figures():
    return build("/lego/kalfa/figures", width=3.0, height=2.0, dpi=40)


def history_lines(turns=4):
    return [{"turn": turn, "global_step": 4 * turn, "train/mse": 1.0 / turn, "val/rmse": 2.0 / turn,
             "lr/main": 0.1 * 0.5 ** (turn - 1), "lr/main/bias": -0.05, "minimizes/main": "mse", "seconds": 0.5,
             "rules": []} for turn in range(1, turns + 1)]


def regression_table(rows=40, seed=0):
    generator = numpy.random.default_rng(seed)
    truth = generator.normal(size=rows)
    guess = truth + generator.normal(scale=0.2, size=rows)
    return pandas.DataFrame({"row": numpy.arange(rows), "price": truth, "raw_y": guess, "pred_y": guess})


def classification_table(rows=40, seed=0):
    generator = numpy.random.default_rng(seed)
    labels = generator.integers(0, 2, size=rows)
    scores = labels * 2.0 - 1.0 + generator.normal(scale=0.7, size=rows)
    return pandas.DataFrame({"row": numpy.arange(rows), "is_hot": labels, "raw_y": scores,
                             "pred_y": (scores > 0).astype("int64")})


def prep_for(data, targets=("price",)):
    fields = [Field(name, [], False, [name]) for name in data.features]
    fields += [Field(name, [], True, [name]) for name in targets]
    return Prep(fields, {}, {}, {}, [])


def table_set(set_name="train", rows=40, seed=0):
    data = frame(rows=rows, seed=seed, set_name=set_name)
    dataset = build("/feed/kalfa/table", frame=data)
    loader = build("/loader/kalfa/torch", data=dataset, set=set_name, size=8, eval_size=8, shuffle=False)
    return data, prep_for(data), loader


def summed_model(out_features=1):
    model = tiny_model(3, out_features)
    with torch.no_grad():
        model.nodes["layer"].weight.fill_(1.0)
        model.nodes["layer"].bias.fill_(0.0)
    return model


def image_model():
    graph = Graph(("image",), ("out",), (GraphNode("same", nn.Identity(), ("image",), ("out",)),))
    return Module(graph, seed=1, name="auto")


def plots_of(record):
    directory = record / "plots"
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def draw(uri, record, params=None, predictions=None, history=None, models=None, **extra):
    plot = build(uri, **(params or {}))
    plot(predictions=predictions, history=history, models=models or {}, record=str(record), figures=figures(),
         **extra)
    return plots_of(record)


def non_empty(record, name):
    return (record / "plots" / name).stat().st_size > 0


def write_turn_samples(record):
    writer = build("/metric/kalfa/sample_writer", n=4)
    for turn in (1, 2):
        writer.reset()
        writer.update({"g": ImageMaker()}, "g", None, str(record), turn, batch={"z": torch.zeros(4, 2)})


def run_all(plots, keys=None, predictions=None, history=None, models=None, predicts=None, bus=None, record=None,
            suffix=""):
    return build("/lego/kalfa/run_all", predictions=predictions, history=history, models=models or {}, plots=plots,
                 keys=keys, predicts=predicts, bus=bus, record=record, figures=figures(), suffix=suffix)


def test_plot_scope_is_every_kalfa_seaborn_torchmetrics_and_torchview_plot():
    assert PLOTS == sorted(KALFA_PLOTS + OTHER_PLOTS)


def test_every_plot_is_partial_and_aliased():
    aliases = registry.aliases()
    for uri in PLOTS:
        assert registry.facts(uri).partial is True
    for uri in KALFA_PLOTS + OTHER_PLOTS[:5]:
        assert aliases[uri.rsplit("/", 1)[1]] == uri
    assert aliases["torchview"] == "/plot/torchview/architecture"
    assert registry.facts("/plot/seaborn/kde").get("requires") == "seaborn"
    assert registry.facts("/plot/torchview/architecture").get("requires") == "torchview"


def test_data_plots_need_the_train_loader_and_the_pipeline_needs_the_report():
    for uri in ("/plot/kalfa/target_vs_features", "/plot/kalfa/target_correlation",
                "/plot/kalfa/correlation_heatmap", "/plot/kalfa/feature_distributions", "/plot/seaborn/kde",
                "/plot/seaborn/pairplot", "/plot/seaborn/violin"):
        assert registry.facts(uri).get("needs") == ["train_loader"]
    assert registry.facts("/plot/kalfa/data_pipeline").get("needs") == ["data_report"]
    assert registry.facts("/plot/kalfa/error_map").refs == {"x": "column", "y": "column", "target": "field"}
    assert registry.facts("/plot/kalfa/residuals").refs == {"target": "field"}


def test_loss_curve_draws_every_history_series(tmp_path):
    assert draw("/plot/kalfa/loss_curve", tmp_path, history=history_lines()) == ["loss_curve.png"]
    assert non_empty(tmp_path, "loss_curve.png")


def test_loss_curve_draws_named_series_and_rates_with_a_learning_rate_panel(tmp_path):
    params = {"series": ["train/mse", "lr/main"], "rates": True, "log": True}
    assert draw("/plot/kalfa/loss_curve", tmp_path, params, history=history_lines(), name="curves") == ["curves.png"]


def test_loss_curve_over_steps_reads_the_steps_file_of_the_record(tmp_path):
    lines = [{"step": step, "turn": 1, "loss/main": 1.0 / step, "lr/main": 0.1, "grad_norm/main": 0.5}
             for step in range(1, 9)]
    (tmp_path / "steps.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert draw("/plot/kalfa/loss_curve", tmp_path, {"x": "step"}, history=history_lines()) == ["loss_curve.png"]


def test_loss_curve_refuses_an_unknown_axis_or_series(tmp_path):
    with pytest.raises(ValueError, match=r"loss_curve.x must be turn or step, got 'epoch'"):
        draw("/plot/kalfa/loss_curve", tmp_path, {"x": "epoch"}, history=history_lines())
    with pytest.raises(ValueError, match=r"\['zz'\] are not in the history; the series are \['train/mse', "
                                          r"'val/rmse'\] and the rates \['lr/main', 'lr/main/bias'\]"):
        draw("/plot/kalfa/loss_curve", tmp_path, {"series": ["train/mse", "zz"]}, history=history_lines())
    assert draw("/plot/kalfa/loss_curve", tmp_path, history=[]) == []


def test_prediction_plots_write_their_files(tmp_path):
    table = regression_table()
    assert draw("/plot/kalfa/pred_vs_true", tmp_path, predictions=table) == ["pred_vs_true.png"]
    assert draw("/plot/kalfa/pred_histogram", tmp_path, {"log": True, "bins": 10}, predictions=table) == [
        "pred_histogram.png", "pred_vs_true.png"]
    assert "residuals.png" in draw("/plot/kalfa/residuals", tmp_path, {"output": "y", "target": "price", "bins": 5},
                                   predictions=table)
    for name in ("pred_vs_true.png", "pred_histogram.png", "residuals.png"):
        assert non_empty(tmp_path, name)


def test_prediction_plots_skip_a_table_without_pairs(tmp_path):
    empty = pandas.DataFrame({"row": [0, 1], "raw_y": [0.1, 0.2]})
    assert draw("/plot/kalfa/pred_vs_true", tmp_path, predictions=empty) == []
    assert draw("/plot/kalfa/pred_histogram", tmp_path, predictions=None) == []
    assert draw("/plot/kalfa/residuals", tmp_path, {"output": "zz"}, predictions=regression_table()) == []


def test_error_map_bins_the_residual_over_two_columns(tmp_path):
    data, prep, loader = table_set("test")
    params = {"x": "x0", "y": "x1", "output": "y", "target": "price", "bins": 4, "min_count": 2}
    files = draw("/plot/kalfa/error_map", tmp_path, params, predictions=regression_table(), loaders={"test": loader},
                 prep=prep, sets=["test"])
    assert files == ["error_map.png"]
    absolute = {**params, "statistic": "abs"}
    assert "abs.png" in draw("/plot/kalfa/error_map", tmp_path, absolute, predictions=regression_table(),
                             loaders={"test": loader}, prep=prep, name="abs")
    assert draw("/plot/kalfa/error_map", tmp_path, {"x": "zz", "y": "x1"}, predictions=regression_table(),
                loaders={"test": loader}, prep=prep, name="none") == ["abs.png", "error_map.png"]


def test_forecast_samples_draws_the_true_and_predicted_horizons(tmp_path):
    table = pandas.DataFrame({"row": [0, 1, 2], "load_0": [1.0, 2.0, 3.0], "load_1": [1.5, 2.5, 3.5],
                              "pred_0": [1.1, 2.1, 3.1], "pred_1": [1.4, 2.4, 3.4]})
    assert draw("/plot/kalfa/forecast_samples", tmp_path, {"n": 2}, predictions=table) == ["forecast_samples.png"]
    assert draw("/plot/kalfa/forecast_samples", tmp_path, predictions=pandas.DataFrame()) == ["forecast_samples.png"]


def test_classification_plots_write_their_files(tmp_path):
    table = classification_table()
    assert draw("/plot/kalfa/class_histogram", tmp_path, {"output": "y", "target": "is_hot", "bins": 10},
                predictions=table) == ["class_histogram.png"]
    assert draw("/plot/kalfa/confusion_matrix", tmp_path, predictions=table) == ["class_histogram.png",
                                                                                 "confusion_matrix.png"]
    assert draw("/plot/kalfa/class_histogram", tmp_path, {"output": "zz"}, predictions=table, name="none") == [
        "class_histogram.png", "confusion_matrix.png"]


def test_torchmetrics_curves_write_their_files(tmp_path):
    table = classification_table()
    assert draw("/plot/torchmetrics/binary_roc", tmp_path, {"output": "y", "target": "is_hot"},
                predictions=table) == ["binary_roc.png"]
    assert draw("/plot/torchmetrics/binary_precision_recall_curve", tmp_path, predictions=table) == [
        "binary_precision_recall_curve.png", "binary_roc.png"]
    lone = table.assign(is_hot=1)
    assert draw("/plot/torchmetrics/binary_roc", tmp_path, predictions=lone, name="lone") == [
        "binary_precision_recall_curve.png", "binary_roc.png"]


def test_data_plots_read_the_set_the_definition_names(tmp_path):
    data, prep, loader = table_set("train")
    loaders = {"train": loader}
    assert draw("/plot/kalfa/correlation_heatmap", tmp_path, {"annotate": True}, loaders=loaders, prep=prep) == [
        "correlation_heatmap.png"]
    assert "feature_distributions.png" in draw("/plot/kalfa/feature_distributions", tmp_path,
                                               {"log": ["x0"], "bins": 10, "per_row": 2}, loaders=loaders,
                                               prep=prep, sets=["train"])
    assert "target_correlation.png" in draw("/plot/kalfa/target_correlation", tmp_path,
                                            {"groups": {"x0": "raw", "x1": "raw"}}, loaders=loaders, prep=prep)
    assert "target_vs_features.png" in draw("/plot/kalfa/target_vs_features", tmp_path,
                                            {"target": "price", "columns": ["x*"], "per_row": 2, "bins": 5},
                                            loaders=loaders, prep=prep)
    for name in ("correlation_heatmap.png", "feature_distributions.png", "target_correlation.png",
                 "target_vs_features.png"):
        assert non_empty(tmp_path, name)


def test_data_plots_skip_a_missing_set_or_target(tmp_path):
    data, prep, loader = table_set("train")
    assert draw("/plot/kalfa/correlation_heatmap", tmp_path, loaders={"train": loader}, prep=prep,
                sets=["valid"]) == []
    assert draw("/plot/kalfa/target_vs_features", tmp_path, {"target": "zz"}, loaders={"train": loader},
                prep=prep) == []
    assert draw("/plot/kalfa/feature_distributions", tmp_path, loaders={"train": loader}, prep=None) == []


def test_data_pipeline_draws_the_report_with_before_and_after_histograms(tmp_path):
    data, prep, loader = table_set("train")
    report = {"stages": [{"rows": 40, "columns": 4}, {"stage": "drop", "rows": 40, "columns": 4, "removed": ["junk"]}],
              "split": {"train": 28, "valid": 6, "test": 6}, "after_set_transforms": {"train": 28},
              "frames": ["group_statistic"],
              "fit": {"preprocessors": {"std": ["x0", "x1"]}, "features": 3, "targets": ["price"], "extras": ["late"]},
              "sets": {"train": {"rows": 28}}, "loaders": {"train": {"batches": 4, "size": 8}}}
    assert draw("/plot/kalfa/data_pipeline", tmp_path, {"columns": ["x0", "price"]}, data_report=report,
                train_df=data.data, train_frame=data, prep=prep) == ["data_pipeline.png"]
    assert non_empty(tmp_path, "data_pipeline.png")
    assert draw("/plot/kalfa/data_pipeline", tmp_path, data_report=None, name="none") == ["data_pipeline.png"]


def test_image_plots_draw_the_report_set_through_the_predicts_model(tmp_path):
    loader = build("/loader/kalfa/torch", data=Images(), set="test", eval_size=4)
    models = {"auto": image_model()}
    assert draw("/plot/kalfa/image_grid", tmp_path, {"n": 5}, models=models, loaders={"test": loader},
                predicts="auto") == ["image_grid.png"]
    assert draw("/plot/kalfa/image_pairs", tmp_path, {"n": 3}, models=models, loaders={"valid": loader},
                predicts="auto") == ["image_grid.png", "image_pairs.png"]
    assert draw("/plot/kalfa/image_grid", tmp_path, models=models, loaders={}, predicts="auto", name="none") == [
        "image_grid.png", "image_pairs.png"]


def test_architecture_draws_every_report_model_with_its_losses_and_optimizers(tmp_path):
    data, prep, loader = table_set("test")
    model = summed_model()
    ema = build("/lego/kalfa/clone", model=model, decay=0.5)
    losses = {"mse": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mse"))}
    optimizer = build("/optimizer/torch/sgd", models={"m": model}, params={"lr": 0.1}, loss="mse")
    files = draw("/plot/kalfa/architecture", tmp_path, models={"m": model, "m.ema": ema}, loaders={"test": loader},
                 predicts="m", losses=losses, losses_keys={"mse": {"output": "y", "target": "price"}},
                 optimizers={"main": optimizer}, prep=prep)
    assert files == ["architecture_m.png"]
    assert non_empty(tmp_path, "architecture_m.png")
    assert "arch_m.png" in draw("/plot/kalfa/architecture", tmp_path, models={"m": model}, name="arch")


def test_architecture_text_prints_the_module_repr_of_every_model(tmp_path):
    model = summed_model()
    assert draw("/plot/kalfa/architecture_text", tmp_path, models={"m": model}) == ["architecture_text.txt"]
    text = (tmp_path / "plots" / "architecture_text.txt").read_text()
    assert text.startswith("== m\n")
    assert repr(model) in text


def test_permutation_importance_scores_every_feature(tmp_path):
    data, prep, loader = table_set("test")
    files = draw("/plot/kalfa/permutation_importance", tmp_path, {"sample": 40, "repeats": 2, "seed": 1},
                 models={"m": summed_model()}, loaders={"test": loader}, prep=prep, predicts="m")
    assert files == ["permutation_importance.png"]
    assert non_empty(tmp_path, "permutation_importance.png")
    assert draw("/plot/kalfa/permutation_importance", tmp_path, models={"m": summed_model()}, loaders={},
                prep=prep, predicts="m", name="none") == ["permutation_importance.png"]


def test_samples_plots_read_the_turn_samples_of_the_record(tmp_path):
    write_turn_samples(tmp_path)
    assert sorted(path.name for path in (tmp_path / "samples").iterdir()) == ["turn_0001.png", "turn_0001.pt",
                                                                               "turn_0002.png", "turn_0002.pt"]
    assert draw("/plot/kalfa/samples_gif", tmp_path, {"duration": 100}) == ["samples_gif.gif"]
    assert draw("/plot/kalfa/samples_matrix", tmp_path, {"n": 3}) == ["samples_gif.gif", "samples_matrix.png"]
    assert non_empty(tmp_path, "samples_gif.gif") and non_empty(tmp_path, "samples_matrix.png")


def test_samples_plots_warn_without_turn_samples(tmp_path):
    with pytest.warns(UserWarning, match=r"samples_gif: no samples/turn_\*.png in the record; add a sample_writer"):
        assert draw("/plot/kalfa/samples_gif", tmp_path) == []
    with pytest.warns(UserWarning, match=r"samples_matrix: no samples/turn_\*.pt in the record"):
        assert draw("/plot/kalfa/samples_matrix", tmp_path) == []
    (tmp_path / "samples").mkdir()
    torch.save(torch.zeros(4, 2), tmp_path / "samples" / "turn_0001.pt")
    with pytest.warns(UserWarning, match=r"samples_matrix: the turn samples are not images"):
        assert draw("/plot/kalfa/samples_matrix", tmp_path) == []


def test_seaborn_kde_draws_one_or_two_columns(tmp_path):
    needs("seaborn")
    data, prep, loader = table_set("train")
    assert draw("/plot/seaborn/kde", tmp_path, {"x": "x0"}, loaders={"train": loader}, prep=prep) == ["kde.png"]
    assert "pair.png" in draw("/plot/seaborn/kde", tmp_path, {"x": "x0", "y": "x1"}, loaders={"train": loader},
                              prep=prep, name="pair")
    assert draw("/plot/seaborn/kde", tmp_path, {"x": "zz"}, loaders={"train": loader}, prep=prep, name="none") == [
        "kde.png", "pair.png"]


def test_seaborn_pairplot_draws_a_few_columns(tmp_path):
    needs("seaborn")
    data, prep, loader = table_set("train")
    assert draw("/plot/seaborn/pairplot", tmp_path, {"columns": ["x0", "x1"], "height": 1.5},
                loaders={"train": loader}, prep=prep) == ["pairplot.png"]
    assert non_empty(tmp_path, "pairplot.png")


def test_seaborn_violin_draws_one_column(tmp_path):
    needs("seaborn")
    data, prep, loader = table_set("train")
    assert draw("/plot/seaborn/violin", tmp_path, {"value": "x0"}, loaders={"train": loader}, prep=prep) == [
        "violin.png"]
    assert draw("/plot/seaborn/violin", tmp_path, loaders={"train": loader}, prep=prep, name="none") == ["violin.png"]


def test_torchview_draws_every_model_the_batch_feeds(tmp_path):
    needs("torchview")
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot is not installed")
    data, prep, loader = table_set("test")
    files = draw("/plot/torchview/architecture", tmp_path, models={"m": summed_model()}, loaders={"test": loader})
    assert files == ["architecture_m.png"]
    assert non_empty(tmp_path, "architecture_m.png")


def test_figures_validate_their_look_and_size_the_panels():
    figure = build("/lego/kalfa/figures", format="svg", width=3.0, height=2.0, dpi=40, style="none")
    assert isinstance(figure, Figures) and isinstance(figure, Figure)
    assert (figure.format, figure.width, figure.height, figure.dpi, figure.style) == ("svg", 3.0, 2.0, 40, "none")
    assert (figure.width_of(5.0), figure.height_of(5.0)) == (3.0, 2.0)
    plain = build("/lego/kalfa/figures")
    assert (plain.format, plain.width, plain.height, plain.dpi, plain.style) == ("png", None, None, 150, "kalfa")
    assert (plain.width_of(), plain.height_of()) == (6.4, 4.2)
    assert (plain.width_of(5.0), plain.height_of(5.0)) == (5.0, 5.0)
    sized = figure.with_size(4.0, None)
    assert (sized.width, sized.height, sized.format, sized.dpi, sized.style) == (4.0, 2.0, "svg", 40, "none")
    with pytest.raises(ValueError, match=r"figures.format must be one of \['png', 'pdf', 'svg'\], got 'bmp'"):
        build("/lego/kalfa/figures", format="bmp")
    with pytest.raises(ValueError, match=r"figures.style must be kalfa or none, got 'dark'"):
        build("/lego/kalfa/figures", style="dark")
    with pytest.raises(ValueError, match=r"figures.width must be a positive number, got -1"):
        build("/lego/kalfa/figures", width=-1)
    with pytest.raises(ValueError, match=r"figures.dpi must be a positive number, got True"):
        build("/lego/kalfa/figures", dpi=True)


def test_figures_format_names_the_file_suffix(tmp_path):
    plot = build("/plot/kalfa/loss_curve")
    plot(predictions=None, history=history_lines(), models={}, record=str(tmp_path),
         figures=build("/lego/kalfa/figures", format="svg", width=3.0, height=2.0))
    assert plots_of(tmp_path) == ["loss_curve.svg"]


def test_run_all_skips_a_plot_whose_need_is_not_on_the_bus(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="kalfa")
    plots = {"heat": build("/plot/kalfa/correlation_heatmap")}
    keys = {"heat": {"lego": "/plot/kalfa/correlation_heatmap"}}
    assert run_all(plots, keys, bus={"valid_loader": None}, record=str(tmp_path)) is None
    assert "heat skipped: the bus has no train_loader" in caplog.text
    assert plots_of(tmp_path) == []
    assert "plots: heat" in caplog.text


def test_run_all_types_the_inputs_by_the_refs_of_the_lego_and_offers_the_run(tmp_path):
    seen = []

    def probe(predictions, history, models, record, series=None, net=None, column=None, prep=None, loaders=None,
              predicts=None, sets=None, name=None, figures=None):
        seen.append({"predictions": predictions, "history": history, "models": models, "record": record,
                     "series": series, "net": net, "column": column, "prep": prep, "loaders": loaders,
                     "predicts": predicts, "sets": sets, "name": name, "figures": figures})

    lego("/plot/test/probe", probe, partial=True, refs={"series": "history", "net": "model", "column": "field"},
         needs=["prep"])
    model, other = tiny_model(), tiny_model(index=1)
    table = regression_table()
    keys = {"probe": {"lego": "/plot/test/probe", "inputs": {"series": "val/rmse", "net": "m", "column": "price"},
                      "sets": ["valid"], "width": 4.0, "height": 3.0}}
    bus = {"prep": "PREP", "train_loader": "L1", "valid_loader": None, "composites": {"full": other}, "device": "D"}
    run_all({"probe": build("/plot/test/probe")}, keys, predictions=table, history=history_lines(), models={"m": model},
            predicts="m", bus=bus, record=str(tmp_path), suffix="_valid")
    assert len(seen) == 1
    call = seen[0]
    assert call["predictions"] is table
    assert call["history"] == history_lines()
    assert call["models"] == {"full": other, "m": model}
    assert call["record"] == str(tmp_path)
    assert call["series"] == [2.0, 1.0, 2.0 / 3.0, 0.5]
    assert call["net"] is model
    assert call["column"].equals(table["price"])
    assert call["prep"] == "PREP"
    assert call["loaders"] == {"train": "L1"}
    assert call["predicts"] == "m"
    assert call["sets"] == ["valid"]
    assert call["name"] == "probe_valid"
    assert (call["figures"].width, call["figures"].height, call["figures"].dpi) == (4.0, 3.0, 40)
    run_all({"probe": build("/plot/test/probe")}, keys, predictions=table, history=history_lines(), models={"m": model},
            bus={}, record=str(tmp_path))
    assert len(seen) == 1


def test_run_all_reads_the_history_of_the_record_and_untyped_inputs_as_names(tmp_path):
    seen = []

    def probe(predictions, history, models, record, series=None, label=None, name=None):
        seen.append({"history": history, "series": series, "label": label, "name": name})

    lego("/plot/test/probe", probe, partial=True, refs={"series": "history"})
    lines = history_lines(2)
    (tmp_path / "history.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    keys = {"probe": {"lego": "/plot/test/probe", "inputs": {"series": "train/mse", "label": "raw"}}}
    run_all({"probe": build("/plot/test/probe")}, keys, history=history_lines(4), record=str(tmp_path))
    assert seen == [{"history": lines, "series": [1.0, 0.5], "label": "raw", "name": "probe"}]
    run_all({"probe": build("/plot/test/probe")}, None, history=history_lines(1), record=None)
    assert seen[1]["history"] == history_lines(1) and seen[1]["series"] is None


def test_history_step_appends_the_turn_line_and_hands_it_to_the_monitor(tmp_path):
    optimizer = build("/optimizer/torch/sgd", models={"m": tiny_model()},
                      params={"lr": 0.1, "groups": [{"name": "bias", "match": "*.bias", "lr": -0.05}]}, loss="mse")
    aux = build("/optimizer/torch/sgd", models={"a": tiny_model(index=1)}, params={"lr": 0.2})
    monitor = Monitor()
    effects = {"main.loss": "mae", "main.lr": 0.5, "ws.terms": {"a": 2.0}, "loss": "x", "m.trainable": False}
    result = build("/lego/kalfa/history", monitor=monitor, metrics={"val/rmse": 0.5, "train/mse": 0.25},
                   turn_index=1, counters_next={"turn": 1, "global_step": 4},
                   optimizers_next={"main": optimizer, "aux": aux}, rules_next={"fired": ["warm"]}, effects=effects,
                   record=str(tmp_path))
    assert result is None
    expected = {"turn": 1, "global_step": 4, "val/rmse": 0.5, "train/mse": 0.25, "lr/main": 0.1,
                "lr/main/bias": -0.05, "lr/aux": 0.2, "minimizes/main": "mae", "effect/main.lr": 0.5,
                "effect/ws.terms": {"a": 2.0}, "effect/m.trainable": False, "seconds": 1.5, "rules": ["warm"]}
    assert monitor.lines == [expected]
    lines = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert lines == [expected]
    assert list(lines[0]) == list(expected)


def test_history_step_without_a_monitor_or_record_writes_nothing(tmp_path):
    assert build("/lego/kalfa/history", metrics={"val/rmse": 0.5}, counters_next={"turn": 2}) is None
    assert list(tmp_path.iterdir()) == []
    monitor = Monitor()
    build("/lego/kalfa/history", monitor=monitor, metrics=None, counters_next=None, optimizers_next=None,
          rules_next=None, effects=None)
    assert monitor.lines == [{"turn": 0, "global_step": 0, "seconds": 1.5, "rules": []}]


def test_evaluate_step_reports_losses_and_metrics_of_a_set():
    data, prep, loader = table_set("test", rows=20)
    model = summed_model()
    model.train()
    losses = {"mse": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mse"))}
    metrics = {"rmse": build("/adapter/kalfa/metric", metric=build("/metric/kalfa/rmse"))}
    out = build("/lego/kalfa/evaluate", models={"m": model}, emas={}, composites={}, counters={"turn": 1},
                loader=loader, set="test", losses=losses, metrics=metrics, losses_keys={}, metrics_keys={},
                predicts="m", prep=prep)
    guess = loader.dataset.x.sum(dim=1)
    mse = float(((guess - loader.dataset.fields["price"]) ** 2).mean())
    assert out == {"mse": pytest.approx(mse), "rmse": pytest.approx(mse ** 0.5)}
    assert model.training is False


def test_evaluate_step_paces_entries_by_every_and_sets():
    data, prep, loader = table_set("valid", rows=16)
    losses = {"mse": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mse")),
              "mae": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mae"))}
    keys = {"mae": {"every": 2, "sets": ["valid", "test"]}}
    first = build("/lego/kalfa/evaluate", models={"m": summed_model()}, emas={}, composites={}, counters={"turn": 1},
                  loader=loader, set="valid", losses=losses, metrics={}, losses_keys=keys, metrics_keys={},
                  predicts="m")
    assert set(first) == {"mse"}
    second = build("/lego/kalfa/evaluate", models={"m": summed_model()}, emas={}, composites={}, counters={"turn": 2},
                   loader=loader, set="valid", losses=losses, metrics={}, losses_keys=keys, metrics_keys={},
                   predicts="m")
    assert set(second) == {"mse", "mae"}
    train = build("/lego/kalfa/evaluate", models={"m": summed_model()}, emas={}, composites={}, counters={"turn": 2},
                  loader=loader, set="train", losses={"mae": losses["mae"]}, metrics={}, losses_keys=keys,
                  metrics_keys={}, predicts="m")
    assert train == {}


def test_evaluate_step_of_an_empty_set_is_empty():
    empty = build("/loader/kalfa/torch", data=build("/feed/kalfa/table", frame=frame(rows=0, set_name="test")),
                  set="test", eval_size=8)
    losses = {"mse": build("/adapter/kalfa/criterion", criterion=build("/criterion/kalfa/mse"))}
    assert build("/lego/kalfa/evaluate", models={"m": summed_model()}, emas={}, composites={}, counters={},
                 loader=empty, set="test", losses=losses, metrics={}, losses_keys={}, metrics_keys={},
                 predicts="m") == {}
    assert build("/lego/kalfa/evaluate", models={"m": summed_model()}, emas={}, composites={}, counters={},
                 loader=None, set="test", losses=losses, metrics={}, losses_keys={}, metrics_keys={},
                 predicts="m") == {}


def predict(models, loader, prep, set_name, target_map=None, calibrations=None, record=None, composites=None,
            predicts="m"):
    return build("/lego/kalfa/predict", models=models, composites=composites or {}, loader=loader, prep=prep,
                 predicts=predicts, set=set_name, target_map=target_map, calibrations=calibrations, record=record,
                 device=None)


def spectator_set(set_name="test", rows=12, targets=("price",)):
    data = frame(rows=rows, seed=2, set_name=set_name)
    table = data.data
    for target in targets:
        if target not in table.columns:
            table[target] = (table["price"] * 0.5).astype("float32")
    extra = pandas.DataFrame({"sample_id": 1000 + numpy.arange(rows)}, index=table.index)
    shaped = TableFrame(data.features, {target: [target] for target in targets}, set_name, data=table, extra=extra)
    loader = build("/loader/kalfa/torch", data=build("/feed/kalfa/table", frame=shaped), set=set_name, eval_size=5)
    return shaped, prep_for(shaped, targets), loader


def test_predict_step_writes_the_prediction_table_of_the_test_set(tmp_path):
    shaped, prep, loader = spectator_set()
    model = summed_model()
    table = predict({"m": model}, loader, prep, "test", record=str(tmp_path))
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y", "sample_id"]
    assert table["row"].tolist() == list(range(12))
    assert numpy.allclose(table["price"].to_numpy(), shaped.data["price"].to_numpy())
    assert numpy.allclose(table["raw_y"].to_numpy(), loader.dataset.x.sum(dim=1).numpy())
    assert numpy.array_equal(table["pred_y"].to_numpy(), table["raw_y"].to_numpy())
    assert table["sample_id"].tolist() == list(range(1000, 1012))
    written = pandas.read_parquet(tmp_path / "predictions.parquet")
    assert written.equals(table)


def test_predict_step_names_another_set_and_applies_the_calibrations(tmp_path):
    shaped, prep, loader = spectator_set("valid")
    threshold = build("/calibrate/kalfa/threshold", set="valid", quantile=0.5)
    threshold.fit({"m": summed_model()}, {"valid": loader}, None, Device.cpu(), "m")
    table = predict({"m": summed_model()}, loader, prep, "valid", calibrations={"cut": threshold}, record=str(tmp_path))
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y", "sample_id", "flag_y"]
    assert table["flag_y"].tolist() == (table["raw_y"] > threshold.threshold).tolist()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["predictions_valid.parquet"]


def test_predict_step_names_the_pred_columns_after_the_target_map():
    shaped, prep, loader = spectator_set(targets=("price", "size"))
    wide = predict({"m": summed_model(2)}, loader, prep, "test", target_map={"y": ["price", "size"]})
    assert list(wide.columns) == ["row", "price", "size", "raw_y_0", "raw_y_1", "pred_y_price", "pred_y_size",
                                  "sample_id"]
    assert numpy.allclose(wide["pred_y_price"].to_numpy(), wide["raw_y_0"].to_numpy())
    assert numpy.allclose(wide["pred_y_size"].to_numpy(), wide["raw_y_1"].to_numpy())
    plain = predict({"m": summed_model(2)}, loader, prep, "test")
    assert list(plain.columns) == ["row", "price", "size", "raw_y_0", "raw_y_1", "pred_y_price", "pred_y_size",
                                   "sample_id"]
    narrow = predict({"m": summed_model()}, loader, prep, "test", target_map={"y": "size"})
    assert list(narrow.columns) == ["row", "price", "size", "raw_y", "pred_y_size", "sample_id"]


def test_predict_step_reaches_a_composite_and_is_empty_without_rows(tmp_path):
    shaped, prep, loader = spectator_set()
    table = predict({}, loader, prep, "test", composites={"full": summed_model()}, predicts="full")
    assert list(table.columns) == ["row", "price", "raw_y", "pred_y", "sample_id"]
    empty = build("/loader/kalfa/torch", data=build("/feed/kalfa/table", frame=frame(rows=0, set_name="test")),
                  set="test", eval_size=8)
    assert predict({"m": summed_model()}, empty, prep, "test", record=str(tmp_path)).empty
    assert predict({"m": summed_model()}, loader, prep, "test", predicts=None, record=str(tmp_path)).empty
    assert list(tmp_path.iterdir()) == []


def test_generate_step_is_a_no_op_without_a_generate_section(tmp_path):
    assert build("/lego/kalfa/generate", models={"g": ImageMaker()}, composites={}, prep=None, generate=None,
                 record=str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []


def test_generate_step_writes_the_samples_of_the_sampler(tmp_path):
    seen = {}

    def sampler(models, prep, rng):
        seen.update({"models": models, "prep": prep, "rng": rng})
        return torch.rand(3, 1, 2, 2, generator=rng)

    other = tiny_model()
    assert build("/lego/kalfa/generate", models={"g": ImageMaker()}, composites={"full": other}, prep="PREP",
                 generate=sampler, record=str(tmp_path)) is None
    assert list(seen["models"]) == ["full", "g"] and seen["prep"] == "PREP"
    assert isinstance(seen["rng"], torch.Generator)
    assert sorted(path.name for path in (tmp_path / "samples").iterdir()) == ["grid.png", "samples.pt"]
    assert torch.load(tmp_path / "samples" / "samples.pt", weights_only=False).shape == (3, 1, 2, 2)
    build("/lego/kalfa/generate", models={}, composites={}, prep=None, generate=lambda models, prep, rng: "text",
          record=str(tmp_path))
    assert (tmp_path / "samples" / "samples.txt").read_text() == "text"
