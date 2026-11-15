import json
from types import SimpleNamespace

import matplotlib
import numpy
import pandas
import pytest
import torch
from matplotlib import pyplot

from kalfa.core.aliases import ALIASES
from kalfa.core.device import Device
from kalfa.core.errors import KalfaError
from kalfa.core.loop import takes_context
from kalfa.records import stats
from kalfa.std import plots
from kalfa.std.figure import Figures
from records.fixtures import make_record as fixture_record


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_lines(path, lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def history_lines(turns):
    return [{"turn": turn, "update": turn * 4, "train/mse": 1.0 / turn, "train/elbo/kl": 0.1 * turn,
             "val/mse": 1.2 / turn, "val/rmse": 1.1 / turn, "test/rmse": 1.0 / turn, "lr/main": 1e-3,
             "minimizes/main": "losses.mse", "track/losses.total.lmbda": 0.5, "seconds": 0.3, "rules": []}
            for turn in range(1, turns + 1)]


def make_record(directory, targets, frame, turns=3, unit="epochs", history=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", {"format": 1, "name": directory.name, "started": "2026-11-07T10:00:00",
                                             "status": "done", "unit": unit, "turns_planned": turns})
    write_json(directory / "plan.json", {
        "format": 1,
        "nodes": {"data": {"slot": "data", "use": "kalfa.std.data.table.Table",
                           "params": {"file": "housing.parquet", "targets": targets}}},
        "train": {"epochs": turns, "checkpoint": {"watch": "val/rmse"}}})
    write_lines(directory / "history.jsonl", history_lines(turns) if history is None else history)
    if frame is not None:
        frame.to_parquet(directory / "predictions.parquet", index=False)
    return directory


class Spy(Figures):
    seen = None

    def save(self, figure, name):
        self.seen = figure
        return super().save(figure, name)


class Holder:
    def __init__(self, directory):
        self.directory = directory


@pytest.fixture
def regression(tmp_path):
    rng = numpy.random.default_rng(0)
    price = rng.normal(200.0, 50.0, 120)
    frame = pandas.DataFrame({"row": numpy.arange(120), "price": price, "pred_net": price + rng.normal(0, 10, 120),
                              "raw_net": rng.normal(0, 1, 120), "city": ["a", "b"] * 60})
    return make_record(tmp_path / "regression", {"price": "standard_scaler"}, frame)


@pytest.fixture
def classification(tmp_path):
    rng = numpy.random.default_rng(1)
    classes = numpy.array(["stay", "churn", "pause"])
    truth = rng.choice(classes, 90)
    guess = numpy.where(rng.random(90) < 0.7, truth, rng.choice(classes, 90))
    frame = pandas.DataFrame({"row": numpy.arange(90), "churned": truth, "pred_net": guess,
                              "raw_net": rng.random(90)})
    return make_record(tmp_path / "classification", {"churned": "label"}, frame)


def test_figures_use_agg_create_sized_figures_and_save_closed(tmp_path):
    assert matplotlib.get_backend().lower() == "agg"
    figures = Figures(tmp_path / "plots", dpi=72)
    figure = figures.new(5, 3)
    assert tuple(figure.get_size_inches()) == (5.0, 3.0)
    figure.subplots().plot([1, 2, 3], [3, 2, 1])
    path = figures.save(figure, "nested/one")
    assert path == tmp_path / "plots" / "nested" / "one.png"
    assert path.is_file() and path.stat().st_size > 0
    assert pyplot.get_fignums() == []
    vector = Figures(tmp_path / "plots", format="svg")
    assert vector.save(vector.new(), "two") == tmp_path / "plots" / "two.svg"
    assert (tmp_path / "plots" / "two.svg").is_file()
    assert pyplot.get_fignums() == []


def test_figures_close_even_when_saving_fails(tmp_path):
    figures = Figures(tmp_path / "plots", format="nope")
    with pytest.raises(ValueError):
        figures.save(figures.new(), "bad")
    assert pyplot.get_fignums() == []
    assert not (tmp_path / "plots" / "bad.nope").exists()


def test_loss_curve_writes_png_and_closes(regression, tmp_path):
    figures = Spy(tmp_path / "plots")
    assert plots.loss_curve(regression, figures) is None
    assert (tmp_path / "plots" / "loss_curve.png").is_file()
    assert pyplot.get_fignums() == []
    axis = figures.seen.axes[0]
    assert [line.get_label() for line in axis.lines] == ["train/mse", "train/elbo/kl", "val/mse", "val/rmse"]
    assert list(axis.lines[0].get_xdata()) == [1, 2, 3]
    assert axis.get_yscale() == "linear"
    assert axis.get_legend() is not None
    assert axis.get_xlabel() == "epoch"
    assert plots.loss_curve(Holder(regression), figures) is None
    assert pyplot.get_fignums() == []


def test_loss_curve_log_scale_legend_limit_and_named_series(tmp_path):
    history = [{"turn": turn, "train/loss": 10.0 ** (3 - turn), "val/loss": 2.0 * 10.0 ** (3 - turn),
                "lr/main": 1e-3 * turn, **{f"val/m{index}": float(index + 1) for index in range(12)}}
               for turn in range(1, 4)]
    record = make_record(tmp_path / "wide", {"y": None}, None, history=history, unit="updates")
    figures = Spy(tmp_path / "plots")
    plots.loss_curve(record, figures)
    axis = figures.seen.axes[0]
    assert len(axis.lines) == 14
    assert axis.get_legend() is None
    assert axis.get_yscale() == "log"
    assert axis.get_xlabel() == "turn"
    plots.loss_curve(record, figures, series=["lr/main", "val/none"])
    axis = figures.seen.axes[0]
    assert [line.get_label() for line in axis.lines] == ["lr/main"]
    assert axis.get_yscale() == "linear"
    assert axis.get_legend() is not None
    plots.loss_curve(record, figures, series="train/loss")
    assert [line.get_label() for line in figures.seen.axes[0].lines] == ["train/loss"]
    assert pyplot.get_fignums() == []


def test_loss_curve_without_data(tmp_path):
    record = make_record(tmp_path / "empty", {"y": None}, None, history=[])
    figures = Figures(tmp_path / "plots")
    assert plots.loss_curve(record, figures) is None
    assert plots.loss_curve(record, figures, series=["val/none"]) is None
    assert plots.loss_curve(tmp_path / "nowhere", figures) is None
    assert not (tmp_path / "plots").exists()
    assert pyplot.get_fignums() == []


def test_pred_vs_true_default_pair(regression, tmp_path):
    figures = Spy(tmp_path / "plots")
    assert plots.pred_vs_true(regression, figures) is None
    assert (tmp_path / "plots" / "pred_vs_true.png").is_file()
    assert pyplot.get_fignums() == []
    assert len(figures.seen.axes) == 1
    axis = figures.seen.axes[0]
    assert len(axis.collections) == 1
    assert [line.get_label() for line in axis.lines] == ["y = x"]
    assert axis.get_xlabel() == "true price"
    assert axis.get_ylabel() == "predicted (net)"
    assert axis.get_title(loc="left").startswith("price (R² ")
    assert axis.get_title(loc="left").endswith("n = 120)")


def test_pred_vs_true_matches_names_and_takes_explicit_pairs(tmp_path):
    rng = numpy.random.default_rng(2)
    frame = pandas.DataFrame({"row": numpy.arange(50), "price": rng.normal(size=50), "area": rng.normal(size=50),
                              "pred_price": rng.normal(size=50), "pred_area": rng.normal(size=50),
                              "pred_other": rng.normal(size=50)})
    record = make_record(tmp_path / "two", {"price": "standard_scaler", "area": None}, frame)
    figures = Spy(tmp_path / "plots")
    plots.pred_vs_true(record, figures)
    assert [axis.get_title(loc="left").split(" (")[0] for axis in figures.seen.axes] == ["price", "area"]
    plots.pred_vs_true(record, figures, pairs={"other": "price"})
    assert [axis.get_ylabel() for axis in figures.seen.axes] == ["predicted (other)"]
    plots.pred_vs_true(record, figures, pairs={"pred_other": "area", "price": "price", "area": "price"})
    assert [axis.get_ylabel() for axis in figures.seen.axes] == ["predicted (other)", "predicted (price)",
                                                                   "predicted (area)"]
    assert pyplot.get_fignums() == []


def test_pred_vs_true_without_numeric_pairs(classification, tmp_path):
    figures = Figures(tmp_path / "plots")
    assert plots.pred_vs_true(classification, figures) is None
    assert plots.pred_vs_true(tmp_path / "nowhere", figures) is None
    assert not (tmp_path / "plots").exists()
    assert pyplot.get_fignums() == []


def test_confusion_matrix(classification, tmp_path):
    figures = Spy(tmp_path / "plots")
    assert plots.confusion_matrix(classification, figures) is None
    assert (tmp_path / "plots" / "confusion_matrix.png").is_file()
    assert pyplot.get_fignums() == []
    axis = figures.seen.axes[0]
    assert [label.get_text() for label in axis.get_xticklabels()] == ["churn", "pause", "stay"]
    assert [label.get_text() for label in axis.get_yticklabels()] == ["churn", "pause", "stay"]
    assert len(axis.texts) == 9
    assert sum(int(text.get_text().split("\n")[0].replace(",", "")) for text in axis.texts) == 90
    assert axis.get_xlabel() == "predicted (net)"
    assert axis.get_ylabel() == "true churned"
    assert axis.get_title(loc="left") == "Confusion matrix (churned, n = 90)"


def test_confusion_matrix_named_pair_and_guards(classification, regression, tmp_path):
    figures = Spy(tmp_path / "plots")
    plots.confusion_matrix(classification, figures, pred="net", target="churned")
    assert figures.seen.axes[0].get_ylabel() == "true churned"
    plots.confusion_matrix(classification, figures, pred="pred_net")
    assert figures.seen.axes[0].get_xlabel() == "predicted (net)"
    assert plots.confusion_matrix(classification, figures, pred="missing") is None
    assert plots.confusion_matrix(classification, figures, target="city") is None
    assert plots.confusion_matrix(regression, Figures(tmp_path / "other")) is None
    assert not (tmp_path / "other").exists()
    assert pyplot.get_fignums() == []


def test_confusion_matrix_with_integer_labels(tmp_path):
    frame = pandas.DataFrame({"row": range(6), "digit": [0, 1, 2, 0, 1, 2],
                              "pred_net": [0.0, 1.0, 1.0, 0.0, 2.0, 2.0]})
    record = make_record(tmp_path / "digits", {"digit": "label"}, frame)
    figures = Spy(tmp_path / "plots")
    plots.confusion_matrix(record, figures)
    axis = figures.seen.axes[0]
    assert [label.get_text() for label in axis.get_xticklabels()] == ["0", "1", "2"]
    assert [text.get_text().split("\n")[0] for text in axis.texts] == ["2", "0", "0", "0", "1", "1", "0", "1", "1"]
    assert pyplot.get_fignums() == []


def test_plots_are_registered():
    for name in ("loss_curve", "pred_vs_true", "confusion_matrix"):
        entry = ALIASES[name]
        assert entry.slot == "plot"
        assert entry.target == f"kalfa.std.plots.{name}"


def image_context(count=20, batch=6, channels=1, device=None, predict=None):
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(count, channels, 8, 8, generator=generator)
    items = [{"image": images[index], "label": torch.tensor(index % 2)} for index in range(count)]
    loader = torch.utils.data.DataLoader(items, batch_size=batch)
    seen, modes = [], []

    def flat(batch, ctx):
        seen.append(len(batch["image"]))
        return {"x_hat": 1.0 - batch["image"].flatten(1), "code": torch.zeros(len(batch["image"]), 2)}

    ctx = SimpleNamespace(data=SimpleNamespace(inputs=("image",)), loaders=lambda: {"train": loader, "test": loader},
                          step=SimpleNamespace(predict=predict or flat), device=device, set_modes=modes.append)
    return ctx, seen, modes


def test_image_pairs_needs_the_run(tmp_path):
    with pytest.raises(KalfaError, match="image_pairs needs the run; redraw it with the record open"):
        plots.image_pairs(tmp_path, Figures(tmp_path / "plots"))
    assert takes_context(plots.image_pairs)
    assert not takes_context(plots.samples_grid)
    assert pyplot.get_fignums() == []


def test_image_pairs_draws_inputs_above_flat_outputs(tmp_path):
    ctx, seen, modes = image_context()
    figures = Spy(tmp_path / "plots")
    assert plots.image_pairs(tmp_path / "run", figures, n=8, ctx=ctx) is None
    assert (tmp_path / "plots" / "image_pairs.png").is_file()
    assert pyplot.get_fignums() == []
    assert seen == [6, 6]
    assert modes == [False]
    axes = figures.seen.axes
    assert len(axes) == 16
    top, bottom = axes[0].images[0].get_array(), axes[8].images[0].get_array()
    assert top.shape == (8, 8) and bottom.shape == (8, 8)
    total = numpy.asarray(top) + numpy.asarray(bottom)
    assert numpy.allclose(total, total[0, 0])
    assert axes[0].get_ylabel() == "image"
    assert axes[8].get_ylabel() == "x_hat"
    assert figures.seen.get_suptitle() == "image and x_hat (test, n = 8)"


def test_image_pairs_moves_batches_and_names_its_errors(tmp_path):
    ctx, seen, _ = image_context(count=5, batch=4, channels=3, device=Device("cpu"))
    figures = Spy(tmp_path / "plots")
    plots.image_pairs(tmp_path, figures, n=3, set="train", ctx=ctx)
    assert seen == [4]
    assert len(figures.seen.axes) == 6
    assert figures.seen.axes[0].images[0].get_array().shape == (8, 8, 3)
    assert figures.seen.get_suptitle() == "image and x_hat (train, n = 3)"
    with pytest.raises(KalfaError, match="no 'valid' set"):
        plots.image_pairs(tmp_path, figures, set="valid", ctx=ctx)
    wide, _, _ = image_context(predict=lambda batch, ctx: {"code": torch.zeros(len(batch["image"]), 5)})
    with pytest.raises(KalfaError, match="output 'code' of shape \\(6, 5\\) cannot be drawn"):
        plots.image_pairs(tmp_path, figures, ctx=wide)
    empty, _, _ = image_context(predict=lambda batch, ctx: {})
    with pytest.raises(KalfaError, match="predicts no tensor"):
        plots.image_pairs(tmp_path, figures, ctx=empty)
    assert pyplot.get_fignums() == []


def test_samples_grid_draws_the_first_n_images_in_a_near_square_grid(tmp_path):
    directory = tmp_path / "run"
    (directory / "samples").mkdir(parents=True)
    torch.save(torch.randn(16, 1, 8, 8), directory / "samples" / "samples.pt")
    torch.save([torch.rand(3, 8, 8) for _ in range(5)], directory / "samples" / "turn_2.pt")
    figures = Spy(tmp_path / "plots")
    assert plots.samples_grid(directory, figures, n=9) is None
    assert (tmp_path / "plots" / "samples_grid.png").is_file()
    assert pyplot.get_fignums() == []
    axis = figures.seen.axes[0]
    assert axis.images[0].get_array().shape == (32, 32, 3)
    assert axis.get_title(loc="left") == "samples.pt (9 of 16 samples)"
    plots.samples_grid(Holder(directory), figures, file="turn_2")
    axis = figures.seen.axes[0]
    assert axis.images[0].get_array().shape == (22, 32, 3)
    assert axis.get_title(loc="left") == "turn_2.pt (5 of 5 samples)"
    plots.samples_grid(directory, figures, n=2, file="turn_2.pt")
    axis = figures.seen.axes[0]
    assert axis.images[0].get_array().shape == (12, 22, 3)
    assert axis.get_title(loc="left") == "turn_2.pt (2 of 5 samples)"
    assert pyplot.get_fignums() == []


def test_samples_grid_draws_text_and_names_missing_files(tmp_path):
    directory = tmp_path / "lm"
    (directory / "samples").mkdir(parents=True)
    lines = [f"line {index}: the fox costs $5 and $6" for index in range(30)]
    lines[3] = "x" * 200
    (directory / "samples" / "samples.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures = Spy(tmp_path / "plots")
    assert plots.samples_grid(directory, figures) is None
    assert (tmp_path / "plots" / "samples_grid.png").is_file()
    axis = figures.seen.axes[0]
    shown = axis.texts[0].get_text().splitlines()
    assert len(shown) == 20
    assert shown[3] == "x" * 117 + "..."
    assert shown[:3] == lines[:3] and shown[4:] == lines[4:20]
    assert axis.texts[0].get_fontfamily() == ["monospace"]
    assert axis.get_title(loc="left") == "samples.txt (first 20 of 30 lines)"
    with pytest.raises(KalfaError, match="samples/nothing.pt or samples/nothing.txt is missing"):
        plots.samples_grid(directory, figures, file="nothing")
    torch.save({"a": 1}, directory / "samples" / "odd.pt")
    with pytest.raises(KalfaError, match="odd.pt holds a dict"):
        plots.samples_grid(directory, figures, file="odd")
    torch.save(torch.zeros(4, 5), directory / "samples" / "flat.pt")
    with pytest.raises(KalfaError, match="shape \\(4, 5\\)"):
        plots.samples_grid(directory, figures, file="flat")
    assert pyplot.get_fignums() == []


def horizon_frame(rows=40, steps=6, target="price", output="net"):
    rng = numpy.random.default_rng(3)
    frame = pandas.DataFrame({"row": numpy.arange(rows)[::-1] + 100})
    for step in range(steps):
        truth = numpy.sin((numpy.arange(rows)[::-1] + step) / 5.0)
        frame[f"{target}_{step}"] = truth
        frame[f"pred_{output}_{step}"] = truth + rng.normal(0.0, 0.1, rows)
    frame["site"] = "a"
    return frame


def test_forecast_samples_draws_evenly_spaced_windows(tmp_path):
    directory = fixture_record(tmp_path / "energy", predictions=False)
    frame = horizon_frame()
    frame.to_parquet(directory / "predictions.parquet", index=False)
    figures = Spy(tmp_path / "plots")
    assert plots.forecast_samples(directory, figures, n=4) is None
    assert (tmp_path / "plots" / "forecast_samples.png").is_file()
    assert pyplot.get_fignums() == []
    axes = figures.seen.axes
    assert len(axes) == 4
    assert [axis.get_title(loc="left") for axis in axes] == ["row 100", "row 113", "row 126", "row 139"]
    first = frame.set_index("row").loc[100]
    assert list(axes[0].lines[0].get_xdata()) == [1, 2, 3, 4, 5, 6]
    assert numpy.allclose(axes[0].lines[0].get_ydata(), [first[f"price_{step}"] for step in range(6)])
    assert numpy.allclose(axes[0].lines[1].get_ydata(), [first[f"pred_net_{step}"] for step in range(6)])
    assert [line.get_label() for line in axes[0].lines] == ["true", "predicted"]
    assert axes[0].get_legend() is not None
    assert [axis.get_xlabel() for axis in axes] == ["", "", "steps ahead", "steps ahead"]
    assert figures.seen.get_suptitle() == "Forecast windows (price against net, 6 steps, 4 of 40 rows)"


def test_forecast_samples_finds_other_names_and_odd_counts(tmp_path):
    directory = fixture_record(tmp_path / "energy", predictions=False)
    horizon_frame(rows=12, steps=3, target="load", output="y").to_parquet(directory / "predictions.parquet",
                                                                         index=False)
    figures = Spy(tmp_path / "plots")
    plots.forecast_samples(directory, figures, n=3)
    axes = figures.seen.axes
    assert len(axes) == 4
    assert [axis.axison for axis in axes] == [True, True, True, False]
    assert figures.seen.get_suptitle() == "Forecast windows (load against y, 3 steps, 3 of 12 rows)"
    plots.forecast_samples(directory, figures, n=1, pred="pred_y", target="load")
    assert len(figures.seen.axes) == 1
    assert figures.seen.axes[0].get_title(loc="left") == "row 100"
    assert plots.forecast_samples(directory, figures, pred="other") is None
    assert pyplot.get_fignums() == []


def test_forecast_samples_falls_back_to_single_columns(tmp_path):
    directory = fixture_record(tmp_path / "reg")
    figures = Spy(tmp_path / "plots")
    assert plots.forecast_samples(directory, figures) is None
    assert (tmp_path / "plots" / "forecast_samples.png").is_file()
    axis = figures.seen.axes[0]
    assert len(figures.seen.axes) == 1
    assert [line.get_label() for line in axis.lines] == ["true", "predicted"]
    assert list(axis.lines[0].get_xdata()) == list(range(50))
    assert axis.get_title(loc="left") == "price over rows (n = 50)"
    bare = fixture_record(tmp_path / "bare", predictions=False)
    assert plots.forecast_samples(bare, Figures(tmp_path / "none")) is None
    assert not (tmp_path / "none").exists()
    assert pyplot.get_fignums() == []


def binary_frame(majority, minority, seed=4):
    rng = numpy.random.default_rng(seed)
    truth = numpy.array(["a"] * majority + ["b"] * minority)
    positive = numpy.concatenate([rng.beta(2, 8, majority), rng.beta(8, 2, minority)])
    return pandas.DataFrame({"row": numpy.arange(len(truth)), "label": truth,
                             "pred_net": numpy.where(positive > 0.5, "b", "a"),
                             "raw_net": (positive > 0.5).astype("int64"), "prob_net_a": 1.0 - positive,
                             "prob_net_b": positive})


def test_class_histogram_draws_one_histogram_per_true_class(tmp_path):
    directory = fixture_record(tmp_path / "binary", classes=["a", "b"])
    frame = pandas.read_parquet(directory / "predictions.parquet")
    support = frame["label"].value_counts()
    figures = Spy(tmp_path / "plots")
    assert plots.class_histogram(directory, figures, bins=10) is None
    assert (tmp_path / "plots" / "class_histogram.png").is_file()
    assert pyplot.get_fignums() == []
    axis = figures.seen.axes[0]
    assert len(axis.patches) == 2
    assert [int(patch.get_data().values.sum()) for patch in axis.patches] == [support["a"], support["b"]]
    assert len(axis.patches[0].get_data().edges) == 11
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [f"a (n = {support['a']})",
                                                                         f"b (n = {support['b']})"]
    assert axis.get_xlabel() == "P(b)"
    assert axis.get_yscale() == "linear"
    assert axis.get_title(loc="left") == "Score by true class (label, n = 50)"


def test_class_histogram_log_scale_raw_scores_and_guards(tmp_path):
    directory = fixture_record(tmp_path / "skewed", classes=["a", "b"])
    frame = binary_frame(2000, 6)
    frame.to_parquet(directory / "predictions.parquet", index=False)
    figures = Spy(tmp_path / "plots")
    plots.class_histogram(directory, figures)
    assert figures.seen.axes[0].get_yscale() == "log"
    frame[["row", "label", "pred_net", "raw_net"]].to_parquet(directory / "predictions.parquet", index=False)
    plots.class_histogram(directory, figures, pred="net", target="label")
    assert figures.seen.axes[0].get_xlabel() == "raw_net"
    frame[["row", "label", "pred_net"]].to_parquet(directory / "predictions.parquet", index=False)
    with pytest.raises(KalfaError, match="carry no probabilities"):
        plots.class_histogram(directory, figures)
    three = fixture_record(tmp_path / "three", classes=["cold", "mild", "hot"])
    plots.class_histogram(three, figures)
    axis = figures.seen.axes[0]
    assert axis.get_xlabel() == "P(hot)"
    assert [text.get_text().split(" ")[0] for text in axis.get_legend().get_texts()] == ["cold", "mild", "hot"]
    with pytest.raises(KalfaError, match="carry no probabilities"):
        plots.class_histogram(fixture_record(tmp_path / "reg"), figures)
    assert plots.class_histogram(fixture_record(tmp_path / "bare", predictions=False), figures) is None
    assert plots.class_histogram(three, figures, pred="missing") is None
    assert pyplot.get_fignums() == []


def test_binary_roc_draws_the_curve_with_its_auc(tmp_path):
    directory = fixture_record(tmp_path / "binary", classes=["a", "b"])
    expected = stats.prediction_stats(directory)["pairs"][0]["stats"]["roc"]
    figures = Spy(tmp_path / "plots")
    assert plots.binary_roc(directory, figures) is None
    assert (tmp_path / "plots" / "binary_roc.png").is_file()
    assert pyplot.get_fignums() == []
    axis = figures.seen.axes[0]
    assert numpy.allclose(axis.lines[0].get_xdata(), expected["fpr"])
    assert numpy.allclose(axis.lines[0].get_ydata(), expected["tpr"])
    assert [line.get_label() for line in axis.lines] == [f"net (AUC {expected['auc']:.3f})", "chance"]
    assert axis.get_title(loc="left") == f"ROC (label = b, AUC {expected['auc']:.3f}, n = 50)"
    assert axis.get_xlabel() == "false positive rate"


def test_binary_precision_recall_curve_draws_the_curve_with_its_ap(tmp_path):
    directory = fixture_record(tmp_path / "binary", classes=["a", "b"])
    found = stats.prediction_stats(directory)["pairs"][0]["stats"]
    expected = found["pr"]
    rate = found["per_class"]["b"]["support"] / 50
    figures = Spy(tmp_path / "plots")
    assert plots.binary_precision_recall_curve(directory, figures, pred="net") is None
    assert (tmp_path / "plots" / "binary_precision_recall_curve.png").is_file()
    assert pyplot.get_fignums() == []
    axis = figures.seen.axes[0]
    assert numpy.allclose(axis.lines[0].get_xdata(), expected["recall"])
    assert numpy.allclose(axis.lines[0].get_ydata(), expected["precision"])
    assert numpy.allclose(axis.lines[1].get_ydata(), [rate, rate])
    assert axis.get_title(loc="left") == f"Precision and recall (label = b, AP {expected['ap']:.3f}, n = 50)"
    assert axis.get_xlabel() == "recall"


def test_binary_curves_read_float_class_labels(tmp_path):
    from sklearn.metrics import roc_auc_score

    directory = fixture_record(tmp_path / "floats", classes=["0", "1"])
    generator = numpy.random.default_rng(6)
    truth = (generator.random(60) < 0.4).astype(float)
    prob = numpy.clip(numpy.where(truth == 1.0, 0.7, 0.3) + generator.normal(0, 0.15, 60), 0.0, 1.0)
    pandas.DataFrame({"row": numpy.arange(60), "label": truth, "pred_net": (prob >= 0.5).astype(float),
                      "raw_net": (prob >= 0.5).astype("int64"), "prob_net_0.0": 1.0 - prob,
                      "prob_net_1.0": prob}).to_parquet(directory / "predictions.parquet", index=False)
    figures = Spy(tmp_path / "plots")
    assert plots.binary_roc(directory, figures) is None
    assert (tmp_path / "plots" / "binary_roc.png").is_file()
    auc = roc_auc_score(truth, prob)
    assert figures.seen.axes[0].get_title(loc="left") == f"ROC (label = 1, AUC {auc:.3f}, n = 60)"
    assert plots.binary_precision_recall_curve(directory, figures) is None
    assert (tmp_path / "plots" / "binary_precision_recall_curve.png").is_file()
    assert pyplot.get_fignums() == []


def test_binary_curves_name_records_without_probabilities(tmp_path):
    figures = Figures(tmp_path / "plots")
    regression = fixture_record(tmp_path / "reg")
    three = fixture_record(tmp_path / "three", classes=["cold", "mild", "hot"])
    raw = fixture_record(tmp_path / "raw", classes=["a", "b"])
    binary_frame(30, 20)[["row", "label", "pred_net", "raw_net"]].to_parquet(raw / "predictions.parquet",
                                                                             index=False)
    for plot in (plots.binary_roc, plots.binary_precision_recall_curve):
        with pytest.raises(KalfaError, match="the predictions carry no probabilities"):
            plot(regression, figures)
        with pytest.raises(KalfaError, match="the predictions carry no probabilities for pred_net"):
            plot(raw, figures)
        with pytest.raises(KalfaError, match="needs two classes; label has 3"):
            plot(three, figures)
        assert plot(fixture_record(tmp_path / "bare", predictions=False), figures) is None
        assert plot(raw, figures, pred="missing") is None
    single = fixture_record(tmp_path / "single", classes=["a", "b"])
    binary_frame(12, 0).to_parquet(single / "predictions.parquet", index=False)
    assert plots.binary_roc(single, figures) is None
    assert not (tmp_path / "plots").exists()
    assert pyplot.get_fignums() == []


def test_new_plots_are_registered():
    for name in ("image_pairs", "samples_grid", "forecast_samples", "class_histogram", "binary_roc",
                 "binary_precision_recall_curve"):
        entry = ALIASES[name]
        assert entry.slot == "plot"
        assert entry.target == f"kalfa.std.plots.{name}"
