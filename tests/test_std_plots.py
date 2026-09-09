"""Plots write their files under plots/."""

import pandas
import pytest

import kalfa  # noqa: F401
from kalfa.std import figure
from kalfa.std.plot import loss_curve, panel_title, pred_vs_true, run_all, series_of


def history():
    return [{"turn": 1, "global_step": 3, "train/l": 1.0, "val/l": 1.1, "lr/m": 0.1, "rules": []},
            {"turn": 2, "global_step": 6, "train/l": 0.5, "val/l": 0.6, "lr/m": 0.1, "rules": ["a"]}]


def test_series_of_skips_bookkeeping_and_selects():
    assert series_of(history()) == {"train/l": [1.0, 0.5], "val/l": [1.1, 0.6]}
    assert series_of(history(), ["val/l"]) == {"val/l": [1.1, 0.6]}


def test_plots_write_files(tmp_path):
    predictions = pandas.DataFrame({"row": [0, 1, 2], "price": [1.0, 2.0, 3.0], "raw_y": [0.1, 0.2, 0.3],
                                    "pred_y": [1.1, 1.9, 3.2]})
    loss_curve(predictions, history(), {}, str(tmp_path))
    pred_vs_true(predictions, history(), {}, str(tmp_path))
    assert (tmp_path / "plots" / "loss_curve.png").exists() and (tmp_path / "plots" / "pred_vs_true.png").exists()
    assert pred_vs_true(pandas.DataFrame(), history(), {}, str(tmp_path)) is None
    assert loss_curve(predictions, [], {}, str(tmp_path)) is None


def test_panel_title_names_the_wire_only_when_two_outputs_share_a_field():
    paired = ["combined_z", "combined_z", "z_a"]
    assert panel_title("pred_combined_base_combined_z", "combined_z", paired) == "combined_z (combined_base)"
    assert panel_title("pred_combined_hat_combined_z", "combined_z", paired) == "combined_z (combined_hat)"
    assert panel_title("pred_z_hat_z_a", "z_a", paired) == "z_a"
    assert panel_title("pred_combined_z", "combined_z", paired) == "combined_z"


def test_run_all_calls_every_plot_and_resolves_inputs_by_refs(tmp_path):
    import cirak

    seen = []

    def plot(predictions, history, models, record):
        seen.append((len(predictions), record))

    run_all(pandas.DataFrame({"a": [1]}), history(), {}, {"one": plot, "two": plot}, record=str(tmp_path))
    assert seen == [(1, str(tmp_path))] * 2
    got = {}

    def typed(predictions, history, models, record, series=None, column=None, net=None):
        got.update({"series": series, "column": list(column), "net": net})

    kalfa.lego("/plot/test/typed", typed, partial=True,
               refs={"series": "history", "column": "field", "net": "model"})
    run_all(pandas.DataFrame({"price": [1.0, 2.0]}), history(), {"m": "model"}, {"t": typed},
            keys={"t": {"inputs": {"series": "val/l", "column": "price", "net": "m"}}}, record=str(tmp_path))
    assert got == {"series": [1.1, 0.6], "column": [1.0, 2.0], "net": "model"}


def test_sample_writer_and_the_sample_plots(tmp_path):
    import functools
    import warnings

    import torch
    from torch import nn

    from kalfa.std.metric import sample_writer
    from kalfa.std.plot import samples_gif, samples_matrix

    def sampler(models, prep, rng, n=4):
        return torch.rand(n, 1, 8, 8, generator=rng)

    writer = sample_writer(n=3, sampler=sampler)
    rng = torch.Generator().manual_seed(1)
    writer.update(models={}, predicts=None, rng=rng, record=str(tmp_path), turn=5)
    writer.update(models={}, predicts=None, rng=rng, record=str(tmp_path), turn=5)
    assert writer.compute() is None
    assert torch.load(tmp_path / "samples" / "turn_0005.pt", weights_only=False).shape == (3, 1, 8, 8)
    assert (tmp_path / "samples" / "turn_0005.png").exists()
    writer.reset()
    writer.update(models={}, predicts=None, rng=rng, record=str(tmp_path), turn=10)
    assert (tmp_path / "samples" / "turn_0010.png").exists()

    class Same(nn.Module):
        inputs = ["image"]
        outputs = ["out"]

        def forward(self, value):
            return value

    from_model = sample_writer(n=2)
    from_model.update(models={"net": Same()}, predicts="net", rng=rng, record=str(tmp_path), turn=15,
                      batch={"image": torch.rand(5, 1, 8, 8)})
    assert torch.load(tmp_path / "samples" / "turn_0015.pt", weights_only=False).shape == (2, 1, 8, 8)
    with pytest.raises(ValueError, match="sampler"):
        sample_writer(n=2).update(models={}, predicts=None, rng=rng, record=str(tmp_path), turn=1)
    plots = {"gif": functools.partial(samples_gif, duration=100), "matrix": functools.partial(samples_matrix, n=2)}
    run_all(None, [], {}, plots, keys={}, predicts=None, bus={"composites": {}}, record=str(tmp_path))
    assert (tmp_path / "plots" / "gif.gif").exists() and (tmp_path / "plots" / "matrix.png").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_all(None, [], {}, plots, keys={}, predicts=None, bus={"composites": {}}, record=str(empty))
    assert len(caught) == 2 and not (empty / "plots" / "gif.gif").exists()


def test_figure_settings_choose_the_format_and_the_panel_size(tmp_path):
    figure.configure({"format": "pdf", "width": 3.0, "height": 2.0})
    try:
        drawing, axis = figure.single()
        assert tuple(drawing.get_size_inches()) == (3.0, 2.0)
        path = figure.save(drawing, str(tmp_path), "sized")
        assert path.name == "sized.pdf" and path.exists()
        drawing, axes = figure.grid(2, 3)
        assert tuple(drawing.get_size_inches()) == (9.0, 4.0)
        figure.pyplot().close(drawing)
    finally:
        figure.configure(None)
    assert figure.settings()["format"] == "png"


def test_figure_profile_and_binned_follow_the_data():
    import numpy

    x = numpy.linspace(0.0, 1.0, 400)
    centers, values = figure.profile(x, 2.0 * x, bins=4)
    assert len(centers) == 4 and values[0] < values[-1]
    edges_x, edges_y, mean = figure.binned(x, x, x, bins=4, min_count=1)
    assert mean.shape == (4, 4) and numpy.isnan(mean).any() and numpy.nanmin(mean) >= 0.0


def test_a_plot_definition_overrides_the_figure_size(tmp_path):
    seen = {}

    def plot(predictions, history, models, record, name=None):
        seen["size"] = (figure.width_of(1.0), figure.height_of(1.0))

    run_all(None, [], {}, {"one": plot}, keys={"one": {"width": 12.0, "height": 3.0}},
            figures={"width": 5.0}, record=str(tmp_path))
    assert seen["size"] == (12.0, 3.0)
    assert figure.settings()["width"] == 5.0
    figure.configure(None)


def test_the_data_and_diagnostic_plots_draw_from_a_run(workdir):
    from pathlib import Path

    from helpers import minimal, write_config
    from kalfa.api import run
    from kalfa.config import parse_sets

    config = minimal()
    config["figures"] = {"format": "pdf", "width": 4.0, "height": 3.0}
    config["plots"] = {"tvf": {"uri": "target_vs_features", "params": {"per_row": 3, "log": ["x0"]}},
                       "corr": {"uri": "correlation_heatmap", "sets": ["train"], "width": 7.0},
                       "resid": {"uri": "residuals"},
                       "map": {"uri": "error_map", "params": {"x": "x0", "y": "x1", "bins": 8, "min_count": 2}},
                       "importance": {"uri": "permutation_importance", "params": {"repeats": 2}}}
    path = write_config(workdir / "cfg.yaml", config)
    record = Path(run([path], parse_sets([])).record) / "plots"
    for drawn in ("tvf", "corr", "resid", "map", "importance"):
        assert (record / f"{drawn}.pdf").exists(), drawn
    assert not list(record.glob("*.png"))
