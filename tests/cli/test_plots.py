import shutil
from pathlib import Path

import pytest

from kalfa.cli import main


NAMES = ["loss_curve", "steps", "curves_rates", "pred_vs_true", "pred_histogram", "residuals", "error_map",
         "correlation_heatmap", "feature_distributions", "target_correlation", "target_vs_features", "data_pipeline",
         "architecture", "architecture_text", "class_histogram", "permutation_importance", "binary_roc",
         "binary_precision_recall_curve"]


@pytest.fixture
def copy(reference, tmp_path):
    target = tmp_path / "ref"
    shutil.copytree(reference.record, target)
    return str(target)


def test_plots_redraws_every_plot_of_the_section(copy, reference, capsys):
    drawn = sorted(path.name for path in (Path(reference.record) / "plots").iterdir())
    shutil.rmtree(Path(copy) / "plots")
    assert main(["plots", copy]) == 0
    assert capsys.readouterr().out == f"plots {', '.join(NAMES)}: {copy}/plots\n"
    assert sorted(path.name for path in (Path(copy) / "plots").iterdir()) == drawn and len(drawn) == 24


def test_plots_only_draws_the_named_ones(copy, capsys):
    shutil.rmtree(Path(copy) / "plots")
    assert main(["plots", copy, "--only", "loss_curve,residuals"]) == 0
    assert capsys.readouterr().out == f"plots loss_curve, residuals: {copy}/plots\n"
    assert sorted(path.name for path in (Path(copy) / "plots").iterdir()) == ["loss_curve.png", "residuals.png"]


def test_plots_set_changes_the_figure_format(copy, capsys):
    shutil.rmtree(Path(copy) / "plots")
    assert main(["plots", copy, "--only", "loss_curve", "--set", "figures.format=pdf"]) == 0
    assert capsys.readouterr().out == f"plots loss_curve: {copy}/plots\n"
    assert sorted(path.name for path in (Path(copy) / "plots").iterdir()) == ["loss_curve.pdf"]


def test_plots_refuses_a_name_outside_the_section(copy, capsys):
    assert main(["plots", copy, "--only", "loss_curve,nope"]) == 1
    assert capsys.readouterr().err == f"plots ['nope'] are not in the plots section, which has {sorted(NAMES)}\n"
