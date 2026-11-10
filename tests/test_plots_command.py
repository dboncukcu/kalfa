import os
from pathlib import Path

import pytest

import kalfa
from helpers import housing_frame
from kalfa import cli
from kalfa.commands.plots import redraw
from kalfa.core.errors import KalfaError


pytestmark = pytest.mark.slow

PLOTS = ["loss_curve", "pred_vs_true"]


def housing_config(home):
    return {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "plots": {name: name for name in PLOTS},
        "train": {"epochs": 1, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    home = tmp_path_factory.mktemp("plots_command")
    housing_frame(160, 0).to_parquet(home / "housing.parquet", index=False)
    return Path(kalfa.run(housing_config(home), record=str(home / "runs" / "net")).record)


def drawn(record, names):
    return [str(record / "plots" / f"{name}.png") for name in names]


def test_redraw_draws_every_plot_and_returns_the_files(trained):
    for path in drawn(trained, PLOTS):
        os.unlink(path)
    found = redraw(trained)
    assert found == drawn(trained, PLOTS)
    assert all(Path(path).stat().st_size > 0 for path in found)


def test_redraw_returns_a_file_drawn_again_over_an_old_one(trained):
    redraw(trained)
    assert redraw(trained, which="last") == drawn(trained, PLOTS)


def test_only_takes_comma_separated_names(trained):
    assert redraw(trained, only="loss_curve") == drawn(trained, ["loss_curve"])
    assert redraw(trained, only="pred_vs_true,loss_curve", which="final", device="cpu") == drawn(trained, PLOTS)


def test_an_unknown_name_lists_the_plots(trained):
    message = r"plots \['nope'\] are not in the plots section, which has \['loss_curve', 'pred_vs_true'\]"
    with pytest.raises(KalfaError, match=message):
        redraw(trained, only="loss_curve,nope")


def test_a_directory_without_a_record_says_so(tmp_path):
    with pytest.raises(KalfaError, match="no resolved.yaml"):
        redraw(tmp_path)


def test_the_cli_prints_each_drawn_file(trained, capsys):
    assert cli.main(["plots", str(trained), "--only", "loss_curve", "--which", "best", "--device", "cpu"]) == 0
    assert capsys.readouterr().out.splitlines() == drawn(trained, ["loss_curve"])
    assert cli.main(["plots", str(trained)]) == 0
    assert capsys.readouterr().out.splitlines() == drawn(trained, PLOTS)


def test_the_cli_names_the_unknown_plot_and_the_known_ones(trained, capsys):
    assert cli.main(["plots", str(trained), "--only", "nope"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: plots ['nope'] are not in the plots section") and "'loss_curve'" in err


def test_the_cli_says_when_nothing_was_drawn(trained, capsys, monkeypatch):
    monkeypatch.setattr("kalfa.commands.plots.redraw", lambda record, only=None, which=None, device=None: [])
    assert cli.main(["plots", str(trained)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and "none of the plots had anything to draw" in captured.err
