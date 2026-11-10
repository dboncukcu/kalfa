import json
import sys
from pathlib import Path

import pytest

from helpers import housing_frame, write_parquet
from kalfa import cli
from kalfa.core import aliases
from kalfa.core import run as running
from kalfa.core.errors import KalfaError


PLUGIN = "layered_legos"

LEGOS = """import torch


def log_cosh(pred, target):
    return torch.log(torch.cosh(pred - target)).mean()
"""

BASE = {
    "params": {"epochs": 2},
    "seed": 3,
    "device": "cpu",
    "data": {"use": "table", "file": "housing.parquet", "split": [0.7, 0.15, 0.15],
             "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
    "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
    "losses": {"log_cosh": f"{PLUGIN}.log_cosh"},
    "metrics": {"rmse": "rmse"},
    "optimizers": {"main": {"use": "adam", "lr": 1.0e-3, "over": "@models.net", "minimizes": "@losses.log_cosh"}},
    "train": {"epochs": "$epochs$"},
}


@pytest.fixture
def layers(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, PLUGIN, raising=False)
    base = tmp_path / "base"
    write_parquet(housing_frame(rows=120), base / "housing.parquet")
    (base / f"{PLUGIN}.py").write_text(LEGOS, encoding="utf-8")
    (base / "config.yaml").write_text(json.dumps(BASE), encoding="utf-8")
    top = tmp_path / "top"
    top.mkdir()
    (top / "config.yaml").write_text("include: [../base/config.yaml]\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return tmp_path, top / "config.yaml", base / "config.yaml", elsewhere


def fresh_process(monkeypatch, root, elsewhere):
    monkeypatch.setattr(aliases, "SEARCH_DIRS", [])
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if not entry.startswith(str(root))])
    monkeypatch.delitem(sys.modules, PLUGIN, raising=False)
    monkeypatch.chdir(elsewhere)


def manifest_of(record):
    return json.loads((Path(record) / "manifest.json").read_text(encoding="utf-8"))


def test_a_record_of_an_include_lists_every_layer_and_predicts_from_a_fresh_process(layers, monkeypatch):
    root, top, base, elsewhere = layers
    first = root / "runs" / "first"
    running.run(paths=[str(top)], record=str(first))
    assert manifest_of(first)["config_files"] == [str(top.resolve()), str(base.resolve())]
    fresh_process(monkeypatch, root, elsewhere)
    path, table = running.predict_record(first)
    assert path == first / "predictions_last.parquet" and len(table) == 18


def test_a_resumed_record_keeps_the_layers_and_the_plugins_of_its_source(layers, monkeypatch):
    root, top, base, elsewhere = layers
    first = root / "runs" / "first"
    running.run(paths=[str(top)], record=str(first))
    (base.parent / f"{PLUGIN}.py").rename(root / "moved.py")
    fresh_process(monkeypatch, root, elsewhere)
    second = root / "runs" / "second"
    result = running.resume(first, sets=[("train.epochs", 3)], target=second)
    assert result.context.turn == 3 and manifest_of(second)["status"] == "done"
    files = manifest_of(second)["config_files"]
    assert files == [str((first / "resolved.yaml").resolve()), str(top.resolve()), str(base.resolve())]
    assert (second / "plugins" / f"{PLUGIN}.py").is_file() and not (second / "plugins" / "plugins").exists()
    fresh_process(monkeypatch, root, elsewhere)
    path, table = running.predict_record(second)
    assert path == second / "predictions_last.parquet" and len(table) == 18


def test_predict_names_why_the_sets_could_not_be_rebuilt(layers, monkeypatch):
    root, top, base, elsewhere = layers
    first = root / "runs" / "first"
    running.run(paths=[str(top)], record=str(first))
    (base.parent / "housing.parquet").rename(root / "moved.parquet")
    fresh_process(monkeypatch, root, elsewhere)
    with pytest.raises(KalfaError, match="could not be rebuilt: data file 'housing.parquet' not found"):
        running.predict_record(first)


def test_check_and_run_of_a_record_find_its_plugins_and_data_from_a_fresh_process(layers, monkeypatch, capsys):
    root, top, base, elsewhere = layers
    first = root / "runs" / "first"
    running.run(paths=[str(top)], record=str(first))
    fresh_process(monkeypatch, root, elsewhere)
    assert cli.main(["check", str(first)]) == 0
    assert capsys.readouterr().out.strip() == "ok"
    fresh_process(monkeypatch, root, elsewhere)
    again = root / "runs" / "again"
    running.run(paths=cli.record_configs([str(first)]), sets=[("train.epochs", 1)], record=str(again))
    files = manifest_of(again)["config_files"]
    assert files == [str((first / "resolved.yaml").resolve()), str(top.resolve()), str(base.resolve())]
    assert (again / "plugins" / f"{PLUGIN}.py").is_file() and not (again / "plugins" / "plugins").exists()
    fresh_process(monkeypatch, root, elsewhere)
    path, table = running.predict_record(again)
    assert path == again / "predictions_last.parquet" and len(table) == 18
