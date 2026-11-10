import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import torch

import kalfa
from helpers import housing_frame
from kalfa import cli
from kalfa.commands.generate import Generated, generate_record
from kalfa.commands.stop import stop_many, stop_record
from kalfa.core.errors import KalfaError
from records.fixtures import make_record, make_sweep


def housing_config(home, generate=True, report="best"):
    config = {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1, "checkpoint": {"watch": "val/rmse"}, "report": report},
    }
    if generate:
        config["generate"] = {"use": "gan_sampler", "model": "@models.net", "latent": 4, "n": 6}
    return config


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("generate")
    housing_frame(160, 0).to_parquet(root / "housing.parquet", index=False)
    return root


@pytest.fixture(scope="module")
def trained(home):
    return Path(kalfa.run(housing_config(home), record=str(home / "runs" / "gan")).record)


@pytest.mark.slow
def test_generate_draws_the_sampler_n_from_the_best_weights(trained):
    found = generate_record(trained)
    assert isinstance(found, Generated)
    assert found.record == str(trained) and found.which == "best"
    assert found.paths == [str(trained / "samples" / "generate_best.pt")]
    assert all(isinstance(path, str) for path in found.paths)
    assert tuple(found.samples.shape) == (6, 1) and not found.samples.requires_grad
    saved = torch.load(trained / "samples" / "generate_best.pt", weights_only=False)
    assert torch.equal(saved, found.samples)


@pytest.mark.slow
def test_generate_takes_n_which_and_out(home, trained):
    target = home / "elsewhere"
    found = generate_record(str(trained), n=3, which="last", out=str(target))
    assert found.which == "last"
    assert found.paths == [str(target / "generate_last.pt")]
    assert tuple(found.samples.shape) == (3, 1)
    assert not (trained / "samples" / "generate_last.pt").exists()
    final = generate_record(trained, n=2, which="final")
    assert final.which == "final" and Path(final.paths[0]).name == "generate_final.pt"


@pytest.mark.slow
def test_generate_falls_back_to_last_without_best(tmp_path, trained):
    copy = shutil.copytree(trained, tmp_path / "copy")
    (copy / "checkpoints" / "best.pt").unlink()
    found = generate_record(copy, n=2)
    assert found.which == "last" and Path(found.paths[0]) == copy / "samples" / "generate_last.pt"


@pytest.mark.slow
def test_generate_rejects_an_unknown_which(trained):
    with pytest.raises(KalfaError, match="which"):
        generate_record(trained, which="middle")


@pytest.mark.slow
def test_generate_needs_a_generate_section(home):
    record = kalfa.run(housing_config(home, generate=False), record=str(home / "runs" / "plain")).record
    with pytest.raises(KalfaError, match="has no generate section"):
        generate_record(record)
    assert not (Path(record) / "samples").exists()


def fresh():
    return datetime.now().isoformat(timespec="seconds")


def test_stop_writes_stop_json_into_a_record(tmp_path):
    record = make_record(tmp_path / "run", status="running", updated=fresh())
    written = stop_record(record, by="test")
    assert written == [str(record / "stop.json")]
    note = json.loads((record / "stop.json").read_text(encoding="utf-8"))
    assert note["by"] == "test"
    moment = datetime.fromisoformat(note["at"])
    assert moment.microsecond == 0 and note["at"] == moment.isoformat(timespec="seconds")


def test_stop_writes_only_into_running_or_stale_sweep_points(tmp_path):
    root, points = make_sweep(tmp_path / "sweep", count=4)
    make_record(points[1], status="running", updated=fresh())
    make_record(points[2], status="running")
    written = stop_record(root, by="test")
    assert written == [str(points[1] / "stop.json"), str(points[2] / "stop.json")]
    assert not (points[0] / "stop.json").exists() and not (points[3] / "stop.json").exists()
    assert not (root / "stop.json").exists()
    assert json.loads((points[2] / "stop.json").read_text(encoding="utf-8"))["by"] == "test"


def test_stop_on_a_finished_sweep_writes_nothing(tmp_path):
    root, _ = make_sweep(tmp_path / "sweep", count=2)
    assert stop_record(root) == []


def test_stop_rejects_a_plain_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(KalfaError, match=re.escape(str(plain))):
        stop_record(plain)
    with pytest.raises(KalfaError, match="missing"):
        stop_record(tmp_path / "missing")


def test_stop_many_joins_records_and_checks_every_path_first(tmp_path):
    record = make_record(tmp_path / "run", status="running")
    root, points = make_sweep(tmp_path / "sweep", count=3)
    make_record(points[0], status="running")
    written = stop_many([str(record), str(root)], by="many")
    assert written == [str(record / "stop.json"), str(points[0] / "stop.json")]
    other = make_record(tmp_path / "other", status="running")
    with pytest.raises(KalfaError, match="nowhere"):
        stop_many([str(other), str(tmp_path / "nowhere")])
    assert not (other / "stop.json").exists()


@pytest.mark.slow
def test_generate_through_the_api_and_the_cli(home, trained, capsys):
    found = kalfa.api.generate(trained, n=5)
    assert found.which == "best" and tuple(found.samples.shape) == (5, 1)
    again = kalfa.open(trained).generate(n=3, which="last", out=str(home / "api"))
    assert tuple(again.samples.shape) == (3, 1) and Path(again.paths[0]).parent == home / "api"
    capsys.readouterr()
    assert cli.main(["generate", str(trained), "--n", "2", "--out", str(home / "cli")]) == 0
    assert capsys.readouterr().out.split() == [str(home / "cli" / "generate_best.pt")]
    assert tuple(torch.load(home / "cli" / "generate_best.pt", weights_only=False).shape) == (2, 1)


def test_stop_through_the_api_and_the_cli(tmp_path, capsys):
    record = make_record(tmp_path / "run", status="running")
    assert [Path(path).name for path in kalfa.api.stop(record, by="test")] == ["stop.json"]
    assert kalfa.open(record).stop() == [str(record / "stop.json")]
    assert json.loads((record / "stop.json").read_text(encoding="utf-8"))["by"] == "api"
    root, points = make_sweep(tmp_path / "sweep", count=3)
    make_record(points[1], status="running")
    assert [Path(path).parent.name for path in kalfa.api.stop(root)] == ["0001"]
    capsys.readouterr()
    assert cli.main(["stop", str(record), str(root)]) == 0
    assert capsys.readouterr().out.split() == [str(record / "stop.json"), str(points[1] / "stop.json")]
    assert json.loads((record / "stop.json").read_text(encoding="utf-8"))["by"] == "cli"


def test_stop_skips_a_finished_record_and_an_idle_root_with_a_warning(tmp_path, caplog):
    record = make_record(tmp_path / "finished", status="done")
    root, points = make_sweep(tmp_path / "sweep", count=2)
    with caplog.at_level(logging.WARNING, logger="kalfa.commands"):
        assert stop_record(record) == []
        assert stop_many([str(record), str(root)]) == []
    assert not (record / "stop.json").exists() and not any((point / "stop.json").exists() for point in points)
    assert caplog.messages.count(f"{record} is done; nothing to stop") == 2
    assert f"nothing to stop under {root}: no running or stale points" in caplog.messages
