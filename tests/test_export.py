import re
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch

import kalfa
from helpers import housing_frame
from kalfa import cli
from kalfa.commands import export as exporting
from kalfa.core import run as running
from kalfa.core.errors import KalfaError


pytestmark = pytest.mark.slow


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
        "train": {"epochs": 1, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
    }


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    housing_frame(160, 0).to_parquet(root / "housing.parquet", index=False)
    return root


@pytest.fixture(scope="module")
def trained(home):
    result = kalfa.run(housing_config(home), record=str(home / "runs" / "net"))
    return Path(result.record)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, module, args, f, **options):
        self.calls.append({"module": module, "args": args, "f": f, "training": module.training, **options})
        Path(f).write_bytes(b"onnx")


@pytest.fixture
def fake_onnx(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", types.ModuleType("onnx"))
    recorder = Recorder()
    monkeypatch.setattr(torch.onnx, "export", recorder)
    return recorder


def test_state_dict_goes_to_the_default_path_with_the_report_weights(trained):
    found = exporting.export_record(trained)
    assert isinstance(found, exporting.Exported)
    assert (found.record, found.format, found.model, found.which) == (str(trained), "state_dict", "net", "best")
    assert Path(found.path) == trained / "exports" / "net_best.pt"
    state = torch.load(found.path, weights_only=True)
    reference = running.open_record(trained, which="best", data=False).models["net"].module.state_dict()
    assert list(state) == list(reference)
    assert all(torch.equal(state[key], reference[key]) for key in reference)
    assert all(value.device.type == "cpu" for value in state.values())


def test_state_dict_takes_which_and_an_explicit_out(trained, tmp_path):
    out = tmp_path / "deep" / "weights.pt"
    found = kalfa.api.export(trained, which="last", out=str(out))
    assert Path(found.path) == out and found.which == "last" and found.model == "net"
    state = torch.load(out, weights_only=True)
    reference = running.open_record(trained, which="last", data=False).models["net"].module.state_dict()
    assert all(torch.equal(state[key], reference[key]) for key in reference)
    named = kalfa.open(trained).export(model="net", which="last")
    assert Path(named.path) == trained / "exports" / "net_last.pt" and named.which == "last"


def test_which_falls_back_to_last_without_a_best_checkpoint(trained, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(trained, copy)
    (copy / "checkpoints" / "best.pt").unlink()
    found = exporting.export_record(copy)
    assert found.which == "last" and Path(found.path) == copy / "exports" / "net_last.pt"
    assert Path(found.path).is_file()


def test_an_unknown_format_is_refused_before_the_record_opens(tmp_path):
    with pytest.raises(KalfaError, match=r"state_dict.*onnx.*torchscript"):
        exporting.export_record(tmp_path / "nothing", format="torchscript")


def test_an_unknown_model_lists_the_models(trained):
    with pytest.raises(KalfaError, match=r"'ghost'.*\['net'\]"):
        exporting.export_record(trained, model="ghost")
    assert not (trained / "exports" / "ghost_best.pt").exists()


def test_no_model_and_no_predicts_lists_the_models(trained, monkeypatch):
    real = running.open_record

    def opened(*args, **kwargs):
        ctx = real(*args, **kwargs)
        ctx.predicts = lambda: None
        return ctx

    monkeypatch.setattr(running, "open_record", opened)
    with pytest.raises(KalfaError, match=r"\['net'\].*train\.predicts"):
        exporting.export_record(trained)


def test_onnx_without_the_onnx_package_names_the_extra(trained, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    message = "onnx export needs the onnx package (uv sync installs the dependencies of kalfa)"
    with pytest.raises(KalfaError, match=re.escape(message)):
        exporting.export_record(trained, format="onnx", out=str(tmp_path / "net.onnx"))
    assert not (tmp_path / "net.onnx").exists()


def test_onnx_traces_the_first_test_batch_with_names_and_a_batch_axis(trained, fake_onnx):
    found = exporting.export_record(trained, format="onnx")
    assert Path(found.path) == trained / "exports" / "net_best.onnx" and found.format == "onnx"
    assert found.which == "best" and found.model == "net"
    [call] = fake_onnx.calls
    reference = running.open_record(trained, data=True)
    batch = next(iter(reference.loaders()["test"]))
    assert isinstance(call["args"], tuple) and len(call["args"]) == 1
    assert torch.equal(call["args"][0], batch["x"])
    assert call["f"] == found.path and call["training"] is False
    assert call["input_names"] == ["x"] and call["output_names"] == ["net"]
    assert call["dynamic_axes"] == {"x": {0: "batch"}, "net": {0: "batch"}}
    assert call["opset_version"] == 17 and call["dynamo"] is False


def test_onnx_static_axes_opset_and_the_valid_fallback(trained, tmp_path, fake_onnx, monkeypatch):
    real = running.open_record

    def opened(*args, **kwargs):
        ctx = real(*args, **kwargs)
        loaders = ctx.loaders()
        ctx.loaders = lambda: {name: loader for name, loader in loaders.items() if name != "test"}
        return ctx

    monkeypatch.setattr(running, "open_record", opened)
    out = tmp_path / "models" / "net.onnx"
    found = exporting.export_record(trained, format="onnx", which="last", out=str(out), opset=13, dynamic=False)
    assert Path(found.path) == out and out.is_file() and found.which == "last"
    [call] = fake_onnx.calls
    batch = next(iter(real(trained, data=True).loaders()["valid"]))
    assert torch.equal(call["args"][0], batch["x"])
    assert call["dynamic_axes"] is None and call["opset_version"] == 13


def test_onnx_without_any_batch_says_so(trained, fake_onnx, monkeypatch):
    real = running.open_record

    def opened(*args, **kwargs):
        ctx = real(*args, **kwargs)
        ctx.loaders = lambda: {}
        return ctx

    monkeypatch.setattr(running, "open_record", opened)
    with pytest.raises(KalfaError, match="batch"):
        exporting.export_record(trained, format="onnx")
    assert not fake_onnx.calls


def test_onnx_round_trip_checks(trained, tmp_path):
    import onnx
    found = exporting.export_record(trained, format="onnx", out=str(tmp_path / "net.onnx"))
    model = onnx.load(found.path)
    onnx.checker.check_model(model)
    assert [value.name for value in model.graph.input] == ["x"]
    assert [value.name for value in model.graph.output] == ["net"]
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    static = exporting.export_record(trained, format="onnx", out=str(tmp_path / "static.onnx"), dynamic=False)
    model = onnx.load(static.path)
    onnx.checker.check_model(model)
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_value > 0


def test_the_cli_prints_the_exported_path(trained, tmp_path, capsys):
    out = tmp_path / "cli.pt"
    assert cli.main(["export", str(trained), "--which", "last", "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip() == str(out)
    assert out.is_file()


def test_an_out_directory_takes_the_default_file_name(trained, tmp_path, capsys):
    folder = tmp_path / "outdir"
    folder.mkdir()
    assert cli.main(["export", str(trained), "--which", "last", "--out", str(folder)]) == 0
    assert capsys.readouterr().out.strip() == str(folder / "net_last.pt")
    assert (folder / "net_last.pt").is_file()
    fresh = tmp_path / "fresh"
    found = exporting.export_record(trained, out=f"{fresh}/")
    assert Path(found.path) == fresh / "net_best.pt" and Path(found.path).is_file()
