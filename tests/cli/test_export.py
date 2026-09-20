import shutil
from pathlib import Path

import pytest
import torch

from helpers import needs
from kalfa.cli import main
from kalfa.std.common.optional import installed


@pytest.fixture
def copy(reference, tmp_path):
    target = tmp_path / "ref"
    shutil.copytree(reference.record, target)
    return str(target)


def test_export_writes_the_state_dict_of_the_predicts_model(copy, capsys):
    assert main(["export", copy]) == 0
    assert capsys.readouterr().out == f"exported full as /export/kalfa/state_dict: {copy}/export/full.pt\n"
    assert sorted(path.name for path in (Path(copy) / "export").iterdir()) == ["full.pt"]
    assert isinstance(torch.load(Path(copy) / "export" / "full.pt"), dict)


def test_export_state_dict_of_a_composite_carries_its_models(copy, capsys):
    assert main(["export", copy]) == 0
    state = torch.load(Path(copy) / "export" / "full.pt")
    assert len(state) == 29 and all(isinstance(value, torch.Tensor) for value in state.values())
    assert {key.split(".")[1] for key in state} == {"h", "y_hat", "aux_hat", "s", "tail_logit"}
    assert all(key.startswith("refs.") for key in state)


def test_export_pt2_of_a_composite_names_the_weights_of_its_models(copy, capsys):
    assert main(["export", copy, "--format", "pt2"]) == 0
    program = torch.export.load(str(Path(copy) / "export" / "full.pt2"))
    assert len(program.state_dict) == 29 and program.constants == {}
    assert {key.split(".")[1] for key in program.state_dict} == {"h", "y_hat", "aux_hat", "s", "tail_logit"}


def test_export_reaches_the_ema_copy_by_name(copy, capsys):
    assert main(["export", copy, "--model", "tower.ema"]) == 0
    assert capsys.readouterr().out == f"exported tower.ema as /export/kalfa/state_dict: {copy}/export/tower.ema.pt\n"
    state = torch.load(Path(copy) / "export" / "tower.ema.pt")
    assert len(state) == 18 and all(key.startswith("model.nodes.") for key in state)
    assert main(["export", copy, "--model", "tower"]) == 0
    live = torch.load(Path(copy) / "export" / "tower.pt")
    assert not torch.equal(state["model.nodes.stem.weight"], live["nodes.stem.weight"])


def test_export_refuses_a_model_the_batch_cannot_feed(copy, capsys):
    assert main(["export", copy, "--model", "head_lin"]) == 1
    assert capsys.readouterr().err.startswith("model input 'h' is not a batch field; the batch has ")
    assert not (Path(copy) / "export").exists()


def test_export_writes_the_same_shape_from_every_checkpoint(copy):
    keys = {}
    for which in ("best", "last", "final"):
        assert main(["export", copy, "--model", "tower", "--which", which, "--out", f"{copy}/{which}"]) == 0
        keys[which] = sorted(torch.load(Path(copy) / which / "tower.pt"))
    assert keys["best"] == keys["last"] == keys["final"] and len(keys["best"]) == 18


def test_export_state_dict_of_a_model_carries_its_layers(copy, capsys):
    assert main(["export", copy, "--model", "tower"]) == 0
    assert capsys.readouterr().out == f"exported tower as /export/kalfa/state_dict: {copy}/export/tower.pt\n"
    state = torch.load(Path(copy) / "export" / "tower.pt")
    assert len(state) == 18 and all(isinstance(value, torch.Tensor) for value in state.values())
    assert sorted(state)[:3] == ["nodes.h_0__f.bias", "nodes.h_0__f.weight", "nodes.h_0__n.bias"]


def test_export_pt2_traces_a_named_model(copy, capsys):
    assert main(["export", copy, "--format", "pt2", "--model", "tower", "--which", "last"]) == 0
    assert capsys.readouterr().out == f"exported tower as /export/kalfa/pt2: {copy}/export/tower.pt2\n"
    assert (Path(copy) / "export" / "tower.pt2").stat().st_size > 0


def test_export_out_chooses_the_directory(copy, tmp_path, capsys):
    out = tmp_path / "exported"
    assert main(["export", copy, "--out", str(out), "--which", "final"]) == 0
    assert capsys.readouterr().out == f"exported full as /export/kalfa/state_dict: {out}/full.pt\n"
    assert sorted(path.name for path in out.iterdir()) == ["full.pt"] and not (Path(copy) / "export").exists()


def test_export_refuses_a_format_that_is_no_export_lego(copy, capsys):
    assert main(["export", copy, "--format", "nope"]) == 1
    assert capsys.readouterr().err == ("export format 'nope' is no export lego; the std ones are onnx, pt2 and "
                                       "state_dict\n")
    assert main(["export", copy, "--format", "adam"]) == 1
    assert capsys.readouterr().err == "/optimizer/torch/adam is a optimizer lego, not an export\n"
    assert not (Path(copy) / "export").exists()


def test_export_says_when_a_format_writes_nothing(copy, monkeypatch, capsys):
    monkeypatch.setattr("kalfa.std.export.kalfa.formats.load", lambda name, purpose: None)
    assert main(["export", copy, "--format", "onnx"]) == 1
    assert capsys.readouterr().err == "nothing written: /export/kalfa/onnx could not export full\n"


def test_format_param_reaches_the_export_lego(copy, capsys):
    needs("onnx")
    import onnx

    assert main(["export", copy, "--format", "onnx", "--model", "tower", "--format-param", "opset=18"]) == 0
    capsys.readouterr()
    written = onnx.load(str(Path(copy) / "export" / "tower.onnx"))
    assert [item.version for item in written.opset_import if item.domain == ""] == [18]


def test_format_param_names_what_the_lego_takes(copy, capsys):
    assert main(["export", copy, "--format", "onnx", "--format-param", "opsett=18"]) == 1
    assert capsys.readouterr().err == ("/export/kalfa/onnx has no parameter 'opsett'; it takes "
                                       "['dynamo', 'opset']\n")
    assert main(["export", copy, "--format", "state_dict", "--format-param", "opset=18"]) == 1
    assert capsys.readouterr().err == "/export/kalfa/state_dict has no parameter 'opset'; it takes none\n"
    assert not (Path(copy) / "export").exists()


def test_format_param_dynamo_says_what_the_new_exporter_needs(copy, capsys):
    needs("onnx")
    if installed("onnxscript"):
        pytest.skip("onnxscript is installed, so the new exporter runs")
    assert main(["export", copy, "--format", "onnx", "--model", "tower", "--format-param", "dynamo=true"]) == 1
    assert capsys.readouterr().err == ("the onnx export with dynamo needs onnxscript, which is not installed; "
                                       "pip install onnxscript, or leave dynamo false for the TorchScript "
                                       "exporter\n")


def test_export_onnx_writes_the_graph(copy, capsys):
    needs("onnx")
    assert main(["export", copy, "--format", "onnx", "--model", "tower"]) == 0
    assert capsys.readouterr().out == f"exported tower as /export/kalfa/onnx: {copy}/export/tower.onnx\n"
    assert (Path(copy) / "export" / "tower.onnx").stat().st_size > 0
