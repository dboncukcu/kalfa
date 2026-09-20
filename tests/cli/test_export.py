import shutil
from pathlib import Path

import pytest
import torch

from helpers import needs
from kalfa.cli import main


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
    assert state and all(isinstance(value, torch.Tensor) for value in state.values())


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


def test_export_onnx_writes_the_graph(copy, capsys):
    needs("onnx")
    assert main(["export", copy, "--format", "onnx", "--model", "tower"]) == 0
    assert capsys.readouterr().out == f"exported tower as /export/kalfa/onnx: {copy}/export/tower.onnx\n"
    assert (Path(copy) / "export" / "tower.onnx").stat().st_size > 0
