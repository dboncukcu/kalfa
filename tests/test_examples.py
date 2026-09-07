"""examples/: every folder is a faithful copy of its config, its includes and its plugins, and two of them run from
their own make_data.py the way the README says."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import generate, predict, run
from kalfa.config import parse_sets
from kalfa.record import read_history

sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from sync_examples import CONFIGS, EXAMPLES, STANDALONE, TABLE, copies  # noqa: E402


def test_every_example_is_a_copy_of_its_config():
    folders = sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir() and not path.name.startswith("."))
    assert folders == sorted([*TABLE, *STANDALONE])
    configs = sorted(path.name for path in CONFIGS.glob("*.yaml") if path.name != "reference.yaml")
    assert configs == sorted(entry["config"] for entry in TABLE.values())
    for name in TABLE:
        for source, target in copies(name):
            assert target.exists(), target
            assert target.read_text() == source.read_text(), \
                f"{target} drifted from {source}; run tests/fixtures/sync_examples.py"
        assert (EXAMPLES / name / "make_data.py").exists()
    for name in STANDALONE:
        folder = EXAMPLES / name
        assert (folder / "config.yaml").exists() and (folder / "make_data.py").exists()
        assert not (CONFIGS / f"{name}.yaml").exists(), f"{name} owns its config, it is not a copy"
    readme = (EXAMPLES / "README.md").read_text()
    assert all(f"## {name}" in readme for name in [*TABLE, *STANDALONE])


def load_make_data(folder):
    spec = importlib.util.spec_from_file_location(f"make_data_{folder.name}", folder / "make_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example(tmp_path, monkeypatch):
    def prepare(name):
        target = tmp_path / name
        shutil.copytree(EXAMPLES / name, target, ignore=shutil.ignore_patterns("runs", "data", "*.parquet", "__pycache__"))
        monkeypatch.chdir(target)
        load_make_data(target).main()
        return target
    return prepare


def test_example_01_runs_from_its_folder(example):
    example("01_mlp_regression")
    result = run(["config.yaml"], parse_sets(["record=runs/01"], ["epochs=3"]))
    assert result.record == "runs/01"
    history = read_history(Path("runs/01"))
    assert [line["turn"] for line in history] == [1, 2, 3]
    fresh = predict("runs/01", data="new.parquet")
    assert len(fresh.table) == 100 and {"row", "price", "pred_y"} <= set(fresh.table.columns)


def test_example_10_runs_from_its_folder(example):
    example("10_char_lm")
    sets = ["device=cpu", "record=runs/10", "data.batch=8",
            "model.models.gpt.nodes=[{uri: embedding, params: {num: {uri: vocab_size}, dim: 16}}, "
            "{uri: gpt, params: {d_model: 16, layers: 1, heads: 2, seq_len: 16}}, "
            "{uri: linear, params: {out_features: {uri: vocab_size}}}]",
            "optimizers.main.schedule={uri: warmup_cosine, params: {warmup: 1, total: 8}}",
            "generate.params.max_new_tokens=10"]
    result = run(["config.yaml"], parse_sets(sets, ["total_steps=8", "turn_steps=4", "seq_len=16"]))
    history = read_history(Path(result.record))
    assert [line["turn"] for line in history] == [1, 2] and history[-1]["global_step"] == 8
    text = generate("runs/10").samples
    assert text.startswith("ROMEO:") and len(text) == len("ROMEO:") + 10
    assert (Path("runs/10") / "plugins" / "gpt_legos.py").read_text() == Path("gpt_legos.py").read_text()


def test_a_plugin_in_a_plugins_folder_next_to_the_config_is_found(tmp_path, monkeypatch):
    from helpers import minimal, write_config
    from kalfa.api import check
    from kalfa.synthetic import write_housing

    monkeypatch.chdir(tmp_path)
    write_housing(tmp_path / "housing.parquet")
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "folder_only_plugin.py").write_text(
        "import kalfa\n\n\n@kalfa.lego('/layer/folder/marker', alias='folder_marker')\n"
        "def marker():\n    import torch\n    return torch.nn.Identity()\n")
    config = minimal()
    config["plugins"] = ["folder_only_plugin"]
    config["model"]["nodes"].insert(0, {"uri": "folder_marker"})
    prepared = check([str(write_config(tmp_path / "cfg.yaml", config))], parse_sets([]))
    assert prepared.problems == []
