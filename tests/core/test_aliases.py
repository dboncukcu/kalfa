import json
import logging
import sys

import pytest
import torch

from kalfa.core.aliases import (
    ALIASES,
    SEARCH_DIRS,
    Entry,
    add_search_dir,
    catalog,
    entry_of,
    import_plugins,
    import_target,
    lego,
    register,
    resolve,
)
from kalfa.core.errors import BuildError


TEST_MODULES = ("kalfa_t_broken", "kalfa_t_plugin", "kalfa_t_file_plugin")


@pytest.fixture(autouse=True)
def restore():
    saved = dict(ALIASES)
    dirs = list(SEARCH_DIRS)
    path = list(sys.path)
    yield
    ALIASES.clear()
    ALIASES.update(saved)
    SEARCH_DIRS[:] = dirs
    sys.path[:] = path
    for name in TEST_MODULES:
        sys.modules.pop(name, None)


def test_register_stores_an_entry_and_repeats_silently():
    entry = register("t_huber", "torch.nn.HuberLoss", slot="loss", doc="huber loss")
    assert isinstance(entry, Entry)
    assert entry_of("t_huber") is entry
    assert entry.target == "torch.nn.HuberLoss"
    assert entry.slot == "loss"
    assert entry.plugin is False
    again = register("t_huber", "torch.nn.HuberLoss", slot="loss")
    assert again is entry
    assert entry_of("t_huber").doc == "huber loss"


def test_register_a_different_target_raises_unless_both_are_plugins(caplog):
    register("t_loss", "torch.nn.HuberLoss", slot="loss")
    with pytest.raises(BuildError, match="already registered"):
        register("t_loss", "torch.nn.L1Loss", slot="loss")
    assert entry_of("t_loss").target == "torch.nn.HuberLoss"
    register("t_plug", torch.nn.HuberLoss, slot="loss", plugin=True)
    with pytest.raises(BuildError):
        register("t_plug", torch.nn.L1Loss, slot="loss")
    with caplog.at_level(logging.WARNING, logger="kalfa.aliases"):
        register("t_plug", torch.nn.L1Loss, slot="loss", plugin=True)
    assert entry_of("t_plug").target is torch.nn.L1Loss
    assert any("t_plug" in record.getMessage() for record in caplog.records)


def test_register_rejects_bad_names_and_targets():
    with pytest.raises(BuildError):
        register("", "torch.nn.L1Loss")
    with pytest.raises(BuildError):
        register("t_none", None)


def test_resolve_returns_targets_import_paths_or_raises():
    register("t_mae", "torch.nn.L1Loss", slot="loss")
    register("t_obj", torch.nn.MSELoss, slot="loss")
    assert resolve("t_mae") == "torch.nn.L1Loss"
    assert resolve("t_obj") is torch.nn.MSELoss
    assert resolve("mymodels.Vae") == "mymodels.Vae"
    with pytest.raises(BuildError, match="kalfa legos lists the names"):
        resolve("no_such_name")
    with pytest.raises(BuildError):
        resolve(".relative.Thing")
    with pytest.raises(BuildError):
        resolve(5)


def test_import_target_walks_modules_and_attributes():
    assert import_target("torch.nn.HuberLoss") is torch.nn.HuberLoss
    assert import_target("torch.nn.functional.mse_loss") is torch.nn.functional.mse_loss
    assert import_target("json.dumps") is json.dumps
    assert import_target(torch.nn.L1Loss) is torch.nn.L1Loss
    assert import_target("torch") is torch


def test_import_target_reports_failures(tmp_path):
    with pytest.raises(BuildError, match="no attribute 'NoSuchLayerXyz'"):
        import_target("torch.nn.NoSuchLayerXyz")
    with pytest.raises(BuildError, match="No module named 'kalfa_no_such_package'"):
        import_target("kalfa_no_such_package.Thing")
    with pytest.raises(BuildError):
        import_target("")
    (tmp_path / "kalfa_t_broken.py").write_text("import kalfa_missing_dependency_xyz\n\nclass Thing:\n    pass\n",
                                                encoding="utf-8")
    add_search_dir(tmp_path)
    with pytest.raises(BuildError, match="kalfa_missing_dependency_xyz"):
        import_target("kalfa_t_broken.Thing")


def test_add_search_dir_once(tmp_path):
    add_search_dir(tmp_path)
    add_search_dir(str(tmp_path))
    assert SEARCH_DIRS.count(str(tmp_path)) == 1
    assert sys.path.count(str(tmp_path)) == 1


def test_lego_registers_a_plugin_and_returns_the_object():
    @lego("t_thing", slot="model", doc="a thing")
    class Thing:
        def __init__(self, width):
            self.width = width

    @lego("t_factory", slot="loss")
    def factory():
        return torch.nn.L1Loss()

    assert Thing.__name__ == "Thing"
    assert resolve("t_thing") is Thing
    assert entry_of("t_thing").plugin is True
    assert entry_of("t_thing").doc == "a thing"
    assert resolve("t_factory") is factory
    assert import_target(resolve("t_thing")) is Thing


def test_catalog_filters_by_slot_and_word():
    register("t_huber", "torch.nn.HuberLoss", slot="loss", doc="Huber loss, robust to outliers")
    register("t_mae", "torch.nn.L1Loss", slot="loss", doc="mean absolute error")
    register("t_linear", "torch.nn.Linear", slot="model", doc="a linear layer")
    names = [entry.name for entry in catalog()]
    assert names == sorted(names)
    assert {entry.name for entry in catalog(slot="loss")} >= {"t_huber", "t_mae"}
    assert all(entry.slot == "loss" for entry in catalog(slot="loss"))
    by_name = [entry.name for entry in catalog(word="huber")]
    assert "t_huber" in by_name and "t_mae" not in by_name
    assert all("huber" in entry.name or "huber" in entry.doc.lower() for entry in catalog(word="huber"))
    by_doc = [entry.name for entry in catalog(word="robust to outliers")]
    assert by_doc == ["t_huber"]
    assert [entry.name for entry in catalog(slot="model", word="t_linear")] == ["t_linear"]
    assert [entry.name for entry in catalog(slot="model", word="ABSOLUTE")] == []
    assert entry_of("t_missing") is None


def test_import_plugins_imports_modules_and_files(tmp_path):
    (tmp_path / "kalfa_t_plugin.py").write_text(
        "from kalfa.core.aliases import lego\n\n\n@lego('t_plugged', slot='model')\nclass Plugged:\n    pass\n",
        encoding="utf-8")
    (tmp_path / "kalfa_t_file_plugin.py").write_text(
        "from kalfa.core.aliases import register\n\nregister('t_from_file', 'torch.nn.Tanh', slot='model', "
        "plugin=True)\n", encoding="utf-8")
    add_search_dir(tmp_path)
    problems = []
    loaded = import_plugins(["kalfa_t_plugin", "kalfa_t_file_plugin.py"], problems)
    assert problems == []
    assert len(loaded) == 2
    assert entry_of("t_plugged").plugin is True
    assert resolve("t_from_file") == "torch.nn.Tanh"
    import_plugins(["kalfa_missing_plugin_xyz", "kalfa_missing_file_xyz.py"], problems)
    assert [problem.code for problem in problems] == ["plugin_import", "plugin_import"]
    assert all(problem.path == "plugins" for problem in problems)
    assert "kalfa_missing_plugin_xyz" in problems[0].message
