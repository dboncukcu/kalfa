from helpers import minimal, write_config
from kalfa.api import check
from kalfa.config import load_surface, parse_sets


def test_a_plugin_in_a_plugins_folder_next_to_the_config_is_found(workdir):
    (workdir / "plugins").mkdir()
    (workdir / "plugins" / "folder_only_plugin.py").write_text(
        "import kalfa\n\n\n@kalfa.lego('/layer/folder/marker', alias='folder_marker')\n"
        "def marker():\n    import torch\n    return torch.nn.Identity()\n")
    config = minimal()
    config["plugins"] = ["folder_only_plugin"]
    config["model"]["models"]["net"]["nodes"].insert(0, {"uri": "folder_marker"})
    prepared = check([str(write_config(workdir / "cfg.yaml", config))], parse_sets([]))
    assert prepared.problems == []


def test_set_and_param_override_the_surface(workdir):
    config = minimal()
    config["params"]["lr"] = 0.1
    path = write_config(workdir / "cfg.yaml", config)
    surface = load_surface([path], parse_sets(["training.epochs=7", "device=cpu"], ["lr=0.5"]))
    assert surface.data["model"]["models"]["net"]["optimizer"]["params"]["lr"] == 0.5
    assert surface.data["training"]["epochs"] == 7 and surface.data["device"] == "/device/kalfa/cpu"
    assert any(path[0] == "training" for path, _, _ in surface.overrides)
    assert "overrides" in surface.layers_text()
