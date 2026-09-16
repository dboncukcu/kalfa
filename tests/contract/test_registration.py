"""kalfa.lego: the kind comes from the first segment of the URI, cirak stores its own kind names for triggers."""

import pytest
from cirak.registry import registry

import kalfa
from kalfa.kinds import cirak_kind, kalfa_kind, kind_of, kinds


def test_kind_is_the_first_uri_segment():
    assert kind_of("/criterion/kalfa/mae") == "criterion" and kind_of("/turn/proj/x") == "turn"
    assert kind_of("/data/kalfa/class_weights") == "data" and kind_of("/builder/kalfa/module") == "builder"
    for bad in ("criterion/kalfa/mae", "/loss/kalfa/mae", "/criterion", "/std/kalfa/pack", 3):
        with pytest.raises(ValueError):
            kind_of(bad)
    assert kalfa_kind("/loss/kalfa/mae") is None and kalfa_kind("/plot/kalfa/loss_curve") == "plot"
    assert cirak_kind("trigger") == "predicate" and cirak_kind("layer") == "layer" and cirak_kind("data") == "data"
    assert "predicate" not in kinds() and kinds()[-2:] == ["builder", "data"]


def test_kalfa_lego_registers_with_the_derived_kind():
    @kalfa.lego("/metric/test/registered", state=True, description="a test metric")
    def registered():
        return None

    entry = registry.lookup("/metric/test/registered")
    assert entry is not None and entry.facts.kind == "metric" and entry.facts.state is True
    assert kalfa_kind("/metric/test/registered") == "metric"

    @kalfa.lego("/trigger/test/always", partial=True)
    def always(metrics, turn_index, state):
        return True

    assert registry.facts("/trigger/test/always").kind == "predicate"
    assert kalfa_kind("/trigger/test/always") == "trigger"
    with pytest.raises(ValueError, match="kalfa kind"):
        kalfa.lego("/nope/test/x")(lambda: None)
    with pytest.raises(ValueError, match="drop kind="):
        kalfa.lego("/layer/test/x", kind="layer")(lambda: None)


def test_every_std_uri_follows_the_rule():
    from kalfa.std import STD_URIS

    for uri in STD_URIS:
        assert kalfa_kind(uri) is not None, uri
        stored = registry.facts(uri).kind
        assert stored == cirak_kind(kalfa_kind(uri)), uri


@pytest.mark.subprocess
def test_importing_kalfa_loads_no_specialised_library():
    import subprocess
    import sys

    code = ("import sys, kalfa; print(sorted(name for name in ('torch', 'numpy', 'pandas', 'sklearn', "
            "'torchmetrics', 'matplotlib', 'PIL', 'scipy', 'optuna', 'seaborn', 'torchview') "
            "if name in sys.modules))")
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert found.stdout.strip() == "[]"


def test_the_uri_is_the_path():
    from pathlib import Path

    from kalfa.std import STD_URIS

    root = Path(kalfa.std.__file__).parent
    files = sorted(path for path in root.glob("*/*/*.py") if path.name not in ("__init__.py", "base.py"))
    modules = {"kalfa.std." + ".".join(path.relative_to(root).with_suffix("").parts) for path in files}
    registered = set()
    for uri in STD_URIS:
        kind, pack, name = uri.strip("/").split("/")
        module = registry.lookup(uri).target.partition(":")[0]
        assert module.startswith(f"kalfa.std.{kind}.{pack}."), (uri, module)
        registered.add(module)
    assert registered == modules, modules ^ registered
    assert len(STD_URIS) == 314
    packages = {path.parent for path in root.rglob("*.py")}
    for directory in packages:
        assert (directory / "__init__.py").exists(), directory


def test_a_pack_declares_only_its_own_uris():
    from kalfa.registration import pack

    declare = pack("kalfa.std.layer.torch")
    for uri in ("/layer/kalfa/mine", "/layer/torch/deep/mine", "/layer/torch/"):
        with pytest.raises(ValueError, match="declares /layer/torch/"):
            declare(uri, "blocks:mine")
    for target in ("blocks.mine", "blocks:", ":mine", "kalfa.std.layer.torch.blocks:mine"):
        with pytest.raises(ValueError, match="module:name"):
            declare("/layer/torch/mine", target)


@pytest.mark.slow
def test_every_std_uri_resolves_to_its_target():
    from kalfa.std import STD_URIS

    for uri in sorted(STD_URIS):
        assert registry.resolve(uri) is not None, uri


@pytest.mark.subprocess
def test_listing_the_catalog_loads_no_lego_module():
    import subprocess
    import sys

    code = ("import contextlib, io, sys\n"
            "from kalfa.cli import main\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            "    main(['ls'])\n"
            "print(sorted(name for name in ('torch', 'numpy', 'pandas') if name in sys.modules))\n")
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert found.stdout.strip() == "[]"
