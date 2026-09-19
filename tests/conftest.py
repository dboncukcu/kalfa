import os
import sys
from pathlib import Path

import pytest
from cirak.registry import registry


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from data import write_all, write_reference  # noqa: E402
from helpers import config_path, inside  # noqa: E402
from kalfa.api import run  # noqa: E402
from kalfa.config import parse_sets  # noqa: E402


KEPT = (str(ROOT / "src"), str(ROOT / "tests"), sys.prefix, sys.base_prefix)

os.environ["NO_COLOR"] = "1"
for name in ("FORCE_COLOR", "CLICOLOR_FORCE", "PYTHON_COLORS"):
    os.environ.pop(name, None)


@pytest.fixture(autouse=True)
def scoped_registry():
    before = set(registry.uris())
    with registry.scoped():
        yield registry
        added = [uri for uri in registry.uris() if uri not in before]
        registrars = {getattr(registry.lookup(uri).target, "__module__", None) for uri in added}
    for name in registrars:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None) or ""
        if module is not None and file and not file.startswith(KEPT):
            del sys.modules[name]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_reference(tmp_path / "reference.parquet")
    write_reference(tmp_path / "new.parquet", rows=100, seed=5)
    return tmp_path


@pytest.fixture(scope="session")
def root(tmp_path_factory):
    return write_all(tmp_path_factory.mktemp("runs"))


@pytest.fixture(scope="session")
def trained(root):
    results = {}

    def get(name, when="fixed", sets=(), params=()):
        key = (name, when, tuple(sets), tuple(params))
        if key not in results:
            with inside(root):
                results[key] = run([config_path(name)], parse_sets(list(sets), list(params)), when=when)
        return results[key]
    return get


@pytest.fixture
def at_root(root, monkeypatch):
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def reference(trained, at_root):
    return trained("reference")
