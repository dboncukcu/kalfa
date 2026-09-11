import sys
from pathlib import Path

import pytest
from cirak.registry import registry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "examples" / "alad"))

from helpers import write_housing  # noqa: E402


KEPT = (str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools"), sys.prefix, sys.base_prefix)


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
    write_housing(tmp_path / "housing.parquet")
    return tmp_path


@pytest.fixture
def write(tmp_path):
    def _write(name, text):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return str(target)
    return _write
