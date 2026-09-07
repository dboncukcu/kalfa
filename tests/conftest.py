import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent / "plugins"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "alad"))

from kalfa.synthetic import write_housing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CONFIG_01 = CONFIGS / "01_mlp_regression.yaml"
DUMP_01 = CONFIGS / "dumps" / "01.flow.yaml"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A temporary working directory holding housing.parquet, the way config 01 expects it."""
    monkeypatch.chdir(tmp_path)
    write_housing(tmp_path / "housing.parquet")
    return tmp_path


@pytest.fixture
def config_01():
    return str(CONFIG_01)


@pytest.fixture
def write(tmp_path):
    def _write(name, text):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return str(target)
    return _write
