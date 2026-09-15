import pytest
import regenerate

from helpers import ROOT


@pytest.mark.parametrize("writer", list(regenerate.WRITERS))
def test_every_golden_file_is_current(writer):
    stale = []
    for path, text in regenerate.WRITERS[writer]().items():
        target = ROOT / path
        if not target.exists() or target.read_text() != text:
            stale.append(str(path))
    assert stale == [], f"stale golden files; run: uv run python tools/regenerate.py {writer}"
