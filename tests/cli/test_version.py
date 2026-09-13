"""kalfa --version: the versions a bug report needs, on one line."""

import pytest

from kalfa import __version__
from kalfa.cli import main, version_text


def test_version_prints_kalfa_with_its_dependencies(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith(f"kalfa {__version__} (") and out == version_text()
    assert all(f"{name} " in out for name in ("cirak", "tezgah", "torch", "python"))
