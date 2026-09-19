from pathlib import Path

import pytest
from PIL import Image

from helpers import load_config


pytestmark = pytest.mark.slow


def test_every_plot_definition_wrote_a_non_empty_file(reference):
    plots = load_config("reference")["plots"]
    written = sorted(path.name for path in (Path(reference.record) / "plots").iterdir())
    models = ["tower", "head_lin", "head_aux", "tail_stem", "tail_head", "lambdas", "full"]
    expected = sorted([*(f"{name}.png" for name in plots if not name.startswith("architecture")),
                       *(f"architecture_{name}.png" for name in models), "architecture_text.txt"])
    assert written == expected
    for path in (Path(reference.record) / "plots").iterdir():
        assert path.stat().st_size > 0


def test_the_figures_section_sets_the_resolution_of_every_plot(reference):
    for path in sorted((Path(reference.record) / "plots").glob("*.png")):
        with Image.open(path) as image:
            assert tuple(round(value) for value in image.info["dpi"]) == (72, 72), path.name
            width, height = image.size
        if path.name == "loss_curve.png":
            assert width < 600 and height < 600


def test_the_architecture_text_describes_every_model(reference):
    text = (Path(reference.record) / "plots" / "architecture_text.txt").read_text()
    for name in ("tower", "head_lin", "head_aux", "tail_stem", "tail_head", "lambdas", "full"):
        assert name in text
    assert "h_0" in text and "h_1" in text
