import json
import re
import subprocess
import sys

import pytest
import yaml
from rich.console import Console

from kalfa import cli
from kalfa.commands import legos
from kalfa.core import aliases
from kalfa.core.errors import BuildError


ANSI = re.compile(r"\x1b\[")
LISTING = ("import json\nfrom kalfa.commands import legos\nfrom kalfa.core import aliases\nlegos.load_registrars()\n"
           "print(json.dumps(sorted(name for name, entry in aliases.ALIASES.items() if not entry.plugin)))")


@pytest.fixture(scope="module")
def std_names():
    found = subprocess.run([sys.executable, "-c", LISTING], capture_output=True, text=True, check=True)
    names = json.loads(found.stdout)
    assert len(names) >= 90
    return names


def entries(names):
    legos.load_registrars()
    return [aliases.entry_of(name) for name in names]


def mentions(tree, name):
    if isinstance(tree, dict):
        return any(mentions(value, name) for value in tree.values())
    if isinstance(tree, list):
        return any(mentions(value, name) for value in tree)
    return tree == name


def test_every_std_name_has_a_yaml_example_that_uses_it(std_names):
    missing = []
    for entry in entries(std_names):
        if not entry.examples:
            missing.append(entry.name)
            continue
        for example in entry.examples:
            assert isinstance(example, str), entry.name
            loaded = yaml.safe_load(example)
            assert isinstance(loaded, dict), (entry.name, example)
            assert mentions(loaded, entry.name), (entry.name, example)
    assert not missing, missing


def test_notes_cover_required_params_and_every_param_of_kalfa_targets(std_names):
    gaps = []
    for entry in entries(std_names):
        found = legos.parameters(entry)
        names = [param for param, _, _ in found]
        stale = [param for param in entry.params if param not in names]
        assert not stale, (entry.name, stale)
        own = aliases.target_path(entry.target).startswith("kalfa.")
        for param, _, required in found:
            if (own or required) and not str(entry.params.get(param) or "").strip():
                gaps.append(f"{entry.name}.{param}")
    assert not gaps, gaps


def test_annotate_merges_notes_and_examples_without_repeats(monkeypatch):
    entry = aliases.Entry("t_phase6_note", "kalfa.std.models.Mlp", "model", "a test entry")
    monkeypatch.setitem(aliases.ALIASES, "t_phase6_note", entry)
    example = "models:\n  net: {use: t_phase6_note, hidden: [8], out: 1}"
    aliases.annotate("t_phase6_note", params={"hidden": "sizes"}, examples=[example])
    aliases.annotate("t_phase6_note", params={"out": "width"}, examples=[example])
    assert entry.params == {"hidden": "sizes", "out": "width"} and entry.examples == [example]
    with pytest.raises(BuildError):
        aliases.annotate("t_phase6_missing", params={"a": "b"})


def test_entry_text_shows_notes_under_params_then_examples_and_used_in():
    text = legos.entry_text("standard_scaler")
    lines = text.splitlines()
    assert not ANSI.search(text)
    assert "examples:" in lines and "used in:" in lines
    assert lines.index("examples:") < lines.index("used in:")
    assert "  examples/01_mlp_regression/config.yaml" in lines
    entry = aliases.entry_of("standard_scaler")
    for param, note in entry.params.items():
        index = next(position for position, line in enumerate(lines) if line.startswith(f"  {param} "))
        assert lines[index + 1].startswith("      ") and note.split()[0] in lines[index + 1], (param, lines)
    mlp = legos.entry_text("mlp").splitlines()
    assert "  hidden (required)" in mlp and "  activation = 'relu'" in mlp and "examples:" in mlp


def test_used_in_counts_lego_uses_and_not_tensor_names():
    assert "examples/01_mlp_regression/config.yaml" in legos.used_in("adam")
    assert "examples/01_mlp_regression/config.yaml" in legos.used_in("mse")
    assert "examples/01_mlp_regression/config.yaml" in legos.used_in("standard_scaler")
    label = legos.used_in("label")
    assert "examples/02_mlp_classification/config.yaml" in label
    assert "examples/07_wgan_gp/config.yaml" not in label
    assert legos.used_in("adam") == sorted(legos.used_in("adam"))
    assert legos.used_in("qqzzxxw") == []


def test_the_renderable_colors_the_yaml_example():
    console = Console(record=True, width=100, force_terminal=True, color_system="truecolor")
    console.print(legos.entry_renderable("adam"))
    colored = console.export_text(styles=True, clear=False)
    plain = console.export_text()
    first = aliases.entry_of("adam").examples[0].strip().splitlines()[0].strip()
    assert first in plain and "lr" in plain
    assert ANSI.search(colored)


def test_the_cli_prints_plain_text_when_piped(capsys):
    assert cli.main(["legos", "adam"]) == 0
    out = capsys.readouterr().out
    assert "examples:" in out and "lr" in out and not ANSI.search(out)


def test_docs_markdown_has_a_section_per_name_with_yaml_blocks(std_names):
    text = legos.docs_markdown()
    assert "```yaml" in text
    missing = [name for name in std_names if f"### {name}\n" not in text]
    assert not missing, missing


def test_used_in_also_counts_the_import_path_of_the_target():
    assert "examples/06_vae/config.yaml" in legos.used_in("huber")


def test_yaml_lists_reach_tuple_and_tensor_parameters():
    import numpy
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from kalfa.core.coerce import coerced
    from kalfa.std.data.columns import step_of
    scaler = step_of({"use": "minmax_scaler", "feature_range": [-1, 1]}, "data.features.x")
    assert scaler.feature_range == (-1, 1)
    assert scaler.fit_transform(numpy.arange(5.0).reshape(-1, 1)).min() == -1.0
    found = coerced(torch.nn.BCEWithLogitsLoss, {"pos_weight": 4.0, "weight": [1.0, 2.0], "reduction": "mean"})
    assert isinstance(found["pos_weight"], torch.Tensor) and found["weight"].tolist() == [1.0, 2.0]
    assert found["reduction"] == "mean"
    assert coerced(MinMaxScaler, {"feature_range": [0, 2]}) == {"feature_range": (0, 2)}
    assert coerced(torch.nn.Linear, {"in_features": 3, "out_features": 2}) == {"in_features": 3, "out_features": 2}
