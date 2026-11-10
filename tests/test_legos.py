import io
import math
import sys
from itertools import takewhile
from pathlib import Path

import pytest
from rich.console import Console
from ruamel.yaml import YAML

import kalfa
from kalfa import cli
from kalfa.commands import legos
from kalfa.core.aliases import ALIASES, annotate, entry_of, register
from kalfa.core.build import SLOT_KEYS, TOP_LEVEL
from kalfa.core.errors import KalfaError
from kalfa.std.notes import LOSSES


def spread(a, b=2, *rest, c, **more):
    return a


class Thing:
    def __init__(self, x, y=1, *rest, **more):
        self.x = x


@pytest.fixture(autouse=True)
def restore():
    legos.load_registrars()
    saved = dict(ALIASES)
    yield
    ALIASES.clear()
    ALIASES.update(saved)


def headings(text):
    return [line for line in text.splitlines() if line and not line.startswith(" ")]


def test_catalog_lists_std_names_grouped_by_category_in_order():
    text = legos.catalog_text()
    assert "mlp" in text and "table" in text and "standard_scaler" in text and "plateau" in text
    groups = headings(text)
    assert groups.index("data") < groups.index("models") < groups.index("losses") < groups.index("metrics")
    assert groups.index("optimizers") < groups.index("triggers") < groups.index("sweep strategies")
    assert groups == [category for category, _ in legos.categories()]
    assert len(groups) == len(set(groups))


def test_catalog_rows_are_sorted_and_wrapped_with_a_hanging_indent():
    text = legos.catalog_text(slot="value")
    for block in text.split("\n\n"):
        rows = [line.split()[0] for line in block.splitlines()[1:] if line.startswith("  ") and line[2] != " "]
        assert rows == sorted(rows)
    lines = text.splitlines()
    assert all(len(line) <= 100 for line in lines)
    model = legos.catalog_text(slot="model").splitlines()
    first = next(index for index, line in enumerate(model) if line.split()[:1] == ["mlp"])
    lead = model[first].index("linear layers")
    assert model[first + 1].startswith(" " * lead) and model[first + 1][lead] != " "


def test_word_filter_matches_name_or_doc_case_insensitively():
    text = legos.catalog_text("scaler")
    assert "standard_scaler" in text and "minmax_scaler" in text and "robust_scaler" in text
    rows = [line.split()[0] for line in text.splitlines() if line.startswith("  ") and line[2] != " "]
    assert rows == ["minmax_scaler", "robust_scaler", "standard_scaler"]
    assert legos.catalog_text("SCALER") == text
    doc = legos.catalog_text("interquartile")
    assert "robust_scaler" in doc and "standard_scaler" not in doc


def test_slot_filter_keeps_one_slot_and_combines_with_the_word():
    text = legos.catalog_text(slot="metric")
    assert "rmse" in text and "adam" not in text and "mlp" not in text
    assert headings(text) == ["metrics"]
    assert legos.catalog_text(slot="METRIC") == text
    both = legos.catalog_text("scaler", slot="value")
    assert "standard_scaler" in both and headings(both) == ["table preprocessing"]
    assert headings(legos.catalog_text(slot="loss")) == ["losses", "objectives"]


def test_empty_results_say_so_and_suggest_a_close_name():
    text = legos.catalog_text("mlpp")
    assert text.splitlines()[0] == "no name matches mlpp"
    assert "did you mean" in text and "mlp" in text
    far = legos.catalog_text("qqzzxxw")
    assert far == "no name matches qqzzxxw"
    narrowed = legos.catalog_text("scaler", slot="metric")
    assert narrowed.startswith("no name matches scaler in slot metric")
    nothing = legos.catalog_text(slot="nothing")
    assert nothing.startswith("no name of slot nothing") and "model" in nothing


def test_entry_text_shows_target_slot_doc_and_params():
    text = legos.entry_text("mlp")
    lines = text.splitlines()
    assert lines[0] == "mlp"
    assert "kalfa.std.models.Mlp" in text and "model" in lines[2]
    assert lines[3] == "  category  models" and lines[4] == "  goes in   models.<name>"
    assert "linear layers over hidden sizes" in text
    assert "params:" in lines
    assert "  hidden (required)" in lines and "  out (required)" in lines
    assert "  activation = 'relu'" in lines and "  dropout = 0.0" in lines and "  in_features = None" in lines
    assert "self" not in text
    assert lines.index("params:") < lines.index("examples:") < lines.index("used in:")


def test_entry_text_of_an_unknown_name_suggests():
    text = legos.entry_text("mlpp")
    assert text.splitlines()[0] == "no name mlpp"
    assert "did you mean mlp" in text
    assert legos.entry_text("qqzzxxw") == "no name qqzzxxw"


def test_signature_of_lists_params_in_order_with_required():
    assert legos.signature_of(entry_of("mlp")) == [("hidden", "required"), ("out", "required"),
                                                    ("activation", "relu"), ("dropout", 0.0), ("in_features", None)]
    assert legos.signature_of("mlp") == legos.signature_of(entry_of("mlp"))
    assert legos.signature_of(entry_of("adam"))[0][0] == "lr"
    with pytest.raises(KalfaError, match="no name"):
        legos.signature_of("qqzzxxw")


def test_signature_skips_self_and_star_args():
    register("t_legos_spread", spread, slot="value", doc="spread")
    register("t_legos_thing", Thing, slot="model", doc="thing")
    assert legos.signature_of(entry_of("t_legos_spread")) == [("a", "required"), ("b", 2), ("c", "required")]
    assert legos.signature_of(entry_of("t_legos_thing")) == [("x", "required"), ("y", 1)]
    text = legos.entry_text("t_legos_thing")
    assert "  x (required)" in text and "  y = 1" in text and "rest" not in text and "more" not in text


def test_entry_without_slot_goes_last_and_shows_examples():
    register("t_legos_free", spread, doc="a free function", examples=["use: t_legos_free\na: 1", {"use": "x"}])
    text = legos.catalog_text()
    assert headings(text)[-1] == "other"
    assert "t_legos_free" in legos.catalog_text(slot="unknown")
    page = legos.entry_text("t_legos_free").splitlines()
    assert "  slot      unknown" in page and "  category  other" in page
    assert not any(line.split()[:2] == ["goes", "in"] for line in page)
    assert "examples:" in page
    assert "  use: t_legos_free" in page and "  a: 1" in page and '  {"use": "x"}' in page


def test_an_unimportable_target_does_not_break_the_pages():
    register("t_legos_broken", "kalfa_t_legos_missing.Thing", slot="model", doc="broken target")
    text = legos.entry_text("t_legos_broken")
    assert "kalfa_t_legos_missing.Thing" in text and "not readable" in text
    assert "t_legos_broken" in legos.catalog_text()
    assert "Params: not readable" in section_of(legos.docs_markdown(), "t_legos_broken")


def section_of(text, name):
    start = text.index(f"### {name}\n")
    end = text.find("\n#", start + 1)
    return text[start:] if end < 0 else text[start:end]


def test_docs_markdown_has_a_title_category_sections_and_a_section_per_name():
    register("t_legos_pipe", spread, slot="value", doc="left | right")
    text = legos.docs_markdown()
    lines = text.splitlines()
    assert lines[0] == "# kalfa legos"
    sections = [line for line in lines if line.startswith("## ")]
    assert sections.index("## data") < sections.index("## models") < sections.index("## losses")
    assert sections[-1] == "## other"
    assert len(sections) == len(set(sections))
    assert "## optimizers\n\nGoes in `optimizers.<name>`.\n\n### adam\n" in text
    assert "## layers\n\nGoes in layers of a stack model.\n" in text
    assert "## other\n\n### " in text
    mlp = section_of(text, "mlp")
    assert "Target `kalfa.std.models.Mlp`." in mlp and "linear layers over hidden sizes" in mlp
    assert "* `hidden` (required): the widths of the hidden layers" in mlp
    assert "* `activation='relu'`: " in mlp and "* `in_features=None`: " in mlp
    assert "```yaml\nmodels:\n" in mlp and "Used in " in mlp and "`examples/01_mlp_regression/config.yaml`" in mlp
    assert "In a sweep section, random is the random_search strategy." in section_of(text, "random")
    pipe = section_of(text, "t_legos_pipe")
    assert "left | right" in pipe and "* `a` (required)\n" in pipe and "Example" not in pipe
    assert "### standard_scaler\n" in text
    assert not text.endswith("\n")


def test_is_name_is_an_exact_name():
    assert legos.is_name("mlp") and legos.is_name("standard_scaler")
    assert not legos.is_name("scaler") and not legos.is_name("MLP") and not legos.is_name(None)


def test_cli_lists_describes_and_prints_docs(capsys):
    assert cli.main(["legos", "mlp"]) == 0
    assert "kalfa.std.models.Mlp" in capsys.readouterr().out
    assert cli.main(["legos", "--slot", "metric"]) == 0
    assert "rmse" in capsys.readouterr().out
    assert cli.main(["legos", "--docs"]) == 0
    assert capsys.readouterr().out.startswith("# kalfa legos")
    assert cli.main(["legos", "mlpp"]) == 0
    assert "did you mean mlp" in capsys.readouterr().out


def test_a_name_that_is_also_a_sweep_strategy_says_so_on_its_page():
    page = legos.entry_text("random").splitlines()
    assert page[2] == "  slot      value"
    assert "  sweep     in a sweep section, random is the random_search strategy" in page
    assert not any("in a sweep section" in line for line in legos.entry_text("grid").splitlines())
    assert not any("in a sweep section" in line for line in legos.entry_text("mlp").splitlines())


def test_plugin_legos_join_the_category_of_their_slot_and_are_marked():
    register("t_legos_plug", spread, slot="loss", doc="a plugin loss", plugin=True)
    register("t_legos_plug_pre", "kalfa.std.data.pre.Abs", slot="value", doc="a plugin value", plugin=True)
    register("t_legos_plug_free", spread, doc="a plugin without a slot", plugin=True)
    assert legos.category_of("t_legos_plug") == "losses" and legos.place_of("t_legos_plug") == "losses.<name>"
    assert legos.category_of("t_legos_plug_pre") == "other" and legos.place_of("t_legos_plug_pre") is None
    assert legos.category_of("t_legos_plug_free") == "other"
    assert [entry.name for entry in dict(legos.categories("t_legos_plug"))["losses"]] == ["t_legos_plug"]
    row = next(line for line in legos.catalog_text().splitlines() if line.split()[:1] == ["t_legos_plug"])
    assert row.split()[1:] == ["(plugin)", "a", "plugin", "loss"]
    page = legos.entry_text("t_legos_plug").splitlines()
    assert "  category  losses" in page and "  goes in   losses.<name>" in page and "  from      a plugin" in page
    assert "(plugin)" in legos.entry_label(entry_of("t_legos_plug")).plain


def test_category_of_takes_names_or_entries_and_unknown_names_are_other():
    assert legos.category_of("mse") == legos.category_of(entry_of("mse")) == "losses"
    assert legos.category_of("ddpm") == "objectives" and legos.category_of("ddpm_sampler") == "samplers"
    assert legos.category_of("qqzzxxw") == "other" and legos.category_of(None) == "other"
    assert legos.place_of("cosine_beta") == "the schedule of ddpm"
    assert legos.place_of("char_tokenizer") == "data.tokenizer" and legos.category_of("text_lines") == "data"


def test_the_rich_page_and_the_category_page_name_the_place():
    page = legos.entry_renderable("kfold")
    lines = [line.rstrip() for line in render(page).splitlines()]
    assert "  category  splits" in lines and "  goes in   data.split" in lines
    found = dict(legos.categories())
    overview = render(legos.category_renderable("splits", found["splits"])).splitlines()
    assert overview[0] == "splits" and overview[1].split() == ["goes", "in", "data.split"]
    assert [line.split()[0] for line in overview[3:]] == [entry.name for entry in found["splits"]]
    other = render(legos.category_renderable("other", [])).splitlines()
    assert other[0] == "other" and not any("goes in" in line for line in other)


def render(renderable):
    console = Console(width=200, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


USES = """
data:
  use: table
  file: t.parquet
  split: {use: sequential, ratios: [0.8, 0.2]}
  features: {"x*": [abs, standard_scaler]}
  targets: {y: label}
  feed: {use: window, size: 4}
models:
  net: {use: mlp, hidden: [8], out: 2, activation: gelu, init: xavier_init, inputs: [relu], outputs: [tanh]}
losses:
  mse: mse
  ce: {use: cross_entropy, pred: sigmoid, target: silu}
optimizers:
  main: {use: adam, over: "@models.net", minimizes: "@losses.ce", schedule: warmup}
train:
  epochs: 1
  step: supervised
  log: [tensorboard]
rules:
  stop: {when: {use: plateau, watch: val/f1, patience: 2}, run: write_samples}
plots:
  curve: loss_curve
generate: lm_sampler
sweep:
  strategy: grid
  space: {lr: {log: [0.1, 1.0]}}
"""
COUNTED = ("table", "sequential", "abs", "standard_scaler", "label", "window", "mlp", "gelu", "xavier_init", "mse",
           "cross_entropy", "adam", "warmup", "supervised", "tensorboard", "plateau", "write_samples", "loss_curve",
           "lm_sampler", "grid")
PLACES = {
    "data": ["data"],
    "model": ["models.*"],
    "loss": ["losses.*"],
    "metric": ["metrics.*"],
    "optimizer": ["optimizers.*"],
    "plot": ["plots.*"],
    "sampler": ["generate", "rules.*.run.sampler"],
    "step": ["train.step"],
    "trigger": ["rules.*.when"],
    "action": ["rules.*.run"],
    "logger": ["train.log.*"],
    "strategy": ["sweep.strategy"],
    "schedule": ["optimizers.*.schedule", "losses.*.w_kl"],
}


def calls_in(tree):
    if isinstance(tree, dict):
        if isinstance(tree.get("use"), str):
            yield tree
        for value in tree.values():
            yield from calls_in(value)
    elif isinstance(tree, list):
        for value in tree:
            yield from calls_in(value)


def mentions_at(tree, name, path=()):
    if tree == name or (isinstance(tree, dict) and tree.get("use") == name):
        yield path
    items = tree.items() if isinstance(tree, dict) else enumerate(tree) if isinstance(tree, list) else ()
    for key, value in items:
        if key != "use":
            yield from mentions_at(value, name, (*path, str(key)))


def placed(path, pattern):
    parts = pattern.split(".")
    return len(path) == len(parts) and all(part in ("*", step) for part, step in zip(parts, path))


def call_problems(call):
    found = entry_of(call["use"])
    if found is None:
        return []
    signature = legos.parameters(found)
    allowed = {param for param, _, _ in signature} | set(SLOT_KEYS.get(found.slot, ()))
    extra = sorted(set(call) - {"use"} - allowed)
    missing = [param for param, _, required in signature if required and param not in call]
    return [(call["use"], extra, missing)] if extra or missing else []


def test_every_example_sits_in_its_section_and_calls_with_signature_params_and_slot_keys():
    problems = []
    for entry in [entry for entry in ALIASES.values() if not entry.plugin]:
        for example in entry.examples:
            tree = YAML(typ="safe", pure=True).load(example)
            unknown = sorted(set(tree) - set(TOP_LEVEL))
            if unknown:
                problems.append((entry.name, "sections", unknown))
            places = PLACES.get(entry.slot, ())
            spots = list(mentions_at(tree, entry.name))
            if places and not any(placed(spot, place) for spot in spots for place in places):
                problems.append((entry.name, "placed", spots))
            problems.extend((entry.name, *found) for call in calls_in(tree) for found in call_problems(call))
    assert not problems, problems


def test_notes_reach_the_sweep_strategies_once_however_often_the_registrars_load():
    legos.load_registrars()
    legos.load_registrars()
    for name in ("grid", "random_search", "optuna"):
        entry = entry_of(name)
        assert entry.examples and len(entry.examples) == len(set(entry.examples))
        assert set(entry.params) == {param for param, _, _ in legos.parameters(entry)}


def test_notes_sit_wrapped_under_their_params_and_the_page_ends_with_the_examples():
    register("t_legos_noted", Thing, slot="model", doc="noted")
    example = "models:\n  net: {use: t_legos_noted, x: 1}"
    annotate("t_legos_noted", params={"x": "a long note " * 20}, examples=[example])
    lines = legos.entry_text("t_legos_noted").splitlines()
    first = lines.index("  x (required)")
    notes = list(takewhile(lambda line: line.startswith("      "), lines[first + 1:]))
    assert len(notes) > 1 and all(len(line) <= 100 for line in notes) and notes[0].startswith("      a long")
    assert lines[lines.index("  y = 1") + 1] == ""
    assert lines[lines.index("examples:") + 1:] == ["  models:", "    net: {use: t_legos_noted, x: 1}"]
    assert "used in:" not in lines


def test_used_in_reads_the_lego_places_and_skips_tensor_names(tmp_path, monkeypatch):
    (tmp_path / "examples" / "a").mkdir(parents=True)
    (tmp_path / "examples" / "a" / "config.yaml").write_text(USES, encoding="utf-8")
    (tmp_path / "examples" / "b").mkdir()
    (tmp_path / "examples" / "b" / "broken.yaml").write_text("data: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(legos, "ROOT", tmp_path)
    for name in COUNTED:
        assert legos.used_in(name) == ["examples/a/config.yaml"], name
    for name in ("relu", "tanh", "sigmoid", "silu", "log", "random", "x"):
        assert legos.used_in(name) == [], name
    monkeypatch.setattr(legos, "ROOT", None)
    assert legos.used_in("mse") == []


def test_the_cli_prints_the_colored_page_on_a_terminal(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli.main(["legos", "adam"]) == 0
    out = capsys.readouterr().out
    assert "\x1b[" in out and "examples:" in out and "optimizers" in out



@pytest.mark.slow
def test_the_bce_logits_example_trains_on_the_anomaly_table(tmp_path):
    config = YAML(typ="safe").load(LOSSES["bce_logits"]["examples"][0])
    folder = Path(__file__).resolve().parents[1] / "examples" / "alad"
    config["data"]["file"] = str(folder / config["data"]["file"])
    config["optimizers"] = {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.bce"}}
    config["train"] = {"epochs": 1}
    result = kalfa.run(config, record=str(tmp_path / "bce"))
    assert math.isfinite(result.history[-1]["train/bce"])
