import ast
from pathlib import Path

import pytest
import torch

import kalfa
from helpers import housing_frame
from kalfa.core import aliases
from kalfa.core.aliases import ALIASES
from kalfa.core.errors import BuildError


SOURCE = Path(__file__).resolve().parents[2] / "src" / "kalfa"


@pytest.fixture
def saved_aliases():
    saved = dict(ALIASES)
    yield
    ALIASES.clear()
    ALIASES.update(saved)


def unused_imports(path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    imported = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if "noqa" in lines[node.lineno - 1] and "F401" in lines[node.lineno - 1]:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = node.lineno
        else:
            for alias in node.names:
                imported[alias.asname or alias.name] = node.lineno
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    exported = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__"
                                                for target in node.targets):
            exported.update(ast.literal_eval(node.value))
    return [f"{name} (line {line})" for name, line in imported.items() if name not in used and name not in exported]


def table_config(home, targets, split=None):
    housing_frame(60, 0).to_parquet(home / "housing.parquet", index=False)
    return {"seed": 0, "device": "cpu",
            "data": {"use": "table", "file": str(home / "housing.parquet"), "split": split or [0.7, 0.15, 0.15],
                     "features": {"x*": "standard_scaler"}, "targets": targets, "batch": 16},
            "models": {"net": {"use": "mlp", "hidden": [4], "out": 1}},
            "losses": {"mse": "mse"},
            "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
            "train": {"epochs": 1}}


def errors_at(checked, path):
    return [problem for problem in checked.problems if problem.severity == "error" and problem.path == path]


def test_unknown_use_message_names_no_development_phase():
    checked = kalfa.check({"models": {"net": {"use": "mlpp", "hidden": [4], "out": 1}}})
    problem = next(problem for problem in checked.problems if problem.code == "unknown_use")
    assert "kalfa legos" in problem.message
    assert "Aşama" not in problem.message and "phase" not in problem.message.lower()
    with pytest.raises(BuildError) as raised:
        aliases.resolve("no_such_review_name")
    assert "Aşama" not in str(raised.value)


def test_every_source_module_uses_the_names_it_imports():
    found = {}
    for path in sorted(SOURCE.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        unused = unused_imports(path)
        if unused:
            found[str(path.relative_to(SOURCE))] = unused
    assert found == {}


def test_registering_a_std_plot_again_with_the_same_object_passes(saved_aliases):
    from kalfa.std import plots

    assert plots.loss_curve is aliases.import_target(ALIASES["loss_curve"].target)
    entry = aliases.register("loss_curve", plots.loss_curve, slot="plot")
    assert entry is ALIASES["loss_curve"]


def test_registering_a_std_loss_again_with_the_same_class_passes(saved_aliases):
    assert aliases.import_target(ALIASES["mse"].target) is torch.nn.MSELoss
    entry = aliases.register("mse", torch.nn.MSELoss, slot="loss")
    assert entry is ALIASES["mse"]


def test_check_reports_a_misspelled_preprocessor_given_by_name(tmp_path):
    called = kalfa.check(table_config(tmp_path, {"price": {"use": "standrad_scaler"}}))
    assert [problem.code for problem in errors_at(called, "data.targets.price")] == ["unknown_use"]
    named = kalfa.check(table_config(tmp_path, {"price": "standrad_scaler"}))
    listed = kalfa.check(table_config(tmp_path, {"price": ["standard_scaler", "lgo"]}))
    assert not named.ok and errors_at(named, "data.targets.price"), named.text()
    assert not listed.ok and errors_at(listed, "data.targets.price"), listed.text()
    problem = errors_at(named, "data.targets.price")[0]
    assert problem.code == "unknown_use" and "standard_scaler" in (problem.hint or ""), named.text()


def test_check_reports_a_misspelled_split_given_by_name(tmp_path):
    called = kalfa.check(table_config(tmp_path, {"price": "standard_scaler"}, {"use": "sequentail", "ratios": [1]}))
    assert errors_at(called, "data.split"), called.text()
    named = kalfa.check(table_config(tmp_path, {"price": "standard_scaler"}, "sequentail"))
    assert not named.ok and errors_at(named, "data.split"), named.text()
