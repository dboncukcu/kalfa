import json
import shutil
import sys
from pathlib import Path

import pytest

import kalfa


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))

import demo  # noqa: E402


FOLDERS = demo.folders(EXAMPLES)
NAMES = [folder.name for folder in FOLDERS]
PLUGIN_SKIP = {"make_data.py", "demo.py"}


@pytest.fixture(scope="session")
def tree(tmp_path_factory):
    copy = tmp_path_factory.mktemp("examples") / "examples"
    shutil.copytree(EXAMPLES, copy, ignore=shutil.ignore_patterns("runs", "__pycache__", "*.parquet", "data",
                                                                  "*.txt"))
    return copy


def test_every_example_has_a_recipe_a_config_and_its_data_script():
    assert len(FOLDERS) >= 17
    for folder in FOLDERS:
        recipe = demo.recipe_of(folder)
        assert recipe.get("title") and recipe.get("steps"), folder.name
        assert (folder / "make_data.py").is_file() or recipe.get("needs"), folder.name
        for step in recipe["steps"]:
            kind = demo.kind_of(step)
            if kind in ("run", "sweep"):
                assert (folder / step[kind]).is_file(), f"{folder.name}: {step[kind]}"


def test_plugin_module_names_are_unique_across_examples():
    seen = {}
    for folder in FOLDERS:
        for module in folder.glob("*.py"):
            if module.name in PLUGIN_SKIP:
                continue
            assert module.name not in seen, f"{module.name} in {folder.name} and {seen.get(module.name)}"
            seen[module.name] = folder.name


def test_examples_readme_lists_every_example():
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"`{name}`" in text, name


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_example_plays_its_recipe(name, tree, tmp_path):
    folder = tree / name
    demo.prepare(folder, tree)
    results = demo.play(folder, tmp_path / "records", device="cpu", log=lambda text: None)
    assert results
    failures = [result for result in results if not result["ok"]]
    assert not failures, json.dumps(failures, indent=1)
    for result in results:
        if result["kind"] == "run":
            record = Path(result["record"])
            assert (record / "history.jsonl").is_file() and (record / "plan.json").is_file()
            assert kalfa.open(record).summary["turn"] >= 1
