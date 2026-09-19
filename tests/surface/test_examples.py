import importlib.util
import re
import shutil

import pytest

from helpers import EXAMPLES, examples, inside
from kalfa.api import check
from kalfa.config import parse_sets


INCLUDING = ["11_kfold_cv", "12_resume", "14_sweep_grid"]

TEACHER_MISSING = ("weights_run_missing",
                   "weights of model 'teacher': run 'runs/cifar_resnet50_x' has no resolved.yaml")


def copy_examples(tmp_path):
    root = tmp_path / "examples"
    shutil.copytree(EXAMPLES, root,
                    ignore=shutil.ignore_patterns("runs", "data", "*.parquet", "*.txt", "__pycache__"))
    return root


def make_data(folder):
    spec = importlib.util.spec_from_file_location(f"make_data_{folder.name}", folder / "make_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def included_files(prepared, name):
    return [loaded.file for loaded in prepared.surface.layer.walk() if loaded.file.endswith(f"{name}/config.yaml")]


def test_every_example_folder_has_a_config_a_generator_and_a_readme_heading():
    names = examples()
    assert names == sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir() and path.name != "__pycache__")
    assert len(names) == 17
    for name in names:
        assert (EXAMPLES / name / "config.yaml").is_file()
        assert (EXAMPLES / name / "make_data.py").is_file()
    headings = re.findall(r"^## (\S+)$", (EXAMPLES / "README.md").read_text(), re.MULTILINE)
    assert headings == names


@pytest.mark.parametrize("name", examples())
def test_example_checks_clean_on_the_data_it_generates(name, tmp_path):
    folder = copy_examples(tmp_path) / name
    with inside(folder):
        make_data(folder)
        prepared = check(["config.yaml"], parse_sets([]))
    errors = [(problem.kind, problem.message) for problem in prepared.errors]
    if name == "13_distillation":
        assert errors == [TEACHER_MISSING]
        assert prepared.document is not None
    else:
        assert errors == []
        assert prepared.document is not None
        assert prepared.pipeline is not None
    assert prepared.sizes is not None
    assert prepared.sizes["train"] > 0
    assert bool(included_files(prepared, "01_mlp_regression")) == (name in INCLUDING or name == "01_mlp_regression")
    assert len(included_files(prepared, name)) == 1


@pytest.mark.parametrize("name", INCLUDING)
def test_configs_built_on_the_first_example_include_it_as_the_lower_layer(name, tmp_path):
    folder = copy_examples(tmp_path) / name
    with inside(folder):
        make_data(folder)
        prepared = check(["config.yaml"], parse_sets([]))
    files = [loaded.file for loaded in prepared.surface.layer.walk()]
    included = files.index(str(folder.parent / "01_mlp_regression" / "config.yaml"))
    assert included < files.index(str(folder / "config.yaml"))
    assert prepared.surface.data["data"]["preprocessors"]["std_scaler"] == {"uri": "/pre/sklearn/standard_scaler"}
    assert prepared.surface.data["losses"]["loss_huber"]["params"] == {"delta": 1.0}
    overridden = {".".join(map(str, path)) for path, _, _ in prepared.surface.overrides}
    assert overridden and overridden <= {"record", "params.epochs", "model.nodes", "data.split", "params.fold",
                                         "training.epochs"}
    assert prepared.surface.layers_text().splitlines()[3] == "  3. config.yaml (included by config.yaml)"


def test_the_distillation_teacher_config_checks_clean(tmp_path):
    folder = copy_examples(tmp_path) / "13_distillation"
    with inside(folder):
        make_data(folder)
        prepared = check(["teacher.yaml"], parse_sets([]))
    assert prepared.errors == []
    assert prepared.surface.data["record"] == "runs/cifar_resnet50_x"
    assert prepared.surface.data["model"]["models"]["net"]["nodes"][0]["uri"] == "/layer/timm/timm_backbone"
