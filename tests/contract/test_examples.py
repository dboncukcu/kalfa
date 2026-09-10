from helpers import EXAMPLES, examples


def test_every_example_folder_is_complete():
    folders = sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir() and not path.name.startswith("."))
    assert folders == examples()
    readme = (EXAMPLES / "README.md").read_text()
    for name in examples():
        assert (EXAMPLES / name / "make_data.py").exists(), name
        assert f"## {name}" in readme, name


def test_the_included_configs_are_the_examples_themselves():
    for name in ("11_kfold_cv", "12_resume", "14_sweep_grid"):
        text = (EXAMPLES / name / "config.yaml").read_text()
        assert "- ../01_mlp_regression/config.yaml" in text
        assert not (EXAMPLES / name / "01_mlp_regression.yaml").exists()
