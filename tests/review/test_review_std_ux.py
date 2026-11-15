import pandas
import pytest
from PIL import Image

import kalfa
from kalfa.core.errors import BuildError
from kalfa.std.data import transforms


def config_around(data):
    return {
        "data": data,
        "models": {"net": {"use": "mlp", "hidden": [4], "out": 1}},
        "losses": {"mse": "mse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 1},
    }


def problems_of(config):
    checked = kalfa.check(config)
    return [(problem.code, problem.path) for problem in checked.problems], checked.text()


def test_image_folder_check_names_every_broken_param(tmp_path):
    data = {"use": "image_folder", "folder": str(tmp_path / "nowhere"), "labels": "yes", "extensions": [],
            "batch": {"size": 0}, "transform": ["to_tensr", {"use": "normalize", "mean": 0.5, "std": 0.5, "scale": 2}]}
    found, text = problems_of(config_around(data))
    assert ("split_required", "data.split") in found
    assert ("unknown_key", "data.transform[1]") in found
    for name in ("folder", "batch", "labels", "extensions", "transform"):
        assert ("data_value", f"data.{name}") in found, (name, text)
    assert "'to_tensr' is not a registered name" in text and "not found" in text


def test_text_lines_check_names_every_broken_param(tmp_path):
    data = {"use": "text_lines", "file": str(tmp_path / "nowhere.txt"), "seq_len": 0, "stride": -3, "batch": True,
            "tokenizer": "nosuch.Tok", "split": [0.9, 0.1]}
    found, text = problems_of(config_around(data))
    assert ("data_file_missing", "data.file") in found
    for name in ("seq_len", "stride", "batch", "tokenizer"):
        assert ("data_value", f"data.{name}") in found, (name, text)
    assert "seq_len must be a positive integer, got 0" in text


def test_table_stream_options_are_checked(tmp_path):
    file = tmp_path / "rows.parquet"
    pandas.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}).to_parquet(file, index=False)
    data = {"use": "table", "file": str(file), "features": {"x*": "standard_scaler"}, "targets": {"y": None},
            "split": {"use": "sequential", "ratios": [0.8, 0.2]}, "stream": {"chunk": 0}, "batch": 2}
    found, text = problems_of(config_around(data))
    assert found == [("data_value", "data.stream")], text
    assert "stream.chunk must be a positive integer, got 0" in text


def test_table_stream_needs_an_ordered_split(tmp_path):
    file = tmp_path / "rows.parquet"
    pandas.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}).to_parquet(file, index=False)
    data = {"use": "table", "file": str(file), "features": {"x*": "standard_scaler"}, "targets": {"y": None},
            "split": [0.8, 0.1, 0.1], "stream": True, "batch": 2}
    found, text = problems_of(config_around(data))
    assert found == [("data_value", "data.split")], text
    assert "a stream needs a sequential or given split; stream cuts the file in order" in text
    data["split"] = {"use": "kfold", "folds": 2, "fold": 0}
    found, text = problems_of(config_around(data))
    assert found == [("data_value", "data.split")], text
    data["split"] = {"use": "sequential", "ratios": [0.8, 0.2]}
    assert problems_of(config_around(data))[0] == []


@pytest.mark.parametrize("spec, expected", [
    ([{"use": "normalize", "mean": "imagenett"}], "normalize: unknown preset 'imagenett'"),
    ([{"use": "random_flip", "p": 2}], "random_flip: p is a probability in [0, 1], got 2"),
    (["to_tensor", {"use": "resize", "size": -4}], "resize: size is a positive integer or [height, width], got -4"),
    (["math.pi"], "data.transform[0]: 'math.pi' is a float, not a transform"),
])
def test_transform_build_errors_name_the_transform(spec, expected):
    with pytest.raises(BuildError, match=expected.replace("[", r"\[").replace("(", r"\(").replace(")", r"\)")):
        transforms.pipeline_of(spec, "data.transform")


def test_objective_params_are_signature_checked(tmp_path):
    root = tmp_path / "pets"
    for name in ("a", "b"):
        (root / name).mkdir(parents=True)
        for index in range(3):
            Image.new("L", (8, 8), color=40 * index).save(root / name / f"{index}.png")
    config = {
        "data": {"use": "image_folder", "folder": str(root), "split": [0.6, 0.2, 0.2], "transform": ["to_tensor"],
                 "batch": 2},
        "models": {"g": {"use": "mlp", "hidden": [4], "out": 64, "in_features": 8, "inputs": ["z"]},
                   "d": {"use": "mlp", "hidden": [4], "out": 1, "in_features": 64, "inputs": ["image"]}},
        "losses": {"d": {"use": "wgan_gp_d", "generator": "@models.g", "critic": "@models.d", "latent": 8},
                   "elbo": {"use": "vae_elbo", "w_kl": 1.0}},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.d", "minimizes": "@losses.d"}},
        "train": {"epochs": 1, "predicts": False},
    }
    found, text = problems_of(config)
    assert found == [("missing_key", "losses.elbo")], text
    assert "VaeElbo needs ['recon']" in text
