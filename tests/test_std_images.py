"""The Dataset pipeline of the image configs: image_folder, samples, per item chains, the balanced sampler."""

import numpy
import pytest
import torch

import kalfa  # noqa: F401
from kalfa.std.data import filter_set
from kalfa.std.feed import SampleDataset, table
from kalfa.std.layer import unflatten
from kalfa.std.loader import balanced_sampler, torch as torch_loader
from kalfa.std.plot import image_pairs
from kalfa.std.pre import (apply, fit, normalize, random_crop_flip, resize, standard_scaler, to_tensor,
                           to_tensor_signed)
from kalfa.std.samples import Samples
from kalfa.std.source import header, image_folder
from kalfa.std.split import kfold, random as random_split
from kalfa.synthetic import write_image_folder


@pytest.fixture
def folder(tmp_path):
    return write_image_folder(tmp_path / "images", classes=("zero", "one", "two"), per_class=8, size=16)


def test_image_folder_source_and_header(folder):
    samples = image_folder(str(folder))
    assert isinstance(samples, Samples) and len(samples) == 24
    assert samples.fields == ["image", "label"] and samples.dtypes == {"image": "image", "label": "int64"}
    item = samples[0]
    assert item["image"].size == (16, 16) and item["label"] == 0
    assert samples.column("label").tolist() == [0] * 8 + [1] * 8 + [2] * 8
    head = header("/source/kalfa/image_folder", {"path": str(folder)})
    assert head == {"columns": ["image", "label"], "dtypes": {"image": "image", "label": "int64"}, "rows": 24,
                    "classes": ["one", "two", "zero"]}
    with pytest.raises(FileNotFoundError):
        image_folder(str(folder / "nowhere"))


def test_samples_query_subset_and_splits(folder):
    samples = image_folder(str(folder))
    ones = samples.query("label == 1")
    assert len(ones) == 8 and set(ones.column("label")) == {1}
    assert len(samples.query("label != 1")) == 16 and len(samples.query("label in [0, 2]")) == 16
    with pytest.raises(ValueError, match="equality"):
        samples.query("label > 1")
    parts = random_split(samples, [0.5, 0.25, 0.25], seed=1)
    assert [len(parts[name]) for name in ("train", "valid", "test")] == [12, 6, 6]
    assert set(parts["train"].index) | set(parts["valid"].index) | set(parts["test"].index) == set(range(24))
    folds = kfold(samples, k=4, fold=1, val=0.5, seed=1)
    assert [len(folds[name]) for name in ("train", "valid", "test")] == [9, 9, 6]
    filtered = filter_set(samples, "train", [{"query": "label == 0", "sets": ["train"]}])
    assert len(filtered) == 8 and len(filter_set(samples, "valid", [{"query": "label == 0", "sets": ["train"]}])) == 24


def test_fit_apply_and_sample_dataset_in_dataset_mode(folder, tmp_path):
    samples = image_folder(str(folder))
    prep = fit(samples, {"image": {"preprocessors": ["a", "t"]}, "label": {"target": True}},
               {"t": to_tensor(), "a": random_crop_flip(16)}, [], keys={"a": {"sets": ["train"]}},
               record=str(tmp_path / "rec"))
    assert prep.dtypes == {"image": "float32", "label": "int64"} and prep.targets == {"label": ["label"]}
    frame = apply(samples, prep, "train", {"a": {"sets": ["train"]}})
    assert frame.dataset is samples and frame.fields == ["image", "label"] and len(frame.chains["image"]) == 2
    valid = apply(samples, prep, "valid", {"a": {"sets": ["train"]}})
    assert len(valid.chains["image"]) == 1
    dataset = table(frame)
    assert isinstance(dataset, SampleDataset) and dataset.inputs == ["image"] and dataset.targets == ["label"]
    item = dataset[0]
    assert item["image"].shape == (1, 16, 16) and item["image"].dtype == torch.float32
    assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    assert item["label"].dtype == torch.int64 and dataset.labels("label").tolist() == samples.column("label").tolist()
    loader = torch_loader(dataset, "train", {"size": 5})
    batch = next(iter(loader))
    assert batch["image"].shape == (5, 1, 16, 16) and batch["label"].shape == (5,)
    with pytest.raises(ValueError, match="need a table"):
        fit(samples, {"im*": {"preprocessors": ["t"]}}, {"t": to_tensor()}, [])
    with pytest.raises(ValueError, match="cannot be read as one"):
        fit(samples, {"image": {"preprocessors": ["s"]}}, {"s": standard_scaler()}, [])
    with pytest.raises(TypeError, match="to_tensor"):
        fit(samples, {"image": {}}, {}, [])


def test_image_preprocessors(folder):
    samples = image_folder(str(folder))
    image = samples[0]["image"]
    signed = to_tensor_signed().apply(image)
    assert float(signed.min()) >= -1.0 and float(signed.max()) <= 1.0 and signed.shape == (1, 16, 16)
    assert resize(8).apply(image).size == (8, 8) and resize([8, 12]).apply(image).size == (12, 8)
    cropped = random_crop_flip(16).apply(image)
    assert cropped.size == (16, 16)
    tensor = to_tensor().apply(image)
    scaled = normalize(0.5, 0.5).apply(tensor)
    assert torch.allclose(scaled, (tensor - 0.5) / 0.5)
    rgb = torch.rand(3, 4, 4)
    imagenet = normalize("imagenet", "imagenet").apply(rgb)
    assert imagenet.shape == (3, 4, 4) and float(imagenet[0, 0, 0]) == pytest.approx((float(rgb[0, 0, 0]) - 0.485) / 0.229)


def test_balanced_sampler_and_unflatten(folder):
    samples = image_folder(str(folder)).query("label != 2")
    skewed = samples.subset([0, 1, 2, 3, 4, 5, 8])
    prep = fit(skewed, {"image": {"preprocessors": ["t"]}, "label": {"target": True}}, {"t": to_tensor()}, [])
    dataset = table(apply(skewed, prep, "train"))
    torch.manual_seed(0)
    sampler = balanced_sampler(dataset)
    drawn = dataset.labels("label")[torch.as_tensor(list(sampler))].tolist()
    assert drawn.count(1) > 1
    loader = torch_loader(dataset, "train", {"size": 7, "balanced": True})
    assert loader.sampler is not None and len(next(iter(loader))["label"]) == 7
    assert unflatten([1, 4, 4])(torch.zeros(2, 16)).shape == (2, 1, 4, 4)


def test_image_pairs_draws_from_the_valid_set_when_test_is_empty(folder, tmp_path):
    from torch import nn

    samples = image_folder(str(folder))
    prep = fit(samples, {"image": {"preprocessors": ["t"]}}, {"t": to_tensor()}, [])
    loader = torch_loader(table(apply(samples, prep, "valid")), "valid", {"size": 4})
    empty = torch_loader(table(apply(samples.subset([]), prep, "test")), "test", {"size": 4})

    class Same(nn.Module):
        inputs = ["image"]
        outputs = ["x_hat"]

        def forward(self, value):
            return value

    image_pairs(None, [], {"ae": Same()}, str(tmp_path), loaders={"valid": loader, "test": empty}, predicts="ae", n=3)
    assert (tmp_path / "plots" / "image_pairs.png").exists()
    assert image_pairs(None, [], {"ae": Same()}, str(tmp_path), loaders={"test": empty}, predicts="ae") is None


def test_two_views_and_simclr_aug(folder):
    from kalfa.std.objective import ntxent
    from kalfa.std.pre import simclr_aug, two_views
    from torch import nn

    samples = image_folder(str(folder))
    image = samples[0]["image"]
    pair = two_views(simclr_aug(16)).apply(image)
    assert len(pair) == 2 and pair[0].size == (16, 16)
    stacked = to_tensor().apply(pair)
    assert stacked.shape == (2, 1, 16, 16)

    class Net(nn.Module):
        inputs = ["image"]
        outputs = ["z"]

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(16 * 16, 8)

        def forward(self, value):
            return self.layer(value.flatten(1))

    torch.manual_seed(0)
    loss = ntxent({"net": Net()}, {"image": torch.rand(4, 2, 1, 16, 16)}, "net", temperature=0.5)
    assert loss.requires_grad and float(loss.detach()) > 0.0
    with pytest.raises(ValueError, match="two views"):
        ntxent({"net": Net()}, {"image": torch.rand(4, 1, 16, 16)}, "net")


def test_text_source_tokenizer_and_next_token(tmp_path):
    from kalfa.std.feed import next_token
    from kalfa.std.metric import perplexity
    from kalfa.std.pre import char_tokenizer
    from kalfa.std.source import header, text_lines
    from kalfa.synthetic import write_text

    path = write_text(tmp_path / "corpus.txt", lines=12)
    samples = text_lines(str(path))
    assert len(samples) == 12 and samples.fields == ["text"] and header("/source/kalfa/text_lines", {"path": str(path)})["rows"] == 12
    prep = fit(samples, {"text": {"preprocessors": ["tok"]}}, {"tok": char_tokenizer()}, [], record=str(tmp_path / "rec"))
    tokenizer = prep.tokenizer()
    assert tokenizer is not None and "\n" in tokenizer.chars and prep.dtypes == {"text": "int64"}
    ids = tokenizer.encode("ROMEO")
    assert tokenizer.decode(ids) == "ROMEO"
    frame = apply(samples, prep, "train")
    dataset = next_token(frame, None, seq_len=8)
    assert len(dataset) > 0 and dataset[0]["input_ids"].shape == (8,) and dataset[0]["targets"].shape == (8,)
    assert torch.equal(dataset[0]["targets"][:-1], dataset[0]["input_ids"][1:])
    metric = perplexity()
    metric.update(torch.zeros(2, 3, 5), torch.zeros(2, 3, dtype=torch.long))
    assert metric.compute() == pytest.approx(5.0)
    assert (tmp_path / "rec" / "preprocessors" / "tok.pkl").exists()
