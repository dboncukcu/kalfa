import json
from pathlib import Path
from types import SimpleNamespace

import numpy
import pandas
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from kalfa.core import aliases, predict
from kalfa.core.errors import BuildError, RunError
from kalfa.core.rng import Rng
from kalfa.std.data import images, transforms


CLASSES = ("cat", "dog", "owl")
SPLIT = [0.5, 0.25, 0.25]


def write_image(path, value, gray=False, size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = numpy.zeros((size, size) if gray else (size, size, 3), dtype="uint8")
    array[:, : size // 2] = value
    Image.fromarray(array).save(path)
    return path


def image_tree(root, counts=(12, 12, 12), gray=False):
    for position, (name, count) in enumerate(zip(CLASSES, counts)):
        for index in range(count):
            write_image(root / name / f"{index:02d}.png", 40 + 60 * position + index, gray)
    return root


def make_ctx(seed=7, directories=()):
    return SimpleNamespace(seed=seed, rng=Rng(0), config=SimpleNamespace(directories=list(directories)))


def fitted(folder, ctx=None, **params):
    settings = {"split": SPLIT, "batch": 4, **params}
    data = images.ImageFolder(str(folder), **settings)
    data.fit(make_ctx() if ctx is None else ctx)
    return data


@pytest.fixture
def folder(tmp_path):
    return image_tree(tmp_path / "images")


def test_the_aliases_are_registered():
    assert aliases.resolve("image_folder") == "kalfa.std.data.images.ImageFolder"
    for name, target in (("to_tensor", "ToTensor"), ("normalize", "Normalize"), ("two_views", "TwoViews"),
                         ("resize", "Resize"), ("center_crop", "CenterCrop"), ("random_crop", "RandomCrop"),
                         ("random_flip", "RandomFlip"), ("grayscale", "Grayscale")):
        assert aliases.resolve(name) == f"kalfa.std.data.transforms.{target}"


def test_split_sizes_and_the_label_tensor(folder):
    data = fitted(folder)
    assert data.counts == {"train": 18, "valid": 9, "test": 9}
    assert data.split.seed == 7
    rows = {name: set(data.rows(name).tolist()) for name in data.counts}
    assert rows["train"].isdisjoint(rows["valid"]) and rows["train"].isdisjoint(rows["test"])
    assert rows["valid"].isdisjoint(rows["test"])
    assert rows["train"] | rows["valid"] | rows["test"] == set(range(36))
    batch = next(iter(data.loaders()["valid"]))
    assert batch["image"].shape == (4, 3, 8, 8)
    assert batch["image"].dtype == torch.float32
    assert 0.0 <= float(batch["image"].min()) and float(batch["image"].max()) <= 1.0
    assert batch["label"].dtype == torch.int64 and batch["label"].shape == (4,)
    assert data.classes == list(CLASSES) and data.n_classes == 3 and data.labels is True
    assert data.target_names == ["label"]
    assert data.is_label("label") and not data.is_label("image")
    assert data.mode == "RGB"


def test_loaders_shuffle_the_train_set_with_the_loader_stream(folder):
    data = fitted(folder, batch={"size": 5, "eval_size": 9})
    loaders = data.loaders()
    assert isinstance(loaders["train"].sampler, RandomSampler) and loaders["train"].batch_size == 5
    assert loaders["train"].generator is data.rng.stream("loader")
    assert isinstance(loaders["valid"].sampler, SequentialSampler) and loaders["valid"].batch_size == 9
    assert sum(len(batch["label"]) for batch in loaders["train"]) == 18
    assert images.ImageFolder(str(folder), split=SPLIT, batch={"workers": 0}).batch["size"] == 64


def test_class_weights_are_inverse_frequency_with_mean_one(tmp_path):
    root = image_tree(tmp_path / "images", counts=(12, 8, 4))
    data = fitted(root, split=[1.0])
    expected = numpy.array([1 / 12, 1 / 8, 1 / 4])
    expected = expected / expected.mean()
    assert data.class_weights.dtype == torch.float32
    numpy.testing.assert_allclose(data.class_weights.numpy(), expected, rtol=1e-6)
    shuffled = fitted(root)
    counts = numpy.bincount(shuffled.frames["train"]["code"].to_numpy(), minlength=3)
    weights = 1.0 / numpy.maximum(counts, 1)
    numpy.testing.assert_allclose(shuffled.class_weights.numpy(), weights / weights.mean(), rtol=1e-6)
    assert float(shuffled.class_weights.mean()) == pytest.approx(1.0)
    assert bool(torch.isfinite(shuffled.class_weights).all())


def test_image_shape_follows_the_transforms(folder, tmp_path):
    assert fitted(folder).image_shape == (3, 8, 8)
    small = fitted(folder, transform=[{"use": "resize", "size": 4}, "to_tensor"])
    assert small.image_shape == (3, 4, 4)
    assert next(iter(small.loaders()["test"]))["image"].shape == (4, 3, 4, 4)
    assert fitted(folder, transform=["grayscale", "to_tensor"]).image_shape == (1, 8, 8)
    gray = fitted(image_tree(tmp_path / "gray", gray=True))
    assert gray.mode == "L" and gray.image_shape == (1, 8, 8)
    assert next(iter(gray.loaders()["train"]))["image"].shape == (4, 1, 8, 8)


def test_train_transform_applies_to_the_train_set_only(folder):
    data = fitted(folder, transform=["to_tensor"], train_transform=[{"use": "random_flip", "p": 1.0}, "to_tensor"])
    plain = transforms.ToTensor()
    for name in ("train", "valid"):
        dataset = data.datasets[name]
        path = data.frames[name]["path"].iloc[0]
        expected = plain(images.open_image(path, "RGB"))
        if name == "train":
            expected = torch.flip(expected, dims=(-1,))
        assert torch.equal(dataset[0]["image"], expected)
    report = data.report()
    assert report["transform"] == ["to_tensor"] and report["train_transform"] == ["random_flip", "to_tensor"]
    alone = fitted(folder, transform=["to_tensor"])
    assert alone.train_transform is alone.transform


def test_two_views_stacks_two_views(folder):
    torch.manual_seed(0)
    each = [{"use": "random_crop", "size": 6, "padding": 1}, "to_tensor"]
    data = fitted(folder, transform=[{"use": "two_views", "each": each}])
    assert data.image_shape == (2, 3, 6, 6)
    batch = next(iter(data.loaders()["test"]))
    assert batch["image"].shape == (4, 2, 3, 6, 6) and batch["label"].shape == (4,)
    image = images.open_image(data.frames["test"]["path"].iloc[0], "RGB")
    same = transforms.TwoViews(["to_tensor"])(image)
    assert same.shape == (2, 3, 8, 8) and torch.equal(same[0], same[1])
    flips = transforms.TwoViews([{"use": "random_flip", "p": 1.0}])(image)
    assert torch.equal(flips[0], torch.flip(same[0], dims=(-1,)))


def test_normalize_presets_numbers_and_lists():
    image = torch.full((3, 2, 2), 0.5)
    for preset in ("imagenet", "cifar10"):
        mean, std = transforms.PRESETS[preset]
        expected = (image - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        assert torch.allclose(transforms.Normalize(preset)(image), expected)
    assert torch.allclose(transforms.Normalize(0.5, 0.5)(torch.ones(1, 2, 2)), torch.ones(1, 2, 2))
    assert torch.allclose(transforms.Normalize([0.0, 0.5, 1.0], [1.0, 0.5, 0.25])(image),
                          torch.tensor([0.5, 0.0, -2.0]).view(3, 1, 1).expand(3, 2, 2))
    gray = Image.fromarray(numpy.full((4, 4), 255, dtype="uint8"))
    mnist = transforms.Normalize(mean="mnist")(gray)
    assert mnist.shape == (1, 4, 4)
    assert torch.allclose(mnist, torch.full((1, 4, 4), (1.0 - 0.1307) / 0.3081))
    mixed = transforms.Normalize(mean="cifar10", std="imagenet")
    assert mixed.mean == transforms.PRESETS["cifar10"][0] and mixed.std == transforms.PRESETS["imagenet"][1]
    assert transforms.Normalize(mean="imagenet", std=2.0).std == [2.0]
    with pytest.raises(BuildError, match="presets"):
        transforms.Normalize("imagenette")
    with pytest.raises(BuildError):
        transforms.Normalize(0.5, 0.0)
    with pytest.raises(RunError, match="channels"):
        transforms.Normalize("imagenet")(torch.zeros(2, 4, 4))


def test_to_tensor_takes_images_arrays_and_tensors():
    convert = transforms.ToTensor()
    white = convert(Image.fromarray(numpy.full((4, 6, 3), 255, dtype="uint8")))
    assert white.shape == (3, 4, 6) and white.dtype == torch.float32 and torch.all(white == 1.0)
    assert convert(Image.fromarray(numpy.zeros((4, 6), dtype="uint8"))).shape == (1, 4, 6)
    array = numpy.arange(24, dtype="uint8").reshape(2, 4, 3)
    out = convert(array)
    assert out.shape == (3, 2, 4) and out[0, 0, 1] == pytest.approx(3 / 255)
    floats = convert(numpy.full((2, 2), 0.25))
    assert floats.shape == (1, 2, 2) and floats.dtype == torch.float32 and torch.all(floats == 0.25)
    tensor = torch.rand(3, 2, 2)
    assert convert(tensor) is tensor
    with pytest.raises(RunError):
        convert("not an image")


def test_geometric_transforms_take_images_and_tensors():
    image = Image.fromarray(numpy.zeros((8, 10, 3), dtype="uint8"))
    assert transforms.Resize(4)(image).size == (4, 4)
    assert transforms.Resize([4, 6])(image).size == (6, 4)
    assert transforms.CenterCrop(6)(image).size == (6, 6)
    assert transforms.RandomCrop(8, padding=2)(image).size == (8, 8)
    assert transforms.Grayscale()(image).mode == "L"
    tensor = torch.rand(3, 8, 10)
    assert transforms.Resize([4, 5])(tensor).shape == (3, 4, 5)
    assert torch.equal(transforms.CenterCrop([4, 4])(tensor), tensor[:, 2:6, 3:7])
    padded = transforms.CenterCrop([10, 12])(tensor)
    assert padded.shape == (3, 10, 12) and torch.equal(padded[:, 1:9, 1:11], tensor)
    assert float(padded[:, 0].abs().sum()) == 0.0
    assert transforms.RandomCrop(6, padding=1)(tensor).shape == (3, 6, 6)
    assert transforms.Grayscale()(tensor).shape == (1, 8, 10)
    assert torch.allclose(transforms.Grayscale()(torch.ones(3, 2, 2)), torch.ones(1, 2, 2))
    with pytest.raises(BuildError):
        transforms.Resize(0)
    with pytest.raises(BuildError):
        transforms.RandomCrop(4, padding=-1)
    with pytest.raises(BuildError):
        transforms.RandomFlip(1.5)


def test_random_flip_draws_from_the_torch_generator():
    array = numpy.zeros((4, 4, 3), dtype="uint8")
    array[:, :2] = 255
    image = Image.fromarray(array)
    assert numpy.array_equal(numpy.array(transforms.RandomFlip(1.0)(image)), array[:, ::-1])
    assert numpy.array_equal(numpy.array(transforms.RandomFlip(0.0)(image)), array)
    flip = transforms.RandomFlip()
    torch.manual_seed(5)
    first = [int(numpy.array(flip(image))[0, 0, 0]) for _ in range(12)]
    torch.manual_seed(5)
    second = [int(numpy.array(flip(image))[0, 0, 0]) for _ in range(12)]
    assert first == second and set(first) == {0, 255}
    assert transforms.RandomFlip(1.0)(torch.arange(4.0).view(1, 1, 4)).flatten().tolist() == [3.0, 2.0, 1.0, 0.0]


def test_pipeline_of_takes_names_import_paths_calls_and_callables():
    def halve(image):
        return image * 0.5

    specs = ["to_tensor", "kalfa.std.data.transforms.Grayscale", {"use": "normalize", "mean": 0.0, "std": 2.0},
             halve]
    pipeline = transforms.pipeline_of(specs, "data.transform")
    out = pipeline(Image.fromarray(numpy.full((2, 2, 3), 255, dtype="uint8")))
    assert out.shape == (1, 2, 2) and torch.allclose(out, torch.full((1, 2, 2), 0.25))
    assert pipeline.names[:3] == ["to_tensor", "kalfa.std.data.transforms.Grayscale", "normalize"]
    built = transforms.pipeline_of([transforms.Resize(2), transforms.ToTensor], "data.transform")
    assert built.names == ["resize", "to_tensor"]
    assert built(Image.fromarray(numpy.zeros((4, 4, 3), dtype="uint8"))).shape == (3, 2, 2)
    assert transforms.pipeline_of(None, "data.transform")("same") == "same"
    with pytest.raises(BuildError, match=r"data.transform\[0\]"):
        transforms.pipeline_of(["no_such_transform"], "data.transform")
    with pytest.raises(BuildError, match="use"):
        transforms.pipeline_of([{"mean": 1.0}], "data.transform")
    with pytest.raises(BuildError):
        transforms.pipeline_of([3], "data.transform")
    with pytest.raises(BuildError, match="could not be built"):
        transforms.pipeline_of(["resize"], "data.transform")


def test_decode_gives_class_names(folder):
    data = fitted(folder)
    assert data.decode("label", torch.tensor([0, 2])).tolist() == ["cat", "owl"]
    scores = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.0, 0.1]])
    assert data.decode("label", scores).tolist() == ["dog", "cat"]
    assert data.decode("label", numpy.array([[1], [2]])).tolist() == ["dog", "owl"]
    with pytest.raises(BuildError):
        data.decode("label", [5])
    with pytest.raises(BuildError):
        data.decode("image", [0])
    with pytest.raises(BuildError):
        data.decode("label", torch.zeros(2, 4))


def test_rows_and_spectators_line_up(folder):
    data = fitted(folder)
    for name in ("train", "valid", "test"):
        frame = data.spectators(name)
        assert list(frame.columns) == ["path", "label"]
        assert frame.index.tolist() == data.rows(name).tolist()
        dataset = data.datasets[name]
        for position, (path, label) in enumerate(zip(frame["path"], frame["label"])):
            assert Path(path).parent.name == label
            assert int(dataset[position]["label"]) == data.classes.index(label)
    assert data.spectators("nowhere") is None
    with pytest.raises(BuildError):
        data.rows("nowhere")


def test_the_prediction_table_decodes_labels_and_carries_paths(folder):
    data = fitted(folder)

    def guess(batch, ctx):
        return {"net": torch.nn.functional.one_hot(batch["label"], 3).float()}

    ctx = SimpleNamespace(data=data, losses={}, metrics={}, device=None, batches=0, set_name=None,
                          set_modes=lambda train: None, loaders=data.loaders, step=SimpleNamespace(predict=guess))
    out = predict.predictions_table(ctx, "test")
    assert out["row"].tolist() == data.rows("test").tolist()
    assert out["label"].tolist() == out["pred_net"].tolist()
    assert out["path"].tolist() == data.spectators("test")["path"].tolist()
    assert [Path(path).parent.name for path in out["path"]] == out["label"].tolist()
    assert {f"prob_net_{name}" for name in CLASSES} <= set(out.columns)


def test_save_load_and_loader_for(folder, tmp_path):
    transform = ["to_tensor", {"use": "normalize", "mean": "imagenet"}]
    data = fitted(folder, transform=transform)
    data.save(tmp_path / "run")
    fields = json.loads((tmp_path / "run" / "fitted" / "fields.json").read_text())
    assert fields["kind"] == "image_folder" and fields["classes"] == list(CLASSES) and fields["labels"] is True
    assert fields["image_shape"] == [3, 8, 8]
    assert fields["transform"] == ["to_tensor", "normalize"] and fields["train_transform"] == ["to_tensor", "normalize"]
    fresh = images.ImageFolder(str(folder), split=SPLIT, transform=transform, batch=4).load(tmp_path / "run")
    assert fresh.classes == list(CLASSES) and fresh.n_classes == 3 and fresh.image_shape == (3, 8, 8)
    assert torch.equal(fresh.class_weights, data.class_weights) and fresh.mode == "RGB"
    with pytest.raises(BuildError):
        fresh.loaders()
    paths = sorted(str(path) for path in (folder / "dog").glob("*.png"))[:5]
    loader = fresh.loader_for(paths)
    batches = list(loader)
    assert sum(len(batch["image"]) for batch in batches) == 5 and "label" not in batches[0]
    assert loader.dataset.rows.tolist() == list(range(5))
    expected = data.datasets["train"].pipeline(images.open_image(paths[0], "RGB"))
    assert torch.allclose(batches[0]["image"][0], expected)
    framed = fresh.loader_for(pandas.DataFrame({"path": paths, "label": ["dog"] * 5}))
    assert torch.cat([batch["label"] for batch in framed]).tolist() == [1] * 5
    whole = fresh.loader_for(str(folder))
    assert len(whole.dataset) == 36 and whole.dataset.codes is not None
    assert len(fresh.loader_for(str(folder / "cat")).dataset) == 12
    with pytest.raises(BuildError, match="missing"):
        fresh.loader_for([str(folder / "missing.png")])
    with pytest.raises(BuildError):
        fresh.loader_for(pandas.DataFrame({"file": paths}))
    with pytest.raises(BuildError):
        images.ImageFolder(str(folder), split=SPLIT).loader_for(paths)
    with pytest.raises(BuildError):
        images.ImageFolder(str(folder), split=SPLIT).load(tmp_path / "elsewhere")


def test_rebuild_gives_the_same_sets(folder, tmp_path):
    data = fitted(folder)
    data.save(tmp_path / "run")
    fresh = images.ImageFolder(str(folder), split=SPLIT, batch=4).load(tmp_path / "run")
    with pytest.raises(BuildError):
        fresh.rows("train")
    fresh.rebuild(make_ctx())
    for name in data.counts:
        assert fresh.rows(name).tolist() == data.rows(name).tolist()
    assert fresh.spectators("test")["path"].tolist() == data.spectators("test")["path"].tolist()
    assert torch.equal(next(iter(fresh.loaders()["test"]))["label"], next(iter(data.loaders()["test"]))["label"])
    with pytest.raises(BuildError):
        images.ImageFolder(str(folder), split=SPLIT).rebuild(make_ctx())


def test_a_flat_folder_has_no_labels(tmp_path):
    root = tmp_path / "flat"
    for index in range(10):
        write_image(root / f"{index}.png", 20 * index)
    data = fitted(root, split=[0.8, 0.2])
    assert data.labels is False and data.classes is None and data.n_classes is None and data.class_weights is None
    assert data.target_names == [] and not data.is_label("label")
    assert data.counts == {"train": 8, "valid": 2}
    assert set(next(iter(data.loaders()["train"]))) == {"image"}
    assert list(data.spectators("valid").columns) == ["path"]
    with pytest.raises(BuildError):
        data.decode("label", [0])
    data.save(tmp_path / "run")
    fresh = images.ImageFolder(str(root), split=[0.8, 0.2]).load(tmp_path / "run")
    assert fresh.labels is False and fresh.classes is None and fresh.image_shape == (3, 8, 8)


def test_labels_false_keeps_the_folder_names_as_spectators(folder):
    data = fitted(folder, labels=False)
    assert data.labels is False and data.classes is None
    assert set(next(iter(data.loaders()["valid"]))) == {"image"}
    assert list(data.spectators("valid").columns) == ["path", "label"]


def test_the_folder_is_found_next_to_the_config(tmp_path):
    image_tree(tmp_path / "kalfa_test_pictures", counts=(4, 4, 4))
    data = fitted("kalfa_test_pictures", ctx=make_ctx(directories=[tmp_path]))
    assert data.counts == {"train": 6, "valid": 3, "test": 3}
    problems = []
    images.ImageFolder.check({"folder": "kalfa_test_pictures", "split": SPLIT}, problems, [str(tmp_path)])
    assert problems == []
    with pytest.raises(BuildError, match="not found"):
        fitted("kalfa_test_pictures")


def test_check_hints_at_close_transform_names(folder):
    problems = []
    images.ImageFolder.check({"folder": str(folder), "split": SPLIT, "transform": ["to_tensr"]}, problems, [])
    assert [(problem.code, problem.path) for problem in problems] == [("data_value", "data.transform")]
    assert problems[0].message.startswith("[0]: 'to_tensr' is not a registered name")
    assert "did you mean to_tensor" in problems[0].message
    with pytest.raises(BuildError, match=r"data.transform\[1\]: 'normalise' is not a registered name.*did you mean "
                                         r"normalize"):
        transforms.pipeline_of(["to_tensor", "normalise"], "data.transform")
    with pytest.raises(BuildError, match=r"data.transform\[0\]: no.such.module.Thing: No module named 'no'"):
        transforms.pipeline_of(["no.such.module.Thing"], "data.transform")


def test_check_reports_a_missing_folder_split_and_batch(folder, tmp_path):
    problems = []
    images.ImageFolder.check({"folder": str(tmp_path / "nowhere"), "batch": {"size": 0},
                              "transform": ["no_such_transform"]}, problems, [])
    found = {(problem.code, problem.path) for problem in problems}
    assert ("data_value", "data.folder") in found
    assert ("split_required", "data.split") in found
    assert ("data_value", "data.batch") in found
    assert ("data_value", "data.transform") in found
    clean = []
    images.ImageFolder.check({"folder": str(folder), "split": SPLIT, "batch": 8, "transform": ["to_tensor"]}, clean,
                             [])
    assert clean == []
    with pytest.raises(BuildError, match="split"):
        images.ImageFolder(str(folder))
    with pytest.raises(BuildError):
        images.ImageFolder(str(folder), split=SPLIT, labels="yes")


def test_an_empty_folder_is_an_error(tmp_path):
    (tmp_path / "empty" / "cat").mkdir(parents=True)
    with pytest.raises(BuildError, match="no images"):
        fitted(tmp_path / "empty")


def test_a_folder_named_from_the_working_directory_gives_absolute_paths(tmp_path, monkeypatch):
    image_tree(tmp_path / "pets", counts=(4, 4, 4))
    monkeypatch.chdir(tmp_path)
    data = fitted("pets")
    assert data.root.is_absolute() and data.root.resolve() == (tmp_path / "pets").resolve()
    for name in data.counts:
        paths = data.spectators(name)["path"].tolist()
        assert paths and all(Path(path).is_absolute() and Path(path).is_file() for path in paths)
    listed = data.listing_of("pets")["path"].tolist() + data.listing_of("pets/cat/00.png")["path"].tolist()
    assert len(listed) == 13 and all(Path(path).is_absolute() for path in listed)
    (tmp_path / "link.png").symlink_to(tmp_path / "pets" / "dog" / "01.png")
    [single, link] = data.listing_of(["pets/dog/01.png", "link.png"])["path"].tolist()
    assert Path(single).is_absolute() and Path(single).resolve() == (tmp_path / "pets/dog/01.png").resolve()
    assert Path(link).is_absolute() and Path(link).name == "link.png"
