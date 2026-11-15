import json
import logging
import os
from pathlib import Path

import numpy
import pandas
import torch
from PIL import Image
from torch.utils.data import Dataset

from kalfa.core.aliases import register
from kalfa.core.errors import BuildError, RunError, error
from kalfa.std.data import flow, transforms
from kalfa.std.data.base import Loaders, config_directories, seeded_split
from kalfa.std.data.columns import name_problems, target_of
from kalfa.std.data.options import batch_of, locate, split_check, split_of


logger = logging.getLogger("kalfa.data")

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
GRAY_MODES = ("1", "L", "LA", "I", "I;16", "F")
BATCH_SIZE = 64
SPLIT_HINT = "a list of ratios such as [0.8, 0.1, 0.1] or a call such as {use: kfold, folds: 5, fold: 0}"


def labels_of(labels):
    if not isinstance(labels, bool):
        raise BuildError(f"labels is true or false, got {labels!r}")
    return labels


def extensions_of(extensions):
    if isinstance(extensions, str):
        extensions = [extensions]
    if not isinstance(extensions, (list, tuple)) or not all(isinstance(item, str) for item in extensions):
        raise BuildError(f"extensions is a list of file suffixes such as ['.png', '.jpg'], got {extensions!r}")
    out = tuple(item.lower() if item.startswith(".") else f".{item.lower()}" for item in extensions if item)
    if not out:
        raise BuildError("extensions names at least one file suffix such as .png")
    return out


def transforms_check(spec):
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    for position, item in enumerate(items):
        if isinstance(item, str):
            target_of(item, f"[{position}]")


VALIDATORS = (
    ("batch", lambda value: batch_of(value, BATCH_SIZE)),
    ("labels", labels_of),
    ("extensions", extensions_of),
    ("split", split_check),
    ("transform", transforms_check),
    ("train_transform", transforms_check),
)


def absolute(path):
    return Path(os.path.abspath(Path(str(path)).expanduser()))


def folder_of(folder, directories=()):
    if not isinstance(folder, (str, Path)) or not str(folder):
        raise BuildError(f"folder is the path of an image folder, got {folder!r}")
    try:
        path = locate(folder, directories)
    except BuildError:
        raise BuildError(f"image folder {str(folder)!r} not found") from None
    if not path.is_dir():
        raise BuildError(f"image folder {str(folder)!r} is a file, not a directory")
    return absolute(path)


def visible(path):
    return not path.name.startswith(".")


def images_in(directory, extensions):
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and visible(path) and path.suffix.lower() in extensions)


def scan(root, extensions):
    found = {}
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and visible(entry):
            files = images_in(entry, extensions)
            if files:
                found[entry.name] = files
    if found:
        loose = images_in(root, extensions)
        if loose:
            logger.warning(f"{root}: {len(loose)} images lie next to the class folders and are left out")
        paths = [str(path) for files in found.values() for path in files]
        names = [name for name, files in found.items() for _ in files]
        return pandas.DataFrame({"path": paths, "label": names}), list(found)
    paths = [str(path) for path in images_in(root, extensions)]
    if not paths:
        raise BuildError(f"image folder {str(root)!r} holds no images with the extensions {list(extensions)}")
    return pandas.DataFrame({"path": paths}), []


def mode_of(path):
    try:
        with Image.open(path) as image:
            return "L" if image.mode in GRAY_MODES else "RGB"
    except OSError as failure:
        raise BuildError(f"{path}: not a readable image ({failure})") from failure


def open_image(path, mode):
    try:
        with Image.open(path) as image:
            return image.convert(mode)
    except OSError as failure:
        raise RunError(f"{path}: not a readable image ({failure})") from failure


class ImageDataset(Dataset):
    def __init__(self, paths, codes, pipeline, mode, rows, frame=None):
        self.paths = list(paths)
        self.codes = None if codes is None else torch.from_numpy(numpy.array(codes, dtype="int64"))
        self.pipeline = pipeline
        self.mode = mode
        self.rows = numpy.array(rows)
        self.frame = frame

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, position):
        image = open_image(self.paths[position], self.mode)
        item = {"image": transforms.image_tensor(self.pipeline(image))}
        if self.codes is not None:
            item["label"] = self.codes[position]
        return item


class ImageFolder(Loaders):
    inputs = ("image",)
    exposes = ("n_classes", "classes", "class_weights", "image_shape")
    predictions = True
    spectator_patterns = ("path", "label")

    @classmethod
    def check(cls, params, problems, directories=()):
        if params.get("split") is None:
            problems.append(error("split_required", f"data.split is required: {SPLIT_HINT}", "data.split"))
        folder = params.get("folder")
        if isinstance(folder, (str, Path)) and str(folder):
            try:
                folder_of(folder, directories)
            except BuildError as failure:
                problems.append(error("data_value", str(failure), "data.folder",
                                      hint="the run looks in the working directory, then next to the config"))
        for name, validator in VALIDATORS:
            if params.get(name) is None:
                continue
            try:
                validator(params[name])
            except BuildError as failure:
                problems.append(error("data_value", str(failure), f"data.{name}"))
        name_problems(params.get("split"), "data.split", problems)

    def __init__(self, folder, split=None, transform=None, train_transform=None, batch=64, labels=True,
                 extensions=EXTENSIONS):
        self.folder = folder
        self.split = split_of(split)
        self.transform = transforms.pipeline_of(transform, "data.transform")
        self.train_transform = (self.transform if train_transform is None
                                else transforms.pipeline_of(train_transform, "data.train_transform"))
        self.batch = batch_of(batch, BATCH_SIZE)
        self.wants_labels = labels_of(labels)
        self.labels = self.wants_labels
        self.extensions = extensions_of(extensions)
        self.classes = None
        self.n_classes = None
        self.class_weights = None
        self.image_shape = None
        self.mode = None
        self.root = None
        self.found = None
        self.frames = {}
        self.datasets = {}
        self.counts = {}
        self.rng = None
        self.loaders_by_set = None
        self.fitted = False

    @property
    def target_names(self):
        return ["label"] if self.labels else []

    def scanned(self, ctx):
        root = folder_of(self.folder, config_directories(ctx))
        frame, classes = scan(root, self.extensions)
        self.root = root
        self.found = len(frame)
        return frame, classes

    def lookup(self):
        return {name: code for code, name in enumerate(self.classes)}

    def coded(self, frame):
        if not self.labels:
            return frame
        lookup = self.lookup()
        unknown = sorted(set(frame["label"]) - set(lookup))
        if unknown:
            raise BuildError(f"{self.root}: the class folders {unknown} are not among the fitted classes "
                             f"{self.classes}")
        return frame.assign(code=frame["label"].map(lookup).astype("int64"))

    def sets_of(self, frame, ctx):
        sets = {name: part for name, part in seeded_split(self.split, ctx).split(frame).items() if len(part)}
        if "train" not in sets:
            raise BuildError("the split leaves no train images")
        return sets

    def weights_of(self, train):
        counts = numpy.bincount(train["code"].to_numpy(dtype="int64"), minlength=len(self.classes))
        weights = 1.0 / numpy.maximum(counts.astype("float64"), 1.0)
        return torch.tensor(weights / weights.mean(), dtype=torch.float32)

    def shape_of(self, train):
        image = open_image(train["path"].iloc[0], self.mode)
        return tuple(int(size) for size in transforms.image_tensor(self.transform(image)).shape)

    def fit(self, ctx):
        frame, classes = self.scanned(ctx)
        self.labels = self.wants_labels and bool(classes)
        if self.wants_labels and not classes:
            logger.info(f"{self.root} has no class folders; the images carry no label")
        self.classes = list(classes) if self.labels else None
        self.n_classes = None if self.classes is None else len(self.classes)
        self.mode = mode_of(frame["path"].iloc[0])
        sets = self.sets_of(self.coded(frame), ctx)
        self.class_weights = self.weights_of(sets["train"]) if self.labels else None
        self.image_shape = self.shape_of(sets["train"])
        self.fitted = True
        self.materialize(sets, ctx)

    def rebuild(self, ctx):
        if not self.fitted:
            raise BuildError("the data is not fitted; load(directory) first")
        frame, classes = self.scanned(ctx)
        if self.labels and not classes:
            raise BuildError(f"{self.root} has no class folders, but the run was fitted on the classes {self.classes}")
        if self.mode is None:
            self.mode = mode_of(frame["path"].iloc[0])
        self.materialize(self.sets_of(self.coded(frame), ctx), ctx)

    def dataset_of(self, frame, pipeline):
        codes = frame["code"].to_numpy() if self.labels and "code" in frame.columns else None
        return ImageDataset(frame["path"].tolist(), codes, pipeline, self.mode, frame.index)

    def materialize(self, sets, ctx):
        self.rng = getattr(ctx, "rng", None)
        self.frames = dict(sets)
        self.datasets = {name: self.dataset_of(part, self.train_transform if name == "train" else self.transform)
                         for name, part in sets.items()}
        self.counts = {name: len(part) for name, part in sets.items()}
        self.loaders_by_set = None
        rows = ", ".join(f"{name} {count}" for name, count in self.counts.items())
        classes = "no labels" if self.classes is None else f"{len(self.classes)} classes"
        logger.info(f"data: {rows} images; {classes}; image shape {self.image_shape}")

    def listing_of(self, frame):
        if isinstance(frame, pandas.DataFrame):
            if "path" not in frame.columns:
                raise BuildError(f"loader_for takes a frame with a path column; the columns are {list(frame.columns)}")
            listing = frame
        elif isinstance(frame, (str, Path)):
            path = absolute(frame)
            if path.is_dir():
                listing, _ = scan(path, self.extensions)
            elif path.is_file():
                listing = pandas.DataFrame({"path": [str(path)]})
            else:
                raise BuildError(f"loader_for: {str(frame)!r} is neither an image folder nor an image file")
        elif isinstance(frame, (list, tuple)) and all(isinstance(item, (str, Path)) for item in frame):
            listing = pandas.DataFrame({"path": [str(absolute(item)) for item in frame]})
        else:
            raise BuildError(f"loader_for takes a frame with a path column, a list of image paths or a folder, got "
                             f"{type(frame).__name__}")
        if not len(listing):
            raise BuildError("loader_for: there are no images to load")
        missing = [str(path) for path in listing["path"] if not Path(str(path)).expanduser().is_file()]
        if missing:
            raise BuildError(f"loader_for: {len(missing)} images are missing, such as {missing[:3]}")
        return listing

    def codes_for(self, listing):
        if not self.labels or "label" not in listing.columns:
            return None
        lookup = self.lookup()
        names = listing["label"].tolist()
        unknown = sorted({str(name) for name in names if name not in lookup})
        if unknown:
            logger.warning(f"loader_for: the labels {unknown[:5]} are not among the classes {self.classes}; the "
                           f"images are loaded without labels")
            return None
        return numpy.array([lookup[name] for name in names], dtype="int64")

    def loader_for(self, frame, set_name="test"):
        if not self.fitted:
            raise BuildError("the data is not fitted; call fit(ctx) or load(directory) first")
        listing = self.listing_of(frame)
        paths = [str(Path(str(path)).expanduser()) for path in listing["path"]]
        mode = self.mode or mode_of(paths[0])
        dataset = ImageDataset(paths, self.codes_for(listing), self.transform, mode, listing.index, frame=listing)
        return self.make_loader(dataset, train=False)

    def spectators(self, set_name):
        part = self.frames.get(set_name)
        if part is None:
            return None
        return part[[column for column in ("path", "label") if column in part.columns]]

    def decode(self, name, values):
        if not self.is_label(name):
            reason = "the images carry no label" if not self.labels else "the only target is label"
            raise BuildError(f"decode({name!r}): {reason}")
        array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else numpy.asarray(values)
        count = len(self.classes)
        if array.ndim >= 2 and array.shape[-1] != 1:
            if array.shape[-1] != count:
                raise BuildError(f"decode({name!r}): takes class indices or scores of width {count}, got the shape "
                                 f"{array.shape}")
            array = array.argmax(axis=-1)
        codes = numpy.asarray(array).astype("int64").reshape(-1)
        if codes.size and (codes.min() < 0 or codes.max() >= count):
            raise BuildError(f"decode({name!r}): class indices lie in [0, {count}), got {codes.min()} to "
                             f"{codes.max()}")
        return numpy.asarray(self.classes, dtype=object)[codes]

    def is_label(self, name):
        return bool(self.labels) and name == "label"

    def rescales(self, name):
        return False

    def inverse(self, name, values):
        return values

    def save(self, directory):
        if not self.fitted:
            raise BuildError("the data is not fitted; there is nothing to save")
        root = Path(directory) / "fitted"
        root.mkdir(parents=True, exist_ok=True)
        fields = {
            "kind": "image_folder",
            "classes": None if self.classes is None else list(self.classes),
            "labels": bool(self.labels),
            "image_shape": None if self.image_shape is None else list(self.image_shape),
            "transform": list(self.transform.names),
            "train_transform": list(self.train_transform.names),
            "mode": self.mode,
            "class_weights": None if self.class_weights is None else self.class_weights.tolist(),
            "targets": list(self.target_names),
            "target_fields": self.target_fields(),
        }
        (root / "fields.json").write_text(json.dumps(fields, indent=2))

    def target_fields(self):
        if not self.labels:
            return []
        weights = None if self.class_weights is None else self.class_weights.tolist()
        return [{"name": "label", "width": 1, "chain": [], "classes": list(self.classes), "class_weights": weights}]

    def load(self, directory):
        file = Path(directory) / "fitted" / "fields.json"
        try:
            fields = json.loads(file.read_text())
        except FileNotFoundError:
            raise BuildError(f"{file.parent} lacks fields.json; save(directory) writes the fitted files") from None
        except json.JSONDecodeError as failure:
            raise BuildError(f"{file} is not valid JSON: {failure}") from failure
        if not isinstance(fields, dict) or fields.get("kind") != "image_folder":
            raise BuildError(f"{file} was not written by image_folder")
        self.labels = bool(fields.get("labels"))
        classes = fields.get("classes")
        self.classes = list(classes) if self.labels and classes is not None else None
        if self.labels and self.classes is None:
            raise BuildError(f"{file} says the images carry labels but lists no classes")
        self.n_classes = None if self.classes is None else len(self.classes)
        weights = fields.get("class_weights")
        self.class_weights = None if weights is None else torch.tensor(weights, dtype=torch.float32)
        shape = fields.get("image_shape")
        self.image_shape = None if shape is None else tuple(int(size) for size in shape)
        self.mode = fields.get("mode") or None
        self.found = None
        self.frames = {}
        self.datasets = {}
        self.counts = {}
        self.loaders_by_set = None
        self.fitted = True
        return self

    def report(self):
        return {"rows": dict(self.counts), "targets": list(self.target_names),
                "classes": None if self.classes is None else list(self.classes), "labels": bool(self.labels),
                "image_shape": None if self.image_shape is None else list(self.image_shape), "mode": self.mode,
                "transform": list(self.transform.names), "train_transform": list(self.train_transform.names)}

    def pipeline(self):
        graph = flow.Flow()
        root = self.root if self.root is not None else self.folder
        images = None if self.found is None else f"{flow.amount(self.found)} images"
        source = flow.source_node(graph, root, self.found, [images])
        rows = source
        classes = None
        if self.labels and self.classes:
            classes = rows = graph.step(source, "classes", "classes", "classes", flow.listed(self.classes),
                                        len(self.classes))
        split = flow.split_node(graph, rows, self.split)
        made = flow.set_nodes(graph, split, self.counts, edges=flow.shares_of(self.split))
        same = self.train_transform is self.transform
        trains = [key for name, key in made.items() if same or name == "train"] or [split]
        evals = [key for name, key in made.items() if name != "train"] or [split]
        ends = graph.chain(trains if same else evals, "transform", flow.pipeline_steps(self.transform), "transform",
                           "transform", "every set" if same else "valid, test and predict")
        if not same:
            ends += graph.chain(trains, "train_transform", flow.pipeline_steps(self.train_transform), "transform",
                                "train_transform", "train set")
        shape = " × ".join(str(size) for size in self.image_shape) if self.image_shape else "C × H × W"
        image = graph.node("tensor:image", "tensor", "image", f"float {shape}")
        graph.join(ends, image)
        tensors = [image]
        if classes is not None:
            about = f"int64, {len(self.classes)} classes"
            tensors.append(graph.step(classes, "tensor:label", "tensor", "label", about))
        flow.batch_node(graph, tensors, self.batch)
        return graph.result()


register("image_folder", "kalfa.std.data.images.ImageFolder", slot="data",
         doc="a folder of class folders of images: the batch carries image (float (C, H, W) after the transforms) and "
             "label (int64); transform for every set, train_transform for the train set")
