"""Small builders shared by the tests: a minimal valid config, tiny models and frames."""

import copy
from io import StringIO

import numpy
import pandas
import torch
from cirak.build import Graph, GraphNode
from ruamel.yaml import YAML
from torch import nn

from kalfa.std.builder import Module
from kalfa.std.pre import Frame

MINIMAL = {
    "include": ["/alias/kalfa/tabular"],
    "seed": 1,
    "data": {
        "source": {"uri": "parquet", "params": {"path": "housing.parquet"}},
        "split": {"ratios": [0.7, 0.15, 0.15], "seed": 1},
        "batch": 64,
        "preprocessors": {"scale": {"uri": "standard_scaler"}},
        "fields": {"x*": {"preprocessors": ["scale"]}, "price": {"target": True}},
        "feed": "table",
    },
    "model": {
        "optimizer": {"uri": "adam", "params": {"lr": 0.01}},
        "inputs": ["x"],
        "outputs": ["y"],
        "nodes": [{"uri": "linear", "params": {"out_features": 1}}],
    },
    "metrics": {"rmse": {"uri": "rmse"}},
    "losses": {"mse": {"uri": "mse"}},
    "training": {"turn": "supervised", "loss": "mse", "epochs": 1, "checkpoint": "last", "report": "last"},
    "plots": {"loss_curve": {"uri": "loss_curve"}},
    "record": "runs/t_$datetime$",
}


def minimal(**changes):
    """A deep copy of the minimal config with dotted overrides applied (``training.epochs=2``)."""
    config = copy.deepcopy(MINIMAL)
    for path, value in changes.items():
        parts = path.split("__")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        if value is None and parts[-1] in target and path.endswith("__DELETE"):
            continue
        target[parts[-1]] = value
    return config


def delete(config, *path):
    target = config
    for part in path[:-1]:
        target = target[part]
    del target[path[-1]]
    return config


def yaml_text(data):
    yaml = YAML()
    stream = StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def write_config(path, data):
    path.write_text(yaml_text(data))
    return str(path)


def linear_graph(in_features=3, out_features=1, lazy=False):
    layer = nn.LazyLinear(out_features) if lazy else nn.Linear(in_features, out_features)
    return Graph(("x",), ("y",), (GraphNode("layer", layer, ("x",), ("y",)),))


def tiny_model(in_features=3, out_features=1, seed=1, index=0, lazy=False, **kwargs):
    return Module(linear_graph(in_features, out_features, lazy), seed=seed, index=index, **kwargs)


def frame(rows=16, features=3, seed=0, set_name="train", target="price"):
    generator = numpy.random.default_rng(seed)
    values = generator.normal(size=(rows, features)).astype("float32")
    data = pandas.DataFrame(values, columns=[f"x{i}" for i in range(features)])
    data[target] = (values.sum(axis=1) * 2.0).astype("float32")
    return Frame(data, list(data.columns[:features]), {target: [target]}, set_name)


def batch(rows=8, features=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, features, generator=generator)
    return {"x": x, "price": x.sum(dim=1) * 2.0}
