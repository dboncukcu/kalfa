import contextlib
import functools
import os
import shutil
from io import StringIO
from pathlib import Path

import numpy
import pandas
import pytest
import torch
from cirak.build import Graph, GraphNode
from cirak.registry import registry
from ruamel.yaml import YAML
from torch import nn

from kalfa.std.builder.kalfa.module import Module
from kalfa.std.pre.base import TableFrame


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "tests" / "configs"
EXAMPLES = ROOT / "examples"


def config_path(name):
    return str(CONFIGS / f"{name}.yaml")


def load_config(name):
    return YAML(typ="safe").load((CONFIGS / f"{name}.yaml").read_text())


def yaml_text(data):
    yaml = YAML()
    stream = StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def write_config(path, data):
    Path(path).write_text(yaml_text(data))
    return str(path)


def examples():
    return sorted(path.name for path in EXAMPLES.iterdir() if (path / "config.yaml").is_file())


def example(name):
    return str(EXAMPLES / name / "config.yaml")


def needs(name):
    return pytest.importorskip(name)


@contextlib.contextmanager
def legacy_onnx():
    with pytest.warns(DeprecationWarning) as caught:
        yield
    assert any("legacy TorchScript-based ONNX export" in str(item.message) for item in caught)


@contextlib.contextmanager
def inside(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield Path(path)
    finally:
        os.chdir(previous)


def build(uri, **params):
    target = registry.resolve(uri)
    if registry.facts(uri).partial:
        return functools.partial(target, **params) if params else target
    return target(**params)


def linear_graph(in_features=3, out_features=1, lazy=False):
    layer = nn.LazyLinear(out_features) if lazy else nn.Linear(in_features, out_features)
    return Graph(("x",), ("y",), (GraphNode("layer", layer, ("x",), ("y",)),))


def tiny_model(in_features=3, out_features=1, seed=1, index=0, lazy=False, **kwargs):
    return Module(linear_graph(in_features, out_features, lazy), seed=seed, name=f"model{index}", index=index,
                  **kwargs)


def frame(rows=16, features=3, seed=0, set_name="train", target="price"):
    generator = numpy.random.default_rng(seed)
    values = generator.normal(size=(rows, features)).astype("float32")
    data = pandas.DataFrame(values, columns=[f"x{position}" for position in range(features)])
    data[target] = (values.sum(axis=1) * 2.0).astype("float32")
    return TableFrame(list(data.columns[:features]), {target: [target]}, set_name, data=data)


def batch(rows=8, features=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, features, generator=generator)
    return {"x": x, "price": x.sum(dim=1) * 2.0}


def reference_sets(seed=11):
    from data import reference_frame

    frame = reference_frame().rename(columns={f"raw_{position}": f"num_{position}" for position in range(4)})
    kept = frame[frame["num_1"] > -2.5]
    return registry.resolve("/split/kalfa/random")(kept, [0.7, 0.15, 0.15], seed)


def copied(record, target):
    shutil.copytree(record, target)
    return Path(target)
