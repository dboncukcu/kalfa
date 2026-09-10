import copy
from io import StringIO
from pathlib import Path

import numpy
import pandas
import torch
from cirak.build import Graph, GraphNode
from ruamel.yaml import YAML
from torch import nn

from kalfa.std.builder.kalfa.module import Module
from kalfa.std.pre.base import TableFrame

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def example(name):
    return str(EXAMPLES / name / "config.yaml")


def examples():
    return sorted(path.name for path in EXAMPLES.iterdir() if (path / "config.yaml").is_file())


MINIMAL = YAML(typ="safe").load((EXAMPLES / "minimal" / "config.yaml").read_text())
MINIMAL["params"]["epochs"] = 1


def minimal():
    return copy.deepcopy(MINIMAL)


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
    return Module(linear_graph(in_features, out_features, lazy), seed=seed, name=f"model{index}", index=index,
                  **kwargs)


def frame(rows=16, features=3, seed=0, set_name="train", target="price"):
    generator = numpy.random.default_rng(seed)
    values = generator.normal(size=(rows, features)).astype("float32")
    data = pandas.DataFrame(values, columns=[f"x{i}" for i in range(features)])
    data[target] = (values.sum(axis=1) * 2.0).astype("float32")
    return TableFrame(list(data.columns[:features]), {target: [target]}, set_name, data=data)


def batch(rows=8, features=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, features, generator=generator)
    return {"x": x, "price": x.sum(dim=1) * 2.0}
