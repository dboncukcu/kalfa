from pathlib import Path

import pandas
from cirak.registry import registry

from kalfa.std.common.files import read_json


def manifest_of(path):
    manifest = read_json(Path(path) / "manifest.json")
    if manifest is None or manifest.get("kind") != "data":
        raise ValueError(f"{path} is no prepared directory; kalfa prepare writes one")
    return manifest


def prepared(path):
    manifest = manifest_of(path)
    folder = Path(path)
    if manifest.get("layout") == "samples":
        source = manifest["source"]
        return registry.resolve(source["uri"])(**(source.get("params") or {}))
    parts = []
    for name in manifest["sets"]:
        frame = pandas.read_parquet(folder / f"{name}.parquet")
        frame.index = frame.pop("row")
        parts.append(frame.assign(kalfa_set=name))
    return pandas.concat(parts) if parts else pandas.DataFrame()
