from pathlib import Path

import pandas
from cirak.registry import registry

from kalfa.registration import lego
from kalfa.std.common.files import read_json


def manifest_of(path):
    manifest = read_json(Path(path) / "manifest.json")
    if manifest is None or manifest.get("kind") != "data":
        raise ValueError(f"{path} is no prepared directory; kalfa prepare writes one")
    return manifest


@lego("/source/kalfa/prepared", returns="df", header="/lego/kalfa/prepared_header",
      description="The data kalfa prepare wrote: the sets of a table read back into one frame marked by set, or "
                  "the items of a Dataset source read from where they are with the split kept as positions")
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
