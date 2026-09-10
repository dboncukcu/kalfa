import json
from pathlib import Path

from kalfa.registration import lego
from kalfa.std.common.samples import is_samples
from kalfa.std.source.kalfa.prepared import manifest_of
from kalfa.std.split.base import report_sets


@lego("/split/kalfa/prepared", returns=["train", "valid", "test"], sizes="/lego/kalfa/prepared_sizes",
      description="The split kalfa prepare recorded: the sets a prepared frame is marked with, or the positions "
                  "of a Dataset source's items per set")
def prepared_split(df, path):
    manifest = manifest_of(path)
    folder = Path(path)
    if is_samples(df):
        parts = {name: df.subset(json.loads((folder / f"{name}.json").read_text())) for name in manifest["sets"]}
    else:
        parts = {name: df[df["kalfa_set"] == name].drop(columns=["kalfa_set"]) for name in manifest["sets"]}
    return report_sets("prepared", parts)
