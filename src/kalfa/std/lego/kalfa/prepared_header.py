from pathlib import Path

from kalfa.registration import lego
from kalfa.std.common.files import read_json


@lego("/lego/kalfa/prepared_header",
      description="The header a prepared directory recorded in its manifest: the columns, the dtypes and the rows")
def prepared_header(path):
    manifest = read_json(Path(path) / "manifest.json")
    if manifest is None or manifest.get("kind") != "data":
        raise ValueError(f"{path} is no prepared directory; kalfa prepare writes one")
    return dict(manifest["header"])
