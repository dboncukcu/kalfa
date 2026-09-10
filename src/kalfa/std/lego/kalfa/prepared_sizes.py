from kalfa.registration import lego
from kalfa.std.source.kalfa.prepared import manifest_of


@lego("/lego/kalfa/prepared_sizes",
      description="The set sizes a prepared directory recorded in its manifest")
def prepared_sizes(rows, path):
    return dict(manifest_of(path)["sizes"])
