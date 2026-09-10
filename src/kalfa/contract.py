from dataclasses import dataclass
from pathlib import Path

from cirak.registry import registry
from ruamel.yaml import YAML

from .errors import KalfaError
from .kinds import names_of


@dataclass
class Contract:
    path: Path
    document: dict

    @classmethod
    def default_path(cls):
        return Path(__file__).parent / "contract.yaml"

    @classmethod
    def load(cls, path=None):
        target = Path(path) if path is not None else cls.default_path()
        if not target.is_file():
            raise KalfaError(f"contract {target} does not exist; kalfa contract --write writes the default one")
        document = YAML(typ="safe").load(target.read_text())
        for key in ("wiring", "blocks"):
            if not isinstance((document or {}).get(key), dict):
                raise KalfaError(f"contract {target} needs a {key} mapping; start from kalfa contract --write")
        return cls(target, document)

    @property
    def wiring(self):
        return self.document["wiring"]

    @property
    def blocks(self):
        return self.document["blocks"]

    @property
    def history_prefix(self):
        return dict(self.wiring["history_prefix"])

    @property
    def sets(self):
        return list(self.history_prefix)

    @property
    def run_inputs(self):
        return list(self.wiring["run_inputs"])

    @property
    def plot_bus(self):
        return list(self.wiring["plot_bus"])

    def roles(self):
        return list(names_of(registry.facts(self.wiring["builder"]).get("roles")))

    def template(self):
        return {"blocks": self.blocks}

    def text(self):
        return self.path.read_text()

    def write(self, target):
        Path(target).write_text(self.text())
