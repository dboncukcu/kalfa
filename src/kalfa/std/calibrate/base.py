import json
import pickle
from pathlib import Path


class Calibration:
    def fit(self, models, loaders, prep, device, predicts) -> None:
        raise NotImplementedError

    def apply(self, table):
        return table

    def note(self) -> dict:
        return {}


def write_calibrations(items, record):
    target = Path(record) / "fitted" / "calibrate"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "calibrations.pkl").open("wb") as stream:
        pickle.dump(dict(items), stream)
    (target / "calibrate.json").write_text(json.dumps({name: item.note() for name, item in items.items()}, indent=2,
                                                      default=float))


def read_calibrations(record):
    path = Path(record) / "fitted" / "calibrate" / "calibrations.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        return dict(pickle.load(stream))
