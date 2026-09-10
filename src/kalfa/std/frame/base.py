import pickle
from pathlib import Path

import pandas


class FrameTransform:
    def fit(self, df) -> None:
        raise NotImplementedError

    def apply(self, df):
        raise NotImplementedError


def table_only(df, what):
    if not isinstance(df, pandas.DataFrame):
        raise ValueError(f"{what} fits on a table in memory; a stream or a Dataset source has no frame to fit on")
    return df


def write_frames(frames, record):
    target = Path(record) / "fitted" / "frames"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "frames.pkl").open("wb") as stream:
        pickle.dump(list(frames), stream)


def read_frames(record):
    path = Path(record) / "fitted" / "frames" / "frames.pkl"
    if not path.exists():
        return []
    with path.open("rb") as stream:
        return list(pickle.load(stream))
