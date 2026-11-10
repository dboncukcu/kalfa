from pathlib import Path

import numpy
import pandas


CLASSES = ("cold", "mild", "hot")


def housing_frame(rows=200, seed=0):
    generator = numpy.random.default_rng(seed)
    features = generator.normal(size=(rows, 4))
    price = 200.0 + features @ numpy.array([1.0, 1.5, 2.0, 2.5]) * 10.0 + generator.normal(scale=2.0, size=rows)
    frame = pandas.DataFrame(features, columns=[f"x{position}" for position in range(4)])
    frame["price"] = price
    return frame


def blobs_frame(rows=240, seed=1):
    generator = numpy.random.default_rng(seed)
    label = generator.choice(len(CLASSES), size=rows, p=[0.5, 0.3, 0.2])
    centers = numpy.array([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]])
    points = centers[label] + generator.normal(scale=0.5, size=(rows, 2))
    frame = pandas.DataFrame({"f1": points[:, 0], "f2": points[:, 1]})
    frame["kind"] = numpy.asarray(CLASSES)[label]
    frame["id"] = numpy.arange(1000, 1000 + rows)
    return frame


def write_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
