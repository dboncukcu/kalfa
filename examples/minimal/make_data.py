"""Data of the minimal example: housing.parquet (x0..x7 and price) and new.parquet to predict on."""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).parent


def housing_frame(rows=2000, seed=0, columns=8):
    generator = numpy.random.default_rng(seed)
    features = generator.normal(size=(rows, columns))
    weights = numpy.linspace(1.0, 2.0, columns)
    price = 200.0 + features @ weights * 10.0 + generator.normal(scale=2.0, size=rows)
    frame = pandas.DataFrame(features, columns=[f"x{position}" for position in range(columns)])
    frame["price"] = price
    return frame


def write_housing(path, rows=2000, seed=0):
    housing_frame(rows, seed).to_parquet(path, index=False)
    return path


def main():
    write_housing(HERE / "housing.parquet")
    write_housing(HERE / "new.parquet", rows=100, seed=5)
    print("wrote housing.parquet (2000 rows) and new.parquet (100 rows)")


if __name__ == "__main__":
    main()
