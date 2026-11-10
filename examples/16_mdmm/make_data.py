from pathlib import Path

import numpy
import pandas


HERE = Path(__file__).resolve().parent
ROWS = 200


def housing_frame(rows=ROWS, seed=0):
    generator = numpy.random.default_rng(seed)
    features = generator.normal(size=(rows, 4))
    weights = numpy.array([1.0, 1.5, 2.0, 2.5])
    price = 200.0 + features @ weights * 10.0 + generator.normal(scale=2.0, size=rows)
    frame = pandas.DataFrame(features, columns=[f"x{position}" for position in range(4)])
    frame["price"] = price
    return frame


def main():
    path = HERE / "housing.parquet"
    housing_frame().to_parquet(path, index=False)
    print(f"wrote {path} ({ROWS} rows: x0 to x3 and price)")


if __name__ == "__main__":
    main()
