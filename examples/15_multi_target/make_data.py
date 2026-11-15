from pathlib import Path

import numpy
import pandas


HERE = Path(__file__).resolve().parent
ROWS = 400
NEW_ROWS = 20
FEATURES = 6
TARGETS = ("yield", "moisture", "defects")


def plant_frame(rows=ROWS, seed=0):
    generator = numpy.random.default_rng(seed)
    x = generator.normal(size=(rows, FEATURES))
    harvest = 50.0 + 6.0 * x[:, 0] - 4.0 * x[:, 1] + 3.0 * x[:, 2] * x[:, 3] + generator.normal(scale=1.5, size=rows)
    moisture = 20.0 + 2.5 * numpy.tanh(x[:, 4]) + 1.5 * x[:, 0] + generator.normal(scale=0.5, size=rows)
    defects = generator.poisson(numpy.exp(0.4 + 0.5 * x[:, 5] - 0.3 * x[:, 1])).astype("float64")
    frame = pandas.DataFrame(x, columns=[f"x{position}" for position in range(FEATURES)])
    frame["yield"] = harvest
    frame["moisture"] = moisture
    frame["defects"] = defects
    return frame


def main():
    path = HERE / "plant.parquet"
    plant_frame().to_parquet(path, index=False)
    new = HERE / "new.parquet"
    plant_frame(NEW_ROWS, seed=1).drop(columns=list(TARGETS)).to_parquet(new, index=False)
    print(f"wrote {path} ({ROWS} rows: x0 to x5, yield, moisture and defects) and {new} ({NEW_ROWS} rows without "
          f"targets)")


if __name__ == "__main__":
    main()
