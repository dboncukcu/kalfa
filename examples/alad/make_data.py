from pathlib import Path

import numpy
import pandas


HERE = Path(__file__).resolve().parent
ROWS = 600
NEW_ROWS = 80
NAMES = [f"x{position}" for position in range(6)] + [f"xx{position}" for position in range(6)]
SCALES = numpy.geomspace(200.0, 6000.0, len(NAMES))


def anomaly_frame(rows, rate, seed):
    generator = numpy.random.default_rng(seed)
    anomaly = generator.random(rows) < rate
    factor = generator.normal(size=(rows, 1))
    loading = numpy.linspace(0.5, 1.5, len(NAMES))
    normal = factor * loading + 0.25 * generator.normal(size=(rows, len(NAMES)))
    odd = 1.2 * generator.normal(size=(rows, len(NAMES))) + generator.choice([-1.5, 1.5], size=(rows, 1))
    values = SCALES * numpy.exp(0.5 * numpy.where(anomaly[:, None], odd, normal))
    values = numpy.where(generator.random(values.shape) < 0.1, -values, values)
    frame = pandas.DataFrame(values, columns=NAMES)
    frame["xx6"] = generator.integers(0, 64, rows)
    frame["xx7"] = generator.normal(scale=1.0e6, size=rows)
    label = anomaly.astype("int64")
    junk = generator.random(rows) < 0.03
    label[junk] = generator.choice([-9, 9], size=int(junk.sum()))
    frame["is_anomaly"] = label
    return frame


def main():
    for name, rows, rate, seed in (("anomaly_data", ROWS, 0.1, 0), ("anomaly_new", NEW_ROWS, 0.25, 1)):
        frame = anomaly_frame(rows, rate, seed)
        path = HERE / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        counts = frame["is_anomaly"].value_counts().sort_index().to_dict()
        print(f"wrote {path} ({rows} rows: x0 to x5, xx0 to xx7 and is_anomaly {counts})")


if __name__ == "__main__":
    main()
