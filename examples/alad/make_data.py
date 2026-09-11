"""Data of the alad example: anomaly_data.parquet, the table the ALAD objectives train on."""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).parent


def anomaly_frame(rows=2000, seed=0):
    generator = numpy.random.default_rng(seed)
    label = generator.choice([0, 1, 5], size=rows, p=[0.88, 0.09, 0.03])
    frame = pandas.DataFrame({f"x{position}": generator.normal(size=rows) for position in range(1, 5)})
    for name in ("xx1", "xx2", "xx6", "xx7"):
        frame[name] = generator.gamma(2.0, 300.0, size=rows)
    shift = (label == 1).astype("float64")
    for name in ("x1", "x2", "xx1"):
        frame[name] = frame[name] + shift * (3.0 if name.startswith("xx") is False else 2000.0)
    frame["is_anomaly"] = label.astype("int64")
    return frame


def write_anomaly(path, rows=2000, seed=0):
    anomaly_frame(rows, seed).to_parquet(path, index=False)
    return path


def main():
    write_anomaly(HERE / "anomaly_data.parquet")
    print("wrote anomaly_data.parquet (2000 rows, about nine percent anomalies)")


if __name__ == "__main__":
    main()
