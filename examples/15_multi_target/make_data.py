"""Data of example 15: scores.parquet (2000 rows, x0..x7, the analysis scores y_a, y_b, y_c, their combination z
and the tail label z_tail)."""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).parent


def scores_frame(rows=2000, seed=0, analyses=3, tail=1.5):
    generator = numpy.random.default_rng(seed)
    features = generator.normal(size=(rows, 8))
    weights = numpy.linspace(0.4, 1.6, 8 * analyses).reshape(8, analyses)
    scores = features @ weights + generator.normal(scale=0.3, size=(rows, analyses))
    signed = numpy.sign(scores) * scores ** 2
    total = signed.sum(axis=1)
    combined = numpy.sign(total) * numpy.sqrt(numpy.abs(total))
    frame = pandas.DataFrame(features, columns=[f"x{position}" for position in range(8)])
    for position, name in zip(range(analyses), "abcdefgh"):
        frame[f"y_{name}"] = scores[:, position]
    frame["z"] = combined
    frame["z_tail"] = (combined > tail).astype("float32")
    return frame


def write_scores(path, rows=2000, seed=0):
    scores_frame(rows, seed).to_parquet(path, index=False)
    return path


def main():
    write_scores(HERE / "scores.parquet")
    print("wrote scores.parquet (2000 rows)")


if __name__ == "__main__":
    main()
