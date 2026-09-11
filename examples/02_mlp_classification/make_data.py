"""Data of example 02: churn.parquet (numbers, two categorical columns, the churned label) and new.parquet."""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).parent


def churn_frame(rows=2000, seed=0):
    generator = numpy.random.default_rng(seed)
    numbers = generator.normal(size=(rows, 4))
    cat_a = generator.choice(["north", "south", "east"], size=rows)
    cat_b = generator.choice(["basic", "plus"], size=rows)
    score = numbers[:, 0] * 1.5 - numbers[:, 1] + (cat_a == "north") * 1.0 + (cat_b == "plus") * -0.8
    probability = 1.0 / (1.0 + numpy.exp(-score))
    churned = numpy.where(generator.uniform(size=rows) < probability, "yes", "no")
    frame = pandas.DataFrame(numbers, columns=[f"num_{position}" for position in range(4)])
    frame["cat_a"] = cat_a
    frame["cat_b"] = cat_b
    frame["churned"] = churned
    return frame


def write_churn(path, rows=2000, seed=0):
    churn_frame(rows, seed).to_parquet(path, index=False)
    return path


def main():
    write_churn(HERE / "churn.parquet")
    write_churn(HERE / "new.parquet", rows=100, seed=9)
    print("wrote churn.parquet (2000 rows) and new.parquet (100 rows)")


if __name__ == "__main__":
    main()
