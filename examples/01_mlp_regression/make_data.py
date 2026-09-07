"""Data of example 01: housing.parquet (2000 rows, x0..x7 and price) and new.parquet to predict on."""

from pathlib import Path

from kalfa.synthetic import write_housing

HERE = Path(__file__).parent


def main():
    write_housing(HERE / "housing.parquet")
    write_housing(HERE / "new.parquet", rows=100, seed=5)
    print("wrote housing.parquet (2000 rows) and new.parquet (100 rows)")


if __name__ == "__main__":
    main()
