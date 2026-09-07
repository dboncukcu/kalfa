"""Data of example 12: housing.parquet, the table config 01 (the lower layer) trains on."""

from pathlib import Path

from kalfa.synthetic import write_housing

HERE = Path(__file__).parent


def main():
    write_housing(HERE / "housing.parquet")
    print("wrote housing.parquet (2000 rows)")


if __name__ == "__main__":
    main()
