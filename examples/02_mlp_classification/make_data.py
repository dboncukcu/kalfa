"""Data of example 02: churn.parquet (numbers, two categorical columns, the churned label) and new.parquet."""

from pathlib import Path

from kalfa.synthetic import write_churn

HERE = Path(__file__).parent


def main():
    write_churn(HERE / "churn.parquet")
    write_churn(HERE / "new.parquet", rows=100, seed=9)
    print("wrote churn.parquet (2000 rows) and new.parquet (100 rows)")


if __name__ == "__main__":
    main()
