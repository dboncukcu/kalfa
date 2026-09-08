"""Data of example 15: scores.parquet (2000 rows, x0..x7, the analysis scores y_a, y_b, y_c, their combination z
and the tail label z_tail)."""

from pathlib import Path

from kalfa.synthetic import write_scores

HERE = Path(__file__).parent


def main():
    write_scores(HERE / "scores.parquet")
    print("wrote scores.parquet (2000 rows)")


if __name__ == "__main__":
    main()
