"""Data of example 10: data/shakespeare.txt, two thousand short generated lines, a ROMEO line every fifth."""

from pathlib import Path

from kalfa.synthetic import write_text

HERE = Path(__file__).parent


def main():
    (HERE / "data").mkdir(exist_ok=True)
    write_text(HERE / "data" / "shakespeare.txt", lines=2000)
    print("wrote data/shakespeare.txt (2000 lines)")


if __name__ == "__main__":
    main()
