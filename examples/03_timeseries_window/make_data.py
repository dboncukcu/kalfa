"""Data of example 03: energy.parquet, three sites of 600 hourly steps with daily and weekly cycles."""

from pathlib import Path

from kalfa.synthetic import write_energy

HERE = Path(__file__).parent


def main():
    write_energy(HERE / "energy.parquet", sites=3, steps=600)
    print("wrote energy.parquet (3 sites, 1800 rows)")


if __name__ == "__main__":
    main()
