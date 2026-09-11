"""Data of example 03: energy.parquet, three sites of 600 hourly steps with daily and weekly cycles."""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).parent


def energy_frame(sites=3, steps=600, seed=0):
    generator = numpy.random.default_rng(seed)
    pieces = []
    for site in range(sites):
        time = numpy.arange(steps)
        x0 = numpy.sin(2 * numpy.pi * time / 24.0) + generator.normal(scale=0.1, size=steps)
        x1 = numpy.cos(2 * numpy.pi * time / 168.0) + generator.normal(scale=0.1, size=steps)
        x2 = generator.normal(size=steps)
        load = 100.0 + 30.0 * x0 + 15.0 * x1 + 5.0 * x2 + 10.0 * site + generator.normal(scale=1.0, size=steps)
        pieces.append(pandas.DataFrame({"site_id": f"site_{site}", "x0": x0, "x1": x1, "x2": x2, "load": load}))
    return pandas.concat(pieces, ignore_index=True)


def write_energy(path, sites=3, steps=600, seed=0):
    energy_frame(sites, steps, seed).to_parquet(path, index=False)
    return path


def main():
    write_energy(HERE / "energy.parquet", sites=3, steps=600)
    print("wrote energy.parquet (3 sites, 1800 rows)")


if __name__ == "__main__":
    main()
