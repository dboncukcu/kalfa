from pathlib import Path

import numpy
import pandas


HERE = Path(__file__).resolve().parent
HOURS = 720
AHEAD = 240
SITES = {"north": (40.0, 12.0, 0.6), "south": (55.0, 18.0, 1.1)}


def site_frame(name, base, swing, cooling, generator, hours):
    clock = numpy.arange(hours)
    hour = clock % 24
    angle = 2.0 * numpy.pi * hour / 24.0
    temperature = (15.0 + 6.0 * numpy.sin(angle - 2.0 * numpy.pi * 9.0 / 24.0) + numpy.sin(clock / 70.0) * 3.0
                   + generator.normal(scale=0.8, size=hours))
    shape = numpy.sin(angle - 2.0 * numpy.pi * 6.0 / 24.0) + 0.5 * numpy.sin(2.0 * angle)
    load = base + swing * shape + cooling * (temperature - 15.0) + generator.normal(scale=1.5, size=hours)
    return pandas.DataFrame({"site_id": name, "x0": numpy.sin(angle), "x1": numpy.cos(angle),
                             "x2": temperature, "load": load})


def frames(seed=3):
    generator = numpy.random.default_rng(seed)
    seen, ahead = [], []
    for name, (base, swing, cooling) in SITES.items():
        frame = site_frame(name, base, swing, cooling, generator, HOURS + AHEAD)
        seen.append(frame.iloc[:HOURS])
        ahead.append(frame.iloc[HOURS:])
    return pandas.concat(seen, ignore_index=True), pandas.concat(ahead, ignore_index=True)


def main():
    seen, ahead = frames()
    seen.to_parquet(HERE / "energy.parquet", index=False)
    ahead.to_parquet(HERE / "energy_next.parquet", index=False)
    print(f"wrote {HERE / 'energy.parquet'} ({len(seen)} hourly rows of {len(SITES)} sites: site_id, x0 to x2, load)")
    print(f"wrote {HERE / 'energy_next.parquet'} (the next {AHEAD} hours of each site)")


if __name__ == "__main__":
    main()
