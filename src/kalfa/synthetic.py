"""Synthetic data for the examples and the tests: small tables, image folders and a text corpus that every
reference config can train on in a minute on a laptop CPU."""

from pathlib import Path



import numpy
import pandas


def housing_frame(rows=2000, seed=0, columns=8):
    generator = numpy.random.default_rng(seed)
    features = generator.normal(size=(rows, columns))
    weights = numpy.linspace(1.0, 2.0, columns)
    price = 200.0 + features @ weights * 10.0 + generator.normal(scale=2.0, size=rows)
    frame = pandas.DataFrame(features, columns=[f"x{position}" for position in range(columns)])
    frame["price"] = price
    return frame


def write_housing(path, rows=2000, seed=0):
    housing_frame(rows, seed).to_parquet(path, index=False)
    return path


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


def anomaly_frame(rows=2000, seed=0):
    """The ALAD table of examples/alad: x1..x4, xx1, xx2, xx6, xx7 and is_anomaly in 0, 1 and a few 5."""
    generator = numpy.random.default_rng(seed)
    label = generator.choice([0, 1, 5], size=rows, p=[0.88, 0.09, 0.03])
    frame = pandas.DataFrame({f"x{position}": generator.normal(size=rows) for position in range(1, 5)})
    for name in ("xx1", "xx2", "xx6", "xx7"):
        frame[name] = generator.gamma(2.0, 300.0, size=rows)
    shift = (label == 1).astype("float64")
    for name in ("x1", "x2", "xx1"):
        frame[name] = frame[name] + shift * (3.0 if name.startswith("xx") is False else 2000.0)
    frame["is_anomaly"] = label.astype("int64")
    return frame


def write_anomaly(path, rows=2000, seed=0):
    anomaly_frame(rows, seed).to_parquet(path, index=False)
    return path


def write_image_folder(root, classes=("zero", "one"), per_class=24, size=28, channels=1, seed=0):
    """A tiny image folder: root/<class>/<n>.png, one bright blob per class at a class specific corner."""
    from pathlib import Path

    from PIL import Image

    generator = numpy.random.default_rng(seed)
    root = Path(root)
    for position, name in enumerate(classes):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            canvas = generator.integers(0, 40, size=(size, size, channels), dtype="uint8")
            half = size // 2
            row = 0 if position % 2 == 0 else half
            column = 0 if position < 2 else half
            canvas[row:row + half, column:column + half] = 200 + generator.integers(0, 50, size=(half, half, channels))
            image = Image.fromarray(canvas[:, :, 0] if channels == 1 else canvas, mode="L" if channels == 1 else "RGB")
            image.save(directory / f"{index:03d}.png")
    return root


def write_text(path, lines=200, seed=0):
    """A small text corpus: lines of a few words drawn from a tiny vocabulary, a ROMEO line now and then."""
    generator = numpy.random.default_rng(seed)
    words = ["love", "night", "sun", "moon", "sword", "fair", "verona", "friend", "death", "light"]
    rows = []
    for position in range(lines):
        count = int(generator.integers(3, 8))
        text = " ".join(generator.choice(words, size=count))
        rows.append(f"ROMEO: {text}" if position % 5 == 0 else text)
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path

