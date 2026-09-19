from pathlib import Path

import numpy
import pandas


def reference_frame(rows=2000, seed=0):
    generator = numpy.random.default_rng(seed)
    raw = generator.normal(size=(rows, 4))
    region = generator.choice(["north", "south", "east", "west"], size=rows)
    tier = generator.choice(["low", "mid", "high"], size=rows, p=[0.5, 0.3, 0.2])
    site_index = generator.integers(0, 8, size=rows)
    site_effect = numpy.linspace(-1.0, 1.0, 8)[site_index]
    late = generator.normal(size=rows)
    late[generator.uniform(size=rows) < 0.08] = numpy.nan
    gap = generator.normal(size=rows)
    gap[generator.uniform(size=rows) < 0.05] = numpy.nan
    signal = 1.5 * raw[:, 2] - raw[:, 3] + (tier == "high")
    y_lin = (3.0 * raw[:, 0] - 2.0 * raw[:, 1] + 0.5 * (region == "north") + 0.3 * site_effect
             + generator.normal(scale=0.5, size=rows))
    y_quad = raw[:, 0] ** 2 + raw[:, 0] * raw[:, 1] - raw[:, 2] + generator.normal(scale=0.3, size=rows)
    return pandas.DataFrame({
        "sample_id": 1000 + numpy.arange(rows),
        "raw_0": raw[:, 0],
        "raw_1": raw[:, 1],
        "raw_2": raw[:, 2],
        "raw_3": raw[:, 3],
        "heavy": 2.0 * generator.standard_t(2, size=rows),
        "frac": generator.uniform(0.01, 0.99, size=rows),
        "late": late,
        "gap": gap,
        "count": generator.poisson(3, size=rows),
        "region": region,
        "tier": tier,
        "site": numpy.array([f"s{position}" for position in range(8)])[site_index],
        "junk": generator.choice(["a", "b"], size=rows),
        "noise_id": generator.integers(0, 1000, size=rows),
        "y_lin": y_lin,
        "y_quad": y_quad,
        "y_heavy": 2.0 * raw[:, 3] + generator.standard_t(2, size=rows),
        "y_frac": 1.0 / (1.0 + numpy.exp(-(signal + generator.normal(scale=0.3, size=rows)))),
        "is_hot": (signal + generator.normal(scale=0.7, size=rows) > 0.8).astype("int64"),
    })


def write_reference(path, rows=2000, seed=0):
    reference_frame(rows, seed).to_parquet(path, index=False)
    return path


def housing_frame(rows=2000, seed=0, columns=8):
    generator = numpy.random.default_rng(seed)
    features = generator.normal(size=(rows, columns))
    weights = numpy.linspace(1.0, 2.0, columns)
    price = 200.0 + features @ weights * 10.0 + generator.normal(scale=2.0, size=rows)
    frame = pandas.DataFrame(features, columns=[f"x{position}" for position in range(columns)])
    frame["price"] = price
    return frame


def write_housing(path, rows=2000, seed=0):
    frame = housing_frame(rows, seed)
    if str(path).endswith(".csv"):
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
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


def write_images(root, counts, size=8, channels=1, seed=0):
    from PIL import Image

    generator = numpy.random.default_rng(seed)
    root = Path(root)
    for position, (name, count) in enumerate(counts.items()):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            canvas = generator.integers(0, 40, size=(size, size, channels), dtype="uint8")
            half = size // 2
            row = 0 if position % 2 == 0 else half
            column = 0 if position < 2 else half
            canvas[row:row + half, column:column + half] = 200 + generator.integers(0, 50, size=(half, half, channels))
            image = Image.fromarray(canvas[:, :, 0] if channels == 1 else canvas, mode="L" if channels == 1 else "RGB")
            image.save(directory / f"{index:03d}.png")
    return root


def write_image_folder(root, classes=("zero", "one"), per_class=24, size=28, channels=1, seed=0):
    return write_images(root, {name: per_class for name in classes}, size, channels, seed)


def write_text(path, lines=200, seed=0):
    generator = numpy.random.default_rng(seed)
    words = ["love", "night", "sun", "moon", "sword", "fair", "verona", "friend", "death", "light"]
    rows = []
    for position in range(lines):
        count = int(generator.integers(3, 8))
        text = " ".join(generator.choice(words, size=count))
        rows.append(f"ROMEO: {text}" if position % 5 == 0 else text)
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_all(root):
    root = Path(root)
    write_reference(root / "reference.parquet")
    write_reference(root / "new.parquet", rows=100, seed=5)
    write_housing(root / "housing.parquet")
    write_housing(root / "housing.csv")
    write_housing(root / "new.csv", rows=50, seed=9)
    write_churn(root / "churn.parquet")
    write_energy(root / "energy.parquet", sites=3, steps=120)
    write_images(root / "images", {"a": 40, "b": 24}, size=8)
    write_text(root / "text.txt")
    return root
