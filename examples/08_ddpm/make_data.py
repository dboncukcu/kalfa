from pathlib import Path

import numpy
from PIL import Image


HERE = Path(__file__).resolve().parent
SIZE = 16
COUNT = 960


def mask(kind, generator):
    rows, columns = numpy.mgrid[0:SIZE, 0:SIZE]
    period = int(generator.integers(2, 5))
    if kind == 0:
        return ((columns + rows) // period) % 2
    if kind == 1:
        return (rows // period + columns // period) % 2
    center = generator.uniform(4.0, 12.0, 2)
    return (numpy.hypot(rows - center[0], columns - center[1]) // period).astype(int) % 2


def patch(index, generator):
    first, second = generator.integers(0, 256, (2, 3))
    chosen = mask(index % 3, generator)[:, :, None]
    pixels = numpy.where(chosen == 1, first, second) + generator.normal(scale=6.0, size=(SIZE, SIZE, 3))
    return Image.fromarray(numpy.clip(pixels, 0, 255).astype("uint8"))


def main():
    generator = numpy.random.default_rng(0)
    root = HERE / "data" / "patches"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(COUNT):
        patch(index, generator).save(root / f"{index:03d}.png")
    print(f"wrote data/patches ({COUNT} images of stripes, checks and rings)")


if __name__ == "__main__":
    main()
