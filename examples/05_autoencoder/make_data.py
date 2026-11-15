from pathlib import Path

import numpy
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SIZE = 16
SEGMENTS = {"zero": "abcdef", "one": "bc", "four": "bcfg", "seven": "abc"}


def corners(generator):
    left = 4 + int(generator.integers(-1, 2))
    top = 2 + int(generator.integers(-1, 2))
    width = 6 + int(generator.integers(0, 2))
    height = 5 + int(generator.integers(0, 2))
    return left, top, left + width, top + height, top + 2 * height


def segment_lines(left, top, right, middle, bottom):
    return {
        "a": [(left, top), (right, top)],
        "b": [(right, top), (right, middle)],
        "c": [(right, middle), (right, bottom)],
        "d": [(left, bottom), (right, bottom)],
        "e": [(left, middle), (left, bottom)],
        "f": [(left, top), (left, middle)],
        "g": [(left, middle), (right, middle)],
    }


def glyph(segments, generator):
    image = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(image)
    lines = segment_lines(*corners(generator))
    shade = int(generator.integers(170, 256))
    for name in segments:
        draw.line(lines[name], fill=shade, width=int(generator.integers(1, 3)))
    noise = generator.normal(scale=12.0, size=(SIZE, SIZE))
    return Image.fromarray(numpy.clip(numpy.asarray(image, dtype=float) + noise, 0, 255).astype("uint8"))


def write_digits(root, count, seed):
    generator = numpy.random.default_rng(seed)
    for name, segments in SEGMENTS.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            glyph(segments, generator).save(directory / f"{index:03d}.png")
    return len(SEGMENTS) * count


def main():
    written = write_digits(HERE / "data" / "digits", 32, seed=0)
    fresh = write_digits(HERE / "data" / "digits_new", 3, seed=1)
    print(f"wrote data/digits ({written} images of {', '.join(SEGMENTS)}) and data/digits_new ({fresh} images)")


if __name__ == "__main__":
    main()
