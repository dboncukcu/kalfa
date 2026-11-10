from pathlib import Path

import numpy
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SIZE = 16
CANVAS = 64
MOODS = ("smile", "frown", "surprise")


def color(generator, low, high):
    return tuple(int(value) for value in generator.integers(low, high, 3))


def mouth(draw, mood, center, generator):
    width = int(generator.integers(9, 14))
    left, right = center - width, center + width
    if mood == "smile":
        draw.arc([left, 30, right, 46], start=20, end=160, fill=(120, 20, 30), width=4)
    elif mood == "frown":
        draw.arc([left, 40, right, 56], start=200, end=340, fill=(120, 20, 30), width=4)
    else:
        draw.ellipse([center - 5, 38, center + 5, 50], outline=(120, 20, 30), width=4)


def face(mood, generator):
    image = Image.new("RGB", (CANVAS, CANVAS), color(generator, 40, 200))
    draw = ImageDraw.Draw(image)
    center = 32 + int(generator.integers(-3, 4))
    skin = (int(generator.integers(170, 250)), int(generator.integers(130, 200)), int(generator.integers(90, 160)))
    draw.ellipse([center - 24, 6, center + 24, 60], fill=skin)
    for side in (-10, 10):
        draw.ellipse([center + side - 4, 20, center + side + 4, 28], fill=(30, 30, 40))
    mouth(draw, mood, center, generator)
    return image.resize((SIZE, SIZE), Image.LANCZOS)


def write_faces(root, count, seed):
    generator = numpy.random.default_rng(seed)
    for mood in MOODS:
        directory = root / mood
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            face(mood, generator).save(directory / f"{index:03d}.png")
    return len(MOODS) * count


def main():
    written = write_faces(HERE / "data" / "faces", 300, seed=0)
    print(f"wrote data/faces ({written} images of {', '.join(MOODS)})")


if __name__ == "__main__":
    main()
