import shutil
from pathlib import Path

import numpy
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SIZE = 16
COUNT = 640
SHAPES = ("disc", "square", "triangle", "bar")


def shape_image(shape, generator):
    background = generator.integers(0, 70, (SIZE, SIZE, 3)).astype("uint8")
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    color = tuple(int(value) for value in generator.integers(120, 256, 3))
    left, top = (int(value) for value in generator.integers(1, 6, 2))
    extent = int(generator.integers(7, 11))
    right, bottom = left + extent, top + extent
    if shape == "disc":
        draw.ellipse((left, top, right, bottom), fill=color)
    elif shape == "square":
        draw.rectangle((left, top, right, bottom), fill=color)
    elif shape == "triangle":
        draw.polygon([(left, bottom), (left + extent // 2, top), (right, bottom)], fill=color)
    else:
        draw.rectangle((left, top + extent // 3, right, top + 2 * extent // 3), fill=color)
    return image


def main():
    root = HERE / "data" / "unlabeled"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    generator = numpy.random.default_rng(0)
    for index in range(COUNT):
        shape = SHAPES[int(generator.integers(len(SHAPES)))]
        shape_image(shape, generator).save(root / f"{index:03d}.png")
    print(f"wrote data/unlabeled ({COUNT} unlabeled images of {SIZE}x{SIZE} in one flat folder)")


if __name__ == "__main__":
    main()
