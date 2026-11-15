import shutil
from pathlib import Path

import numpy
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SIZE = 16
CLASSES = ("ring", "cross", "stripes", "dot")
PER_CLASS = 30


def object_image(name, generator):
    background = generator.integers(0, 80, (SIZE, SIZE, 3)).astype("uint8")
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    color = tuple(int(value) for value in generator.integers(140, 256, 3))
    left, top = (int(value) for value in generator.integers(1, 5, 2))
    extent = int(generator.integers(9, 12))
    right, bottom = left + extent, top + extent
    middle_x, middle_y = (left + right) // 2, (top + bottom) // 2
    if name == "ring":
        draw.ellipse((left, top, right, bottom), outline=color, width=2)
    elif name == "cross":
        draw.line((left, middle_y, right, middle_y), fill=color, width=2)
        draw.line((middle_x, top, middle_x, bottom), fill=color, width=2)
    elif name == "stripes":
        for row in range(top, bottom + 1, 3):
            draw.line((left, row, right, row), fill=color, width=1)
    else:
        radius = int(generator.integers(2, 4))
        draw.ellipse((middle_x - radius, middle_y - radius, middle_x + radius, middle_y + radius), fill=color)
    return image


def main():
    root = HERE / "data" / "objects"
    if root.exists():
        shutil.rmtree(root)
    generator = numpy.random.default_rng(0)
    for name in CLASSES:
        directory = root / name
        directory.mkdir(parents=True)
        for index in range(PER_CLASS):
            object_image(name, generator).save(directory / f"{index:03d}.png")
    print(f"wrote data/objects ({PER_CLASS * len(CLASSES)} images of {SIZE}x{SIZE} in {len(CLASSES)} classes)")


if __name__ == "__main__":
    main()
