import shutil
from pathlib import Path

import numpy
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SIZE = 16
CLASSES = {"cat": (230, 140, 40), "dog": (150, 90, 50), "owl": (190, 190, 210)}
PER_CLASS = 24
NEW_PER_CLASS = 4


def pet_image(name, generator):
    background = generator.integers(0, 60, (SIZE, SIZE, 3)).astype("uint8")
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    shade = numpy.clip(numpy.array(CLASSES[name]) + generator.integers(-30, 31, 3), 0, 255)
    color = tuple(int(value) for value in shade)
    left, top = (int(value) for value in generator.integers(1, 6, 2))
    extent = int(generator.integers(8, 11))
    right, bottom = left + extent, top + extent
    if name == "cat":
        draw.polygon([(left, top), (left + extent // 2, bottom), (right, top)], fill=color)
    elif name == "dog":
        draw.rectangle((left, top, right, bottom), fill=color)
    else:
        draw.ellipse((left, top, right, bottom), fill=color)
    return image


def write_folder(root, per_class, seed):
    if root.exists():
        shutil.rmtree(root)
    generator = numpy.random.default_rng(seed)
    for name in CLASSES:
        directory = root / name
        directory.mkdir(parents=True)
        for index in range(per_class):
            pet_image(name, generator).save(directory / f"{index:03d}.png")
    return per_class * len(CLASSES)


def main():
    count = write_folder(HERE / "data" / "pets", PER_CLASS, seed=0)
    fresh = write_folder(HERE / "data" / "pets_new", NEW_PER_CLASS, seed=1)
    print(f"wrote data/pets ({count} images of {SIZE}x{SIZE} in {len(CLASSES)} classes) and data/pets_new "
          f"({fresh} images)")


if __name__ == "__main__":
    main()
