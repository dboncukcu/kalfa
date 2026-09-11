"""Data of example 04: data/pets, six classes of small RGB images (a bright patch per class), and data/pets_new."""

from pathlib import Path

import numpy

HERE = Path(__file__).parent


def write_image_folder(root, classes=("zero", "one"), per_class=24, size=28, channels=1, seed=0):
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


def main():
    classes = ("bengal", "boxer", "husky", "persian", "pug", "sphynx")
    write_image_folder(HERE / "data" / "pets", classes=classes, per_class=24, size=64, channels=3)
    write_image_folder(HERE / "data" / "pets_new", classes=classes, per_class=2, size=64, channels=3, seed=7)
    print("wrote data/pets (6 classes, 144 images) and data/pets_new (12 images)")


if __name__ == "__main__":
    main()
