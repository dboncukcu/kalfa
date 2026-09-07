"""Data of example 06: data/mnist, ten classes of 28 by 28 grayscale images, and data/mnist_new."""

from pathlib import Path

from kalfa.synthetic import write_image_folder

HERE = Path(__file__).parent


def main():
    classes = tuple(str(digit) for digit in range(10))
    write_image_folder(HERE / "data" / "mnist", classes=classes, per_class=20, size=28)
    write_image_folder(HERE / "data" / "mnist_new", classes=classes, per_class=2, size=28, seed=7)
    print("wrote data/mnist (10 classes, 200 images) and data/mnist_new (20 images)")


if __name__ == "__main__":
    main()
