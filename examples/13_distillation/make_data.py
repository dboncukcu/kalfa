"""Data of example 13: data/cifar10 (ten classes of 32 by 32 RGB images) for the teacher and the student, and data/cifar10_new."""

from pathlib import Path

from kalfa.synthetic import write_image_folder

HERE = Path(__file__).parent


def main():
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    write_image_folder(HERE / "data" / "cifar10", classes=classes, per_class=24, size=32, channels=3)
    write_image_folder(HERE / "data" / "cifar10_new", classes=classes, per_class=2, size=32, channels=3, seed=7)
    print("wrote data/cifar10 (10 classes, 240 images) and data/cifar10_new (20 images)")


if __name__ == "__main__":
    main()
