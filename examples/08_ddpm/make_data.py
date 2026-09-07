"""Data of example 08: data/cifar10, ten classes of 32 by 32 RGB images."""

from pathlib import Path

from kalfa.synthetic import write_image_folder

HERE = Path(__file__).parent


def main():
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    write_image_folder(HERE / "data" / "cifar10", classes=classes, per_class=16, size=32, channels=3)
    print("wrote data/cifar10 (10 classes, 160 images)")


if __name__ == "__main__":
    main()
