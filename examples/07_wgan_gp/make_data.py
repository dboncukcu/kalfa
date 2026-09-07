"""Data of example 07: data/celeba64, ten classes of 32 by 32 RGB images (the generator produces 32 by 32)."""

from pathlib import Path

from kalfa.synthetic import write_image_folder

HERE = Path(__file__).parent


def main():
    classes = tuple(f"attr_{position}" for position in range(10))
    write_image_folder(HERE / "data" / "celeba64", classes=classes, per_class=48, size=32, channels=3)
    print("wrote data/celeba64 (10 classes, 480 images)")


if __name__ == "__main__":
    main()
