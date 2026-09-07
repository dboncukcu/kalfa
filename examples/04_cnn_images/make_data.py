"""Data of example 04: data/pets, six classes of small RGB images (a bright patch per class), and data/pets_new."""

from pathlib import Path

from kalfa.synthetic import write_image_folder

HERE = Path(__file__).parent


def main():
    classes = ("bengal", "boxer", "husky", "persian", "pug", "sphynx")
    write_image_folder(HERE / "data" / "pets", classes=classes, per_class=24, size=64, channels=3)
    write_image_folder(HERE / "data" / "pets_new", classes=classes, per_class=2, size=64, channels=3, seed=7)
    print("wrote data/pets (6 classes, 144 images) and data/pets_new (12 images)")


if __name__ == "__main__":
    main()
