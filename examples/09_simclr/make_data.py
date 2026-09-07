"""Data of example 09: data/stl10/unlabeled (one folder of RGB images) and data/stl10/test (four classes)."""

from pathlib import Path

from kalfa.synthetic import write_image_folder

HERE = Path(__file__).parent


def main():
    write_image_folder(HERE / "data" / "stl10" / "unlabeled", classes=("all",), per_class=512, size=64, channels=3)
    write_image_folder(HERE / "data" / "stl10" / "test", classes=("bird", "cat", "deer", "dog"), per_class=8, size=64,
                       channels=3, seed=7)
    print("wrote data/stl10/unlabeled (512 images) and data/stl10/test (4 classes, 32 images)")


if __name__ == "__main__":
    main()
