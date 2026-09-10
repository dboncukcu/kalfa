from pathlib import Path
import pandas


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


class ImageFolder:
    fields = ["image", "label"]
    dtypes = {"image": "image", "label": "int64"}

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"image folder {root!r} does not exist")
        self.classes = sorted(entry.name for entry in self.root.iterdir() if entry.is_dir())
        self.samples = []
        for position, name in enumerate(self.classes):
            for path in sorted((self.root / name).iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, position))
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, position):
        from PIL import Image

        path, label = self.samples[position]
        with Image.open(path) as image:
            image.load()
            return {"image": image.copy(), "label": label}

    def column(self, name):
        if name == "label":
            return list(self.labels)
        raise ValueError(f"field {name!r} is not readable as a column; only label is")


class TextLines:
    fields = ["text"]
    dtypes = {"text": "string"}

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"text file {path!r} does not exist")
        self.lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, position):
        return {"text": self.lines[position]}

    def column(self, name):
        if name == "text":
            return list(self.lines)
        raise ValueError(f"field {name!r} is not readable as a column; only text is")
