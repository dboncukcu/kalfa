from pathlib import Path

from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.samples import Samples


logger = logger_for("data.source")


class ImageFolder:
    fields = ["image", "label"]
    dtypes = {"image": "image", "label": "int64"}
    suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"image folder {root!r} does not exist")
        self.classes = sorted(entry.name for entry in self.root.iterdir() if entry.is_dir())
        self.samples = []
        for position, name in enumerate(self.classes):
            for path in sorted((self.root / name).iterdir()):
                if path.suffix.lower() in self.suffixes:
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


@lego("/source/kalfa/image_folder", returns="df", alias="image_folder",
      header="/lego/kalfa/image_folder_header", samples=True,
      description="Images under root/<class>/ as a Dataset with fields image and label")
def image_folder(path):
    logger.info(f"reading {path}")
    folder = ImageFolder(path)
    logger.info(f"{len(folder)} images in {len(folder.classes)} classes")
    return Samples(folder)


@lego("/source/kalfa/text_lines", returns="df", alias="text_lines", header="/lego/kalfa/text_lines_header",
      samples=True,
      description="The lines of a text file as a Dataset with the field text")
def text_lines(path):
    logger.info(f"reading {path}")
    source = TextLines(path)
    logger.info(f"{len(source)} lines")
    return Samples(source)
