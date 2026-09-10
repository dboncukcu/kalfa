import numpy
import torch

from kalfa.std.pre.base import Frame


class Dataset(torch.utils.data.Dataset):
    inputs: list = ()
    targets: list = ()
    frame: Frame | None = None

    def rows(self) -> numpy.ndarray:
        raise NotImplementedError

    def labels(self, name: str) -> torch.Tensor:
        raise ValueError(f"{type(self).__name__} cannot count its labels; balanced and class_weights need a table "
                         f"dataset")

    def size(self) -> int | None:
        return len(self)

    def count(self) -> int:
        return len(self)


class IterableDataset(torch.utils.data.IterableDataset, Dataset):
    shuffle = False
    buffer = 4096

    def size(self) -> int | None:
        return None

    def count(self) -> int:
        return sum(1 for _ in self)
