from torch.utils import data


class Dataset(data.Dataset):
    inputs = ()
    targets = ()

    def rows(self):
        raise NotImplementedError

    def labels(self, name):
        raise NotImplementedError


def dataset_size(dataset):
    """The number of items of a dataset, counted by a pass when it has no length."""
    try:
        return len(dataset)
    except TypeError:
        return sum(1 for _ in dataset)


def sized(dataset):
    """The length of a dataset, None for a stream that has to be read to be counted."""
    try:
        return len(dataset)
    except TypeError:
        return None
