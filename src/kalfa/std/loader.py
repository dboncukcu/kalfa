"""Loaders: legos that wrap a Dataset into batches."""


from torch.utils.data import DataLoader, IterableDataset

from ..registration import lego
from .log import logger_for

logger = logger_for("data.loader")


def _built(loader, set, size):
    try:
        logger.info(f"{set}: {len(loader)} batches of {size}")
    except TypeError:
        logger.info(f"{set}: a stream in batches of {size}")
    return loader


@lego("/loader/kalfa/torch",
            description="torch DataLoader; shuffles the train set only, eval_size for the other sets; a stream "
                        "dataset shuffles through its buffer and takes no sampler or workers")
def torch(data, set, batch):
    train = set == "train"
    size = int(batch["size"]) if train else int(batch.get("eval_size") or batch["size"])
    shuffle = bool(batch.get("shuffle", True)) if train else False
    drop_last = bool(batch.get("drop_last", False)) if train else False
    if isinstance(data, IterableDataset):
        if batch.get("balanced"):
            raise ValueError("balanced needs a table dataset; a lazy set cannot be counted")
        if int(batch.get("workers", 0) or 0):
            raise ValueError("a lazy set runs with workers: 0; every worker would replay the whole stream")
        if train and hasattr(data, "shuffle"):
            data.shuffle = shuffle
            data.buffer = int(batch.get("buffer", 4096) or 4096)
        return _built(DataLoader(data, batch_size=size, drop_last=drop_last, num_workers=0,
                                 collate_fn=batch.get("collate")), set, size)
    sampler = None
    if batch.get("balanced") and train and len(data):
        sampler = balanced_sampler(data)
        shuffle = False
    return _built(DataLoader(data, batch_size=size, shuffle=shuffle, drop_last=drop_last, sampler=sampler,
                             num_workers=int(batch.get("workers", 0) or 0), collate_fn=batch.get("collate")),
                  set, size)


def balanced_sampler(data):
    """A sampler drawing every class of the single target field equally often, with replacement."""
    import torch as torch_module
    from torch.utils.data import WeightedRandomSampler

    targets = list(getattr(data, "targets", []))
    if len(targets) != 1:
        raise ValueError(f"balanced needs exactly one target field, the dataset has {targets}")
    labels = data.labels(targets[0]).reshape(-1).long()
    counts = torch_module.bincount(labels).float()
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
