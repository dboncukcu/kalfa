import torch

from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.rng import seed_worker
from kalfa.std.feed.base import IterableDataset


logger = logger_for("data.loader")


def report_built(loader, set, size):
    try:
        logger.info(f"{set}: {len(loader)} batches of {size}")
    except TypeError:
        logger.info(f"{set}: a stream in batches of {size}")
    return loader


def balanced_sampler(data):
    targets = list(data.targets)
    if len(targets) != 1:
        raise ValueError(f"balanced needs exactly one target field, the dataset has {targets}")
    labels = data.labels(targets[0]).reshape(-1).long()
    counts = torch.bincount(labels).float()
    weights = 1.0 / counts[labels]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def stream_loader(data, train, size, shuffle, drop_last, workers, collate, balanced, buffer):
    if balanced:
        raise ValueError("balanced needs a table dataset; a lazy set cannot be counted")
    if workers:
        raise ValueError("a lazy set runs with workers: 0; every worker would replay the whole stream")
    if train and isinstance(data, IterableDataset):
        data.shuffle = shuffle
        data.buffer = int(buffer)
    return torch.utils.data.DataLoader(data, batch_size=size, drop_last=drop_last, num_workers=0, collate_fn=collate)


@lego("/loader/kalfa/torch",
      description="torch DataLoader over a dataset: size batches shuffled for the train set, eval_size batches "
                  "in order for the other sets; balanced puts a class balancing sampler over the single target "
                  "field; every worker is seeded from the torch seed on its own; a stream dataset shuffles "
                  "through buffer rows and takes no sampler or workers")
def torch_loader(data, set, size, eval_size=None, shuffle=True, drop_last=False, workers=0, collate=None,
                 balanced=False, buffer=4096):
    train = set == "train"
    size = int(size) if train else int(eval_size or size)
    shuffle = bool(shuffle) if train else False
    drop_last = bool(drop_last) if train else False
    workers = int(workers or 0)
    if isinstance(data, torch.utils.data.IterableDataset):
        return report_built(stream_loader(data, train, size, shuffle, drop_last, workers, collate, balanced, buffer),
                            set, size)
    sampler = None
    if balanced and train and len(data):
        sampler = balanced_sampler(data)
        shuffle = False
    loader = torch.utils.data.DataLoader(data, batch_size=size, shuffle=shuffle, drop_last=drop_last, sampler=sampler,
                                         num_workers=workers, collate_fn=collate,
                                         worker_init_fn=seed_worker if workers else None)
    return report_built(loader, set, size)
