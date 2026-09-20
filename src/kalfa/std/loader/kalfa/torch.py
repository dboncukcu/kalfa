import torch

from kalfa.std.common.log import logger_for
from kalfa.std.common.rng import seed_worker
from kalfa.std.feed.base import IterableDataset


logger = logger_for("data.loader")


def report_built(loader, set, size, workers=0, pin=False):
    notes = [f"{workers} workers, kept alive" if workers else "", "pinned memory" if pin else ""]
    detail = "".join(f", {note}" for note in notes if note)
    try:
        logger.info(f"{set}: {len(loader)} batches of {size}{detail}")
    except TypeError:
        logger.info(f"{set}: a stream in batches of {size}{detail}")
    return loader


def resolved_size(size, data):
    if size is not None:
        return int(size)
    if isinstance(data, torch.utils.data.IterableDataset):
        raise ValueError("a stream dataset needs batch.size; without batches the whole set would have to be read")
    return max(len(data), 1)


def resolved_drop_last(drop_last, data, size):
    if drop_last != "auto":
        return bool(drop_last)
    if isinstance(data, torch.utils.data.IterableDataset):
        return False
    return len(data) % size == 1


def resolved_workers(workers, eval_workers, train):
    if train or eval_workers is None:
        return int(workers or 0)
    return int(eval_workers or 0)


def pinned(pin_memory, device):
    if pin_memory is not None:
        return bool(pin_memory)
    return device is not None and device.type == "cuda"


def worker_options(workers, persistent, prefetch):
    if not workers:
        return {}
    options = {"worker_init_fn": seed_worker,
               "persistent_workers": True if persistent is None else bool(persistent)}
    if prefetch is not None:
        options["prefetch_factor"] = int(prefetch)
    return options


def balanced_sampler(data):
    targets = list(data.targets)
    if len(targets) != 1:
        raise ValueError(f"balanced needs exactly one target field, the dataset has {targets}")
    labels = data.labels(targets[0]).reshape(-1).long()
    counts = torch.bincount(labels).float()
    weights = 1.0 / counts[labels]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def stream_loader(data, train, size, shuffle, drop_last, workers, collate, balanced, buffer, pin):
    if balanced:
        raise ValueError("balanced needs a table dataset; a lazy set cannot be counted")
    if workers:
        raise ValueError("a lazy set runs with workers: 0; every worker would replay the whole stream")
    if train and isinstance(data, IterableDataset):
        data.shuffle = shuffle
        data.buffer = int(buffer)
    return torch.utils.data.DataLoader(data, batch_size=size, drop_last=drop_last, num_workers=0, collate_fn=collate,
                                       pin_memory=pin)


def torch_loader(data, set, size=None, eval_size=None, shuffle=True, drop_last=False, workers=0, eval_workers=None,
                 collate=None, balanced=False, buffer=4096, persistent=None, pin_memory=None, prefetch=None,
                 device=None):
    train = set == "train"
    size = resolved_size(size if train else (eval_size or size), data)
    shuffle = bool(shuffle) if train else False
    drop_last = resolved_drop_last(drop_last, data, size) if train else False
    workers = resolved_workers(workers, eval_workers, train)
    pin = pinned(pin_memory, device)
    if isinstance(data, torch.utils.data.IterableDataset):
        return report_built(stream_loader(data, train, size, shuffle, drop_last, workers, collate, balanced, buffer,
                                          pin), set, size, 0, pin)
    sampler = None
    if balanced and train and len(data):
        sampler = balanced_sampler(data)
        shuffle = False
    loader = torch.utils.data.DataLoader(data, batch_size=size, shuffle=shuffle, drop_last=drop_last, sampler=sampler,
                                         num_workers=workers, collate_fn=collate, pin_memory=pin,
                                         **worker_options(workers, persistent, prefetch))
    return report_built(loader, set, size, workers, pin)
