from torch.utils.data import DataLoader, IterableDataset

from kalfa.core.errors import BuildError


def config_directories(ctx):
    return tuple(getattr(getattr(ctx, "config", None), "directories", None) or ())


def seeded_split(split, ctx):
    seed = getattr(ctx, "seed", None)
    if getattr(split, "seed", 0) is None and seed is not None:
        split.seed = seed
    return split


class Loaders:
    def make_loader(self, dataset, train):
        spec = self.batch
        if isinstance(dataset, IterableDataset):
            return DataLoader(dataset, batch_size=spec["size"] if train else (spec["eval_size"] or spec["size"]),
                              shuffle=False, drop_last=train and spec["drop_last"], num_workers=spec["workers"])
        if train:
            shuffle = spec["shuffle"]
            generator = self.rng.stream("loader") if shuffle and self.rng is not None else None
            return DataLoader(dataset, batch_size=spec["size"], shuffle=shuffle, drop_last=spec["drop_last"],
                              num_workers=spec["workers"], generator=generator)
        return DataLoader(dataset, batch_size=spec["eval_size"] or spec["size"], shuffle=False,
                          num_workers=spec["workers"])

    def loaders(self):
        if not self.datasets:
            raise BuildError("the data is not fitted; call fit(ctx) before loaders()")
        if self.loaders_by_set is None:
            self.loaders_by_set = {name: self.make_loader(dataset, name == "train")
                                   for name, dataset in self.datasets.items()}
        return dict(self.loaders_by_set)

    def rows(self, set_name):
        if set_name not in self.datasets:
            raise BuildError(f"there is no {set_name!r} set; the sets are {list(self.datasets)}")
        rows = getattr(self.datasets[set_name], "rows", None)
        if rows is None:
            raise BuildError(f"the {set_name} set is streamed and has no row table")
        return rows
