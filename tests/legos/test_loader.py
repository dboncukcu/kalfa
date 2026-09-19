import re

import numpy
import pandas
import pytest
import torch
from cirak.registry import registry
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from helpers import build, frame
from kalfa.std import STD_URIS
from kalfa.std.pre.base import TableFrame


LOADER = "/loader/kalfa/torch"


def dataset(rows=10, features=2, seed=0):
    table = frame(rows=rows, features=features, seed=seed)
    return build("/feed/kalfa/table", frame=table, frames={"train": table})


def labelled(labels, targets=("label",)):
    data = pandas.DataFrame({"x0": numpy.arange(len(labels), dtype="float32")})
    for name in targets:
        data[name] = numpy.asarray(labels, dtype="int64")
    table = TableFrame(["x0"], {name: [name] for name in targets}, "train", data=data)
    return build("/feed/kalfa/table", frame=table, frames={"train": table})


def stream_dataset(root, set_name="train", chunk=700):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=chunk)
    prep = build("/lego/kalfa/fit", df=stream, fields={"x0": {}, "price": {"target": True}}, preprocessors={},
                 drop=[])
    view = build("/lego/kalfa/apply", df=stream, prep=prep, set=set_name)
    return build("/feed/kalfa/table", frame=view, frames={set_name: view})


def prices(loader):
    return torch.cat([batch["price"] for batch in loader])


def test_loader_catalog_is_the_torch_loader_bound_to_the_device():
    assert [uri for uri in STD_URIS if uri.startswith("/loader/")] == [LOADER]
    assert registry.facts(LOADER).bus == {"device": "device"}
    assert registry.facts(LOADER).alias == ()


def test_train_loader_batches_the_set_by_size():
    data = dataset(10)
    loader = build(LOADER, data=data, set="train", size=4, shuffle=False)
    assert isinstance(loader, DataLoader)
    assert loader.dataset is data
    assert loader.batch_size == 4
    assert loader.drop_last is False
    assert len(loader) == 3
    batches = list(loader)
    assert [set(batch) for batch in batches] == [{"x", "price"}] * 3
    assert [tuple(batch["x"].shape) for batch in batches] == [(4, 2), (4, 2), (2, 2)]
    assert [tuple(batch["price"].shape) for batch in batches] == [(4,), (4,), (2,)]
    torch.testing.assert_close(prices(loader), data.labels("price"))
    torch.testing.assert_close(torch.cat([batch["x"] for batch in batches]), data.x)


def test_train_loader_shuffles_and_the_evaluation_loaders_keep_the_order():
    data = dataset(10)
    torch.manual_seed(1)
    train = build(LOADER, data=data, set="train", size=4)
    assert isinstance(train.sampler, RandomSampler)
    seen = prices(train)
    assert sorted(seen.tolist()) == sorted(data.labels("price").tolist())
    assert seen.tolist() != data.labels("price").tolist()
    for name in ("valid", "test"):
        loader = build(LOADER, data=data, set=name, size=4, shuffle=True)
        assert isinstance(loader.sampler, SequentialSampler)
        torch.testing.assert_close(prices(loader), data.labels("price"))
    ordered = build(LOADER, data=data, set="train", size=4, shuffle=False)
    assert isinstance(ordered.sampler, SequentialSampler)


def test_drop_last_auto_drops_the_last_train_batch_only_when_it_would_hold_one_row():
    lone = build(LOADER, data=dataset(9), set="train", size=4, drop_last="auto")
    assert lone.drop_last is True
    assert len(lone) == 2
    assert [tuple(batch["x"].shape) for batch in lone] == [(4, 2), (4, 2)]
    pair = build(LOADER, data=dataset(10), set="train", size=4, drop_last="auto")
    assert pair.drop_last is False
    assert len(pair) == 3
    exact = build(LOADER, data=dataset(8), set="train", size=4, drop_last="auto")
    assert exact.drop_last is False
    assert len(exact) == 2
    assert build(LOADER, data=dataset(9), set="train", size=4).drop_last is False
    assert len(build(LOADER, data=dataset(9), set="train", size=4)) == 3
    assert len(build(LOADER, data=dataset(10), set="train", size=4, drop_last=True)) == 2
    kept = build(LOADER, data=dataset(9), set="valid", size=4, drop_last=True)
    assert kept.drop_last is False
    assert len(kept) == 3
    assert build(LOADER, data=dataset(9), set="test", size=4, drop_last="auto").drop_last is False


def test_eval_size_sizes_the_evaluation_loaders_and_no_size_means_the_whole_set():
    data = dataset(10)
    assert build(LOADER, data=data, set="valid", size=4, eval_size=8).batch_size == 8
    assert len(build(LOADER, data=data, set="valid", size=4, eval_size=8)) == 2
    assert build(LOADER, data=data, set="valid", size=4).batch_size == 4
    assert build(LOADER, data=data, set="train", size=4, eval_size=8).batch_size == 4
    whole = build(LOADER, data=data, set="train")
    assert whole.batch_size == 10
    assert len(whole) == 1
    assert next(iter(whole))["x"].shape == (10, 2)
    assert build(LOADER, data=data, set="train", eval_size=8).batch_size == 10
    assert build(LOADER, data=data, set="test", eval_size=3).batch_size == 3
    empty = build(LOADER, data=dataset(0), set="valid")
    assert empty.batch_size == 1
    assert len(empty) == 0


def test_balanced_puts_a_weighted_sampler_over_the_single_target():
    data = labelled([0, 0, 0, 1])
    loader = build(LOADER, data=data, set="train", size=2, balanced=True)
    sampler = loader.sampler
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 4
    assert sampler.replacement is True
    torch.testing.assert_close(sampler.weights, torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0], dtype=torch.double))
    assert sum(len(batch["label"]) for batch in loader) == 4
    gap = build(LOADER, data=labelled([0, 2, 2]), set="train", size=3, balanced=True)
    torch.testing.assert_close(gap.sampler.weights, torch.tensor([1.0, 0.5, 0.5], dtype=torch.double))
    assert isinstance(build(LOADER, data=data, set="valid", size=2, balanced=True).sampler, SequentialSampler)
    hollow = build(LOADER, data=labelled([]), set="train", size=2, balanced=True, shuffle=False)
    assert isinstance(hollow.sampler, SequentialSampler)
    assert len(hollow) == 0
    with pytest.raises(ValueError, match=re.escape("balanced needs exactly one target field, the dataset has ['a', "
                                                   "'b']")):
        build(LOADER, data=labelled([0, 1], targets=("a", "b")), set="train", size=2, balanced=True)


def test_workers_default_to_zero_and_eval_workers_follow_workers():
    data = dataset(10)
    plain = build(LOADER, data=data, set="train", size=4)
    assert plain.num_workers == 0
    assert plain.persistent_workers is False
    assert plain.prefetch_factor is None
    assert plain.worker_init_fn is None
    kept = build(LOADER, data=data, set="train", size=4, workers=2)
    assert kept.num_workers == 2
    assert kept.persistent_workers is True
    assert kept.prefetch_factor == 2
    assert kept.worker_init_fn.__name__ == "seed_worker"
    fresh = build(LOADER, data=data, set="train", size=4, workers=2, persistent=False, prefetch=4)
    assert fresh.persistent_workers is False
    assert fresh.prefetch_factor == 4
    assert build(LOADER, data=data, set="valid", size=4, workers=2, eval_workers=1).num_workers == 1
    assert build(LOADER, data=data, set="valid", size=4, workers=2).num_workers == 2
    assert build(LOADER, data=data, set="valid", size=4, workers=2, eval_workers=0).num_workers == 0
    assert build(LOADER, data=data, set="train", size=4, workers=2, eval_workers=1).num_workers == 2
    assert build(LOADER, data=data, set="train", size=4, workers=None).num_workers == 0


def test_pin_memory_follows_the_device_unless_written():
    data = dataset(10)
    assert build(LOADER, data=data, set="train", size=4).pin_memory is False
    assert build(LOADER, data=data, set="train", size=4, device=torch.device("cpu")).pin_memory is False
    assert build(LOADER, data=data, set="train", size=4, device=torch.device("cuda")).pin_memory is True
    assert build(LOADER, data=data, set="valid", size=4, device=torch.device("cuda")).pin_memory is True
    assert build(LOADER, data=data, set="train", size=4, pin_memory=True).pin_memory is True
    assert build(LOADER, data=data, set="train", size=4, pin_memory=False, device=torch.device("cuda")).pin_memory \
        is False


def test_collate_reaches_the_loader():
    def as_list(items):
        return list(items)

    loader = build(LOADER, data=dataset(10), set="valid", size=4, collate=as_list)
    assert loader.collate_fn is as_list
    batch = next(iter(loader))
    assert isinstance(batch, list) and len(batch) == 4
    assert set(batch[0]) == {"x", "price"}


def test_stream_loader_batches_across_chunks_and_shuffles_through_the_buffer(root):
    data = stream_dataset(root, chunk=700)
    loader = build(LOADER, data=data, set="train", size=600, buffer=50)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 600
    assert loader.num_workers == 0
    assert loader.drop_last is False
    assert (data.shuffle, data.buffer) == (True, 50)
    torch.manual_seed(0)
    batches = list(loader)
    assert [tuple(batch["x"].shape) for batch in batches] == [(600, 1), (600, 1), (600, 1), (200, 1)]
    assert sorted(data.rows().tolist()) == list(range(2000))
    assert data.rows().tolist() != list(range(2000))
    plain = stream_dataset(root, chunk=700)
    build(LOADER, data=plain, set="train", size=600, shuffle=False)
    assert plain.shuffle is False
    assert plain.buffer == 4096
    valid = stream_dataset(root, set_name="valid", chunk=700)
    loader = build(LOADER, data=valid, set="valid", size=600, eval_size=500, shuffle=True)
    assert loader.batch_size == 500
    assert valid.shuffle is False
    assert [tuple(batch["price"].shape) for batch in loader] == [(500,)] * 4
    assert valid.rows().tolist() == list(range(2000))


def test_stream_loader_refuses_a_missing_size_balanced_and_workers(root):
    data = stream_dataset(root)
    with pytest.raises(ValueError, match=re.escape("a stream dataset needs batch.size; without batches the whole set "
                                                   "would have to be read")):
        build(LOADER, data=data, set="train")
    with pytest.raises(ValueError, match=re.escape("balanced needs a table dataset; a lazy set cannot be counted")):
        build(LOADER, data=data, set="train", size=8, balanced=True)
    with pytest.raises(ValueError, match=re.escape("a lazy set runs with workers: 0; every worker would replay the "
                                                   "whole stream")):
        build(LOADER, data=data, set="train", size=8, workers=1)
    assert build(LOADER, data=data, set="valid", size=8, workers=2, eval_workers=0).num_workers == 0


@pytest.mark.xfail(strict=True, reason="bug: resolved_drop_last (src/kalfa/std/loader/kalfa/torch.py:31) calls len() "
                                       "on the dataset before the stream branch, so drop_last auto raises TypeError "
                                       "on the lazy set instead of keeping every batch")
def test_stream_loader_with_drop_last_auto_keeps_every_batch(root):
    loader = build(LOADER, data=stream_dataset(root), set="train", size=600, drop_last="auto")
    assert loader.drop_last is False
