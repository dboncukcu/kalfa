import hashlib
import os
import random

import numpy
import pytest
import torch
from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.common.rng import (
    Draws,
    derived_seed,
    forked,
    process_seed,
    restore_rng,
    rng_states,
    seed_all,
    seed_worker,
)


RNG_URIS = sorted(uri for uri in STD_URIS if uri.startswith("/rng/"))


def sha_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big") % (2 ** 63)


def test_the_rng_pack_holds_the_three_rules():
    assert RNG_URIS == ["/rng/kalfa/derived", "/rng/kalfa/global", "/rng/kalfa/indexed"]


@pytest.mark.parametrize("uri", RNG_URIS, ids=[uri.rsplit("/", 1)[1] for uri in RNG_URIS])
def test_rng_lego_is_a_partial_named_by_its_alias(uri):
    assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
    assert registry.facts(uri).partial is True
    assert callable(build(uri))


def test_derived_seeds_the_build_by_name_alone():
    derived = build("/rng/kalfa/derived")
    assert derived(7, "net", 0) == sha_seed(7, "net") == derived_seed(7, "net")
    assert derived(7, "net", 5) == derived(7, "net", 0)
    assert derived(7, "other", 0) != derived(7, "net", 0)
    assert derived(8, "net", 0) != derived(7, "net", 0)
    assert derived(None, "net", 0) is None
    assert 0 <= derived(7, "net", 0) < 2 ** 63


def test_indexed_seeds_the_build_by_position_alone():
    indexed = build("/rng/kalfa/indexed")
    assert indexed(7, "net", 1) == sha_seed(7, 1) == derived_seed(7, 1)
    assert indexed(7, "other", 1) == indexed(7, "net", 1)
    assert indexed(7, "net", 2) != indexed(7, "net", 1)
    assert indexed(None, "net", 1) is None


def test_global_never_seeds_a_build():
    global_stream = build("/rng/kalfa/global")
    assert global_stream(7, "net", 0) is None
    assert global_stream(None, "net", 3) is None


def test_derived_seed_is_a_stable_function_of_seed_and_name():
    assert derived_seed(11, "tower") == derived_seed(11, "tower") == sha_seed(11, "tower")
    assert derived_seed(11, "tower") != derived_seed(11, "head")
    assert derived_seed(0, "a") != derived_seed(1, "a")
    assert {derived_seed(11, name) for name in ("a", "b", "c", "d")} == {sha_seed(11, name) for name in "abcd"}


def test_seed_all_restarts_the_three_streams_and_none_leaves_them_alone():
    seed_all(11)
    draws = (torch.rand(3).tolist(), random.random(), numpy.random.random())
    seed_all(11)
    assert (torch.rand(3).tolist(), random.random(), numpy.random.random()) == draws
    assert torch.initial_seed() == 11
    states = rng_states()
    seed_all(None)
    assert torch.equal(torch.get_rng_state(), states["torch"])
    assert random.getstate() == states["python"]
    assert numpy.random.get_state()[2] == states["numpy"][2]
    assert numpy.array_equal(numpy.random.get_state()[1], states["numpy"][1])


def test_seed_all_folds_the_numpy_seed_into_32_bits():
    seed_all(2 ** 32 + 5)
    assert torch.initial_seed() == 2 ** 32 + 5
    assert numpy.random.random() == numpy.random.RandomState(5).random_sample()
    assert random.random() == random.Random(2 ** 32 + 5).random()


def test_rng_states_capture_and_restore_every_stream():
    seed_all(5)
    states = rng_states()
    expected = {"python", "numpy", "torch"} | ({"cuda"} if torch.cuda.is_available() else set())
    assert set(states) == expected
    draws = (numpy.random.random(), float(torch.rand(1)), random.random())
    restore_rng(states)
    assert (numpy.random.random(), float(torch.rand(1)), random.random()) == draws
    restore_rng(states)
    only_torch = float(torch.rand(1))
    assert numpy.random.random() == draws[0]
    restore_rng({"torch": states["torch"]})
    assert float(torch.rand(1)) == only_torch
    assert numpy.random.random() != draws[0]
    before = rng_states()
    restore_rng(None)
    restore_rng({})
    assert torch.equal(torch.get_rng_state(), before["torch"]) and random.getstate() == before["python"]


def test_forked_runs_under_its_seed_and_hands_the_global_stream_back():
    seed_all(11)
    first = torch.rand(2)
    seed_all(11)
    context = forked(3)
    assert context.seed == 3
    with context:
        inside = torch.rand(2)
    with forked(3):
        again = torch.rand(2)
    assert torch.equal(inside, again)
    assert torch.equal(torch.rand(2), first)
    torch.manual_seed(3)
    assert torch.equal(torch.rand(2), inside)


def test_forked_without_a_seed_draws_from_the_global_stream():
    seed_all(11)
    first = torch.rand(2)
    seed_all(11)
    with forked(None):
        consumed = torch.rand(2)
    assert torch.equal(consumed, first)
    assert not torch.equal(torch.rand(2), first)


def test_seed_worker_seeds_numpy_and_python_from_the_torch_seed():
    seed_all(21)
    assert process_seed() == torch.initial_seed() == 21
    seed_worker(0)
    assert numpy.random.random() == numpy.random.RandomState(21).random_sample()
    assert random.random() == random.Random(21).random()
    seed_all(2 ** 40 + 3)
    seed_worker(4)
    assert numpy.random.random() == numpy.random.RandomState(3).random_sample()
    assert random.random() == random.Random(3).random()


def test_draws_are_salted_and_seeded_once_per_process():
    seed_all(9)
    draws = Draws("crop")
    assert draws.salt == int.from_bytes(hashlib.sha256(b"crop").digest()[:4], "big")
    assert draws.process is None and draws.generator is None
    generator = draws.numpy()
    assert draws.numpy() is generator and draws.process == os.getpid()
    assert generator.random() == numpy.random.default_rng([9, draws.salt]).random()
    assert Draws("crop").numpy().random() == numpy.random.default_rng([9, draws.salt]).random()
    assert Draws("jitter").salt != draws.salt
    assert Draws("jitter").numpy().random() != numpy.random.default_rng([9, draws.salt]).random()
    draws.process = -1
    fresh = draws.numpy()
    assert fresh is not generator and draws.process == os.getpid()
    assert fresh.random() == numpy.random.default_rng([9, draws.salt]).random()
