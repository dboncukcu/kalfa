"""The rng legos, the seeding helpers and the per process draws."""


import numpy
import torch

import kalfa  # noqa: F401
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
from kalfa.std.rng.kalfa.seeds import derived, global_stream, indexed


def test_the_three_rules():
    assert derived(7, "net", 0) == derived_seed(7, "net") and derived(7, "net", 0) == derived(7, "net", 5)
    assert derived(7, "net", 0) != derived(7, "other", 0) and derived(None, "net", 0) is None
    assert indexed(7, "net", 1) == derived_seed(7, 1) and indexed(7, "other", 1) == indexed(7, "net", 1)
    assert indexed(None, "net", 1) is None
    assert global_stream(7, "net", 0) is None and global_stream(None, "net", 0) is None


def test_seed_all_and_the_fork_leave_the_global_stream_where_it_was():
    seed_all(11)
    first = torch.rand(2)
    seed_all(11)
    with forked(3):
        inside = torch.rand(2)
    with forked(3):
        again = torch.rand(2)
    assert torch.equal(inside, again) and torch.equal(torch.rand(2), first)
    with forked(None):
        torch.rand(2)
    assert not torch.equal(torch.rand(2), first)


def test_rng_states_round_trip():
    seed_all(5)
    states = rng_states()
    expected = (numpy.random.random(), float(torch.rand(1)))
    restore_rng(states)
    assert (numpy.random.random(), float(torch.rand(1))) == expected
    restore_rng(None)


def test_draws_are_per_process_and_salted():
    seed_all(9)
    same = Draws("crop")
    other = Draws("jitter")
    assert same.numpy() is same.numpy() and process_seed() == torch.initial_seed()
    seed_all(9)
    assert Draws("crop").numpy().random() == Draws("crop").numpy().random()
    assert same.numpy().random() != other.numpy().random()


def test_seed_worker_seeds_numpy_from_the_torch_seed():
    seed_all(21)
    seed_worker(0)
    first = numpy.random.random()
    seed_worker(0)
    assert numpy.random.random() == first
