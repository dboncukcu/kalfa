import torch

from kalfa.core.rng import Rng


def test_same_seed_same_numbers():
    first = Rng(7).normal("noise", (4,))
    second = Rng(7).normal("noise", (4,))
    assert torch.equal(first, second)
    assert first.shape == (4,) and first.dtype == torch.float32 and first.device.type == "cpu"
    assert torch.equal(Rng(7).normal("noise", 4), first)


def test_names_and_seeds_give_different_streams():
    rng = Rng(7)
    assert not torch.equal(rng.normal("noise", (4,)), rng.normal("other", (4,)))
    assert not torch.equal(Rng(7).normal("noise", (4,)), Rng(8).normal("noise", (4,)))
    assert Rng(7).seed_for("noise") != Rng(7).seed_for("other")


def test_streams_do_not_depend_on_draw_order():
    first = Rng(7)
    a1, b1 = first.normal("a", (3,)), first.normal("b", (3,))
    second = Rng(7)
    b2, a2 = second.normal("b", (3,)), second.normal("a", (3,))
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert first.stream("a") is first.stream("a", "cpu")
    assert set(first.streams) == {"a@cpu", "b@cpu"}


def test_like_supplies_shape_device_and_dtype():
    like = torch.zeros(2, 3, dtype=torch.float64)
    draw = Rng(1).normal("noise", None, like=like)
    assert draw.shape == (2, 3) and draw.dtype == torch.float64 and draw.device == like.device
    assert Rng(1).normal("noise", (5,), like=like, dtype=torch.float32).dtype == torch.float32
    ints = torch.zeros(5, dtype=torch.int64)
    assert Rng(1).uniform("u", (5,), like=ints).dtype == torch.float32
    assert Rng(1).uniform("u", (2, 2), device=torch.device("cpu")).shape == (2, 2)


def test_uniform_integers_and_permutation():
    rng = Rng(3)
    uniform = rng.uniform("u", (1000,))
    assert float(uniform.min()) >= 0.0 and float(uniform.max()) < 1.0
    integers = rng.integers("i", 2, 5, (200,))
    assert integers.dtype == torch.int64 and int(integers.min()) >= 2 and int(integers.max()) <= 4
    permutation = rng.permutation("p", 10)
    assert sorted(permutation.tolist()) == list(range(10))
    assert torch.equal(Rng(3).permutation("p", 10), permutation)
    assert torch.equal(Rng(3).integers("i", 2, 5, (200,)), integers)
    assert torch.equal(Rng(3).uniform("u", (1000,)), uniform)


def test_state_round_trip_continues_the_sequence():
    rng = Rng(7)
    rng.normal("noise", (3,))
    rng.integers("i", 0, 10, (2,))
    state = rng.state_dict()
    assert state["seed"] == 7 and set(state["streams"]) == {"noise@cpu", "i@cpu"}
    assert all(tensor.device.type == "cpu" for tensor in state["streams"].values())
    expected_noise = rng.normal("noise", (3,))
    expected_ints = rng.integers("i", 0, 10, (2,))
    fresh = Rng(0)
    fresh.load_state_dict(state)
    assert fresh.seed == 7 and set(fresh.pending) == {"noise@cpu", "i@cpu"} and fresh.streams == {}
    assert torch.equal(fresh.normal("noise", (3,)), expected_noise)
    assert torch.equal(fresh.integers("i", 0, 10, (2,)), expected_ints)
    assert fresh.pending == {}
    live = Rng(7)
    live.normal("noise", (1,))
    live.load_state_dict(state)
    assert torch.equal(live.normal("noise", (3,)), expected_noise)


def test_state_dict_keeps_pending_streams_and_empty_state_is_ignored():
    rng = Rng(7)
    rng.normal("noise", (3,))
    state = rng.state_dict()
    fresh = Rng(7)
    fresh.load_state_dict(state)
    again = fresh.state_dict()
    assert torch.equal(again["streams"]["noise@cpu"], state["streams"]["noise@cpu"])
    fresh.load_state_dict(None)
    fresh.load_state_dict({})
    assert set(fresh.pending) == {"noise@cpu"}


def test_seed_everything_is_repeatable():
    Rng(5).seed_everything()
    first = torch.rand(3)
    Rng(5).seed_everything()
    second = torch.rand(3)
    assert torch.equal(first, second)
