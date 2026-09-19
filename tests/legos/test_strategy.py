import math

import numpy
import pytest
from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.strategy.base import Choices, Range, Strategy, parse_space
from kalfa.std.strategy.kalfa.deterministic import Grid, RandomSearch, SobolSearch
from kalfa.std.strategy.kalfa.optuna import OptunaSearch


STRATEGIES = sorted(uri for uri in STD_URIS if uri.startswith("/strategy/"))

SPACE = {"lr": {"low": 0.1, "high": 0.5}, "width": [8, 16, 32, 64], "depth": {"low": 1, "high": 5, "int": True},
         "decay": {"low": 1e-4, "high": 1e-2, "log": True}}


def inside(point, space):
    assert set(point) == set(space)
    assert 0.1 <= point["lr"] <= 0.5
    assert point["width"] in (8, 16, 32, 64)
    assert isinstance(point["depth"], int) and 1 <= point["depth"] <= 5
    assert 1e-4 <= point["decay"] <= 1e-2


def test_strategy_scope_is_the_four_kalfa_strategies():
    assert STRATEGIES == ["/strategy/kalfa/grid", "/strategy/kalfa/optuna", "/strategy/kalfa/random",
                          "/strategy/kalfa/sobol"]
    for uri in STRATEGIES:
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri


def test_only_the_grid_enumerates_its_space():
    assert registry.facts("/strategy/kalfa/grid").get("enumerates") is True
    for uri in ("/strategy/kalfa/random", "/strategy/kalfa/sobol", "/strategy/kalfa/optuna"):
        assert registry.facts(uri).get("enumerates") is None


def test_parse_space_reads_choices_and_ranges():
    space = parse_space(SPACE)
    assert space["lr"] == Range(0.1, 0.5, False, False, None)
    assert space["width"] == Choices([8, 16, 32, 64])
    assert space["depth"] == Range(1.0, 5.0, False, True, None)
    assert space["decay"] == Range(1e-4, 1e-2, True, False, None)
    assert parse_space({"n": {"low": 2, "high": 4, "steps": 3}})["n"] == Range(2.0, 4.0, False, False, 3)


def test_parse_space_names_every_shape_it_refuses():
    with pytest.raises(ValueError, match=r"sweep.space maps param names to a list of choices or a range "
                                          r"\{low, high, log, int, steps\}"):
        parse_space([])
    with pytest.raises(ValueError, match=r"sweep.space.lr: the list of choices is empty"):
        parse_space({"lr": []})
    with pytest.raises(ValueError, match=r"sweep.space.w: a range is \{low, high, log, int, steps\}, got "
                                          r"\['high', 'low', 'max'\]"):
        parse_space({"w": {"low": 1, "high": 2, "max": 3}})
    with pytest.raises(ValueError, match=r"sweep.space.w: low must be a number below high"):
        parse_space({"w": {"low": 2, "high": 1}})
    with pytest.raises(ValueError, match=r"sweep.space.w: low must be a number below high"):
        parse_space({"w": {"low": True, "high": 2}})
    with pytest.raises(ValueError, match=r"sweep.space.w: a log range needs low > 0"):
        parse_space({"w": {"low": 0, "high": 1, "log": True}})
    with pytest.raises(ValueError, match=r"sweep.space.w: a list of choices or a range mapping"):
        parse_space({"w": 3})


def test_grid_enumerates_every_combination_with_the_last_param_fastest():
    grid = build("/strategy/kalfa/grid")
    assert isinstance(grid, Grid) and isinstance(grid, Strategy)
    assert grid.deterministic is True
    space = parse_space({"a": [1, 2], "b": ["x", "y"]})
    assert grid.total(space) == 4
    assert [grid.point(space, index) for index in range(4)] == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"},
                                                                {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]


def test_grid_walks_a_range_in_steps_linear_log_or_integer():
    grid = build("/strategy/kalfa/grid")
    space = parse_space({"w": {"low": 8, "high": 32, "int": True, "steps": 3},
                         "d": {"low": 1e-4, "high": 1e-2, "log": True, "steps": 3},
                         "f": {"low": 0.0, "high": 1.0, "steps": 5}})
    assert grid.total(space) == 45
    assert [grid.point(space, index)["w"] for index in (0, 15, 30)] == [8, 20, 32]
    assert [grid.point(space, index)["d"] for index in (0, 5, 10)] == pytest.approx([1e-4, 1e-3, 1e-2])
    assert [grid.point(space, index)["f"] for index in range(5)] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert grid.point(space, 44) == {"w": 32, "d": pytest.approx(1e-2), "f": 1.0}


def test_grid_refuses_a_point_outside_and_a_range_without_steps():
    grid = build("/strategy/kalfa/grid")
    space = parse_space({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(ValueError, match=r"point 4 is outside the grid of 4 points"):
        grid.point(space, 4)
    with pytest.raises(ValueError, match=r"point -1 is outside the grid of 4 points"):
        grid.point(space, -1)
    with pytest.raises(ValueError, match=r"sweep.space.a: grid needs a list of choices or a range with steps >= 2"):
        grid.total(parse_space({"a": {"low": 0, "high": 1}}))
    with pytest.raises(ValueError, match=r"sweep.space.a: grid needs a list of choices or a range with steps >= 2"):
        grid.total(parse_space({"a": {"low": 0, "high": 1, "steps": 1}}))


def test_random_draws_the_same_points_for_the_same_seed_and_id():
    space = parse_space(SPACE)
    first = build("/strategy/kalfa/random", count=5, seed=7)
    assert isinstance(first, RandomSearch) and first.deterministic is True
    assert first.total(space) == 5
    again = build("/strategy/kalfa/random", count=5, seed=7)
    points = [first.point(space, index) for index in range(5)]
    assert points == [again.point(space, index) for index in range(5)]
    for point in points:
        inside(point, space)
    assert len({tuple(point.values()) for point in points}) == 5
    other = build("/strategy/kalfa/random", count=5, seed=8)
    assert [other.point(space, index) for index in range(5)] != points


def test_random_maps_uniform_fractions_onto_the_space():
    space = parse_space(SPACE)
    fractions = numpy.random.default_rng([7, 3]).random(4)
    point = build("/strategy/kalfa/random", count=5, seed=7).point(space, 3)
    assert point["lr"] == 0.1 + fractions[0] * (0.5 - 0.1)
    assert point["width"] == [8, 16, 32, 64][min(int(fractions[1] * 4), 3)]
    assert point["depth"] == int(round(1.0 + fractions[2] * 4.0))
    assert point["decay"] == math.exp(math.log(1e-4) + fractions[3] * (math.log(1e-2) - math.log(1e-4)))


def test_random_refuses_a_point_outside_its_count():
    strategy = build("/strategy/kalfa/random", count=5)
    with pytest.raises(ValueError, match=r"point 5 is outside the 5 random points"):
        strategy.point(parse_space(SPACE), 5)


def test_unscrambled_sobol_walks_the_known_sequence():
    space = parse_space({"lr": {"low": 0.0, "high": 1.0}, "width": [8, 16, 32, 64]})
    strategy = build("/strategy/kalfa/sobol", count=4, scramble=False)
    assert isinstance(strategy, SobolSearch) and strategy.deterministic is True
    assert strategy.total(space) == 4
    assert [strategy.point(space, index) for index in range(4)] == [
        {"lr": 0.0, "width": 8}, {"lr": 0.5, "width": 32}, {"lr": 0.75, "width": 16}, {"lr": 0.25, "width": 64}]


def test_scrambled_sobol_is_deterministic_by_seed():
    space = parse_space(SPACE)
    first = build("/strategy/kalfa/sobol", count=6, seed=3)
    again = build("/strategy/kalfa/sobol", count=6, seed=3)
    points = [first.point(space, index) for index in range(6)]
    assert points == [again.point(space, index) for index in range(6)]
    for point in points:
        inside(point, space)
    assert [build("/strategy/kalfa/sobol", count=6, seed=4).point(space, index) for index in range(6)] != points
    with pytest.raises(ValueError, match=r"point 6 is outside the 6 sobol points"):
        first.point(space, 6)


def test_optuna_is_not_deterministic_and_asks_its_study_for_points():
    strategy = build("/strategy/kalfa/optuna", trials=3, seed=1, sampler="random")
    assert isinstance(strategy, OptunaSearch) and strategy.deterministic is False
    space = parse_space(SPACE)
    assert strategy.total(space) == 3
    assert strategy.study is None
    trial, point = strategy.ask(space)
    inside(point, space)
    assert strategy.study.direction.name == "MINIMIZE"
    strategy.tell(trial, 0.5)
    assert strategy.study.trials[0].value == 0.5
    failed, _ = strategy.ask(space)
    strategy.tell(failed, None)
    assert strategy.study.trials[1].state.name == "FAIL"
    nan_trial, _ = strategy.ask(space)
    strategy.tell(nan_trial, math.nan)
    assert strategy.study.trials[2].state.name == "FAIL"


def test_optuna_random_sampler_repeats_its_first_point_under_a_seed():
    space = parse_space(SPACE)
    first = build("/strategy/kalfa/optuna", trials=2, seed=5, sampler="random").ask(space)[1]
    again = build("/strategy/kalfa/optuna", trials=2, seed=5, sampler="random").ask(space)[1]
    assert first == again
    tpe = build("/strategy/kalfa/optuna", trials=2, seed=5)
    inside(tpe.ask(space, mode="max")[1], space)
    assert tpe.study.direction.name == "MAXIMIZE"
