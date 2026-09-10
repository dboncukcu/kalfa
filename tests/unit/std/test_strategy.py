import json

import pytest
from ruamel.yaml import YAML

from kalfa import sweep as sweeper
from kalfa.std.strategy.base import grid_values, parse_space
from kalfa.std.strategy.kalfa.deterministic import Grid, RandomSearch, SobolSearch


def test_space_parsing_and_grid_points():
    space = parse_space({"lr": [1e-2, 1e-3, 1e-4], "width": [64, 128]})
    grid = Grid()
    assert grid.total(space) == 6
    assert grid.point(space, 0) == {"lr": 1e-2, "width": 64} and grid.point(space, 1) == {"lr": 1e-2, "width": 128}
    assert grid.point(space, 4) == {"lr": 1e-4, "width": 64} and grid.point(space, 5) == {"lr": 1e-4, "width": 128}
    ranged = parse_space({"lr": {"low": 1e-4, "high": 1e-2, "log": True, "steps": 3}})
    assert grid_values("lr", ranged["lr"]) == pytest.approx([1e-4, 1e-3, 1e-2])
    with pytest.raises(ValueError, match="steps"):
        Grid().total(parse_space({"lr": {"low": 0.0, "high": 1.0}}))
    with pytest.raises(ValueError):
        grid.point(space, 6)
    for bad in ({"lr": []}, {"lr": {"low": 1, "high": 0}}, {"lr": {"low": 0, "high": 1, "log": True}}, {"lr": 3}, {},
                {"lr": {"low": 0, "high": 1, "ghost": 2}}):
        with pytest.raises(ValueError):
            parse_space(bad)


def test_random_and_sobol_are_deterministic_by_id():
    space = parse_space({"lr": {"low": 1e-4, "high": 1e-2, "log": True}, "width": [32, 64, 128],
                         "depth": {"low": 1, "high": 4, "int": True}})
    for strategy in (RandomSearch(5, seed=3), SobolSearch(5, seed=3)):
        points = [strategy.point(space, index) for index in range(5)]
        again = type(strategy)(5, seed=3)
        assert [again.point(space, index) for index in (4, 2, 0)] == [points[4], points[2], points[0]]
        assert len({json.dumps(point, sort_keys=True) for point in points}) > 1
        for point in points:
            assert 1e-4 <= point["lr"] <= 1e-2 and point["width"] in (32, 64, 128)
            assert isinstance(point["depth"], int) and 1 <= point["depth"] <= 4
        with pytest.raises(ValueError):
            strategy.point(space, 5)
    assert RandomSearch(5, seed=4).point(space, 0) != RandomSearch(5, seed=3).point(space, 0)
    assert SobolSearch(5, seed=4).point(space, 1) != SobolSearch(5, seed=3).point(space, 1)


def test_point_values_read_back_as_the_same_scalars():
    for value in (1e-05, 0.001, 3, "relu", True, 12.5):
        assert YAML(typ="safe").load(sweeper.yaml_value(value)) == value
