"""The aliasing check: unordered siblings sharing a mutated object are reported; the std flow is clean; a run on
the thread executor gives the same history as a serial one."""

import cirak
import pytest
import tezgah
from cirak.registry import registry

import kalfa  # noqa: F401
from conftest import ROOT
from helpers import minimal, write_config
from kalfa.aliasing import aliasing_problems, keys_of
from kalfa.api import prepare, run
from kalfa.config import parse_sets
from kalfa.record import read_history


@kalfa.lego("/lego/test/mutate", mutates=["thing"], returns=["thing"])
def mutate(thing):
    thing["count"] = thing.get("count", 0) + 1
    return {"thing": thing}


@kalfa.lego("/lego/test/peek", returns="seen")
def peek(thing):
    return dict(thing)


@kalfa.lego("/lego/test/wrap", aliases="thing", returns="box")
def wrap(thing):
    return {"inner": thing}


def test_keys_of_flattens_bundles():
    assert keys_of("a") == ["a"]
    assert keys_of({"x": "a", "y": {"z": "b"}}) == ["a", "b"]
    assert keys_of(["a", ["b"]]) == ["a", "b"] and keys_of(None) == []


def test_unordered_siblings_sharing_a_mutated_object_are_reported():
    pipe = tezgah.Pipeline([
        tezgah.Step(mutate, inputs={"thing": "thing"}, outputs={"thing": "thing_next"}, name="bump"),
        tezgah.Step(peek, inputs={"thing": "thing"}, outputs=["seen"], name="look"),
    ], name="root")
    pipe.validate(["thing"])
    problems = aliasing_problems(pipe, registry)
    assert [problem.kind for problem in problems] == ["aliasing"]
    assert "bump mutates ['thing']" in problems[0].message and "look" in problems[0].message
    ordered = tezgah.Pipeline([
        tezgah.Step(mutate, inputs={"thing": "thing"}, outputs={"thing": "thing_next"}, name="bump"),
        tezgah.Step(peek, inputs={"thing": "thing_next"}, outputs=["seen"], name="look"),
    ], name="root")
    ordered.validate(["thing"])
    assert aliasing_problems(ordered, registry) == []


def test_aliases_fact_joins_the_output_with_its_inputs():
    pipe = tezgah.Pipeline([
        tezgah.Step(wrap, inputs={"thing": "thing"}, outputs=["box"], name="wrap"),
        tezgah.Step(mutate, inputs={"thing": "box"}, outputs={"thing": "box_next"}, name="bump"),
        tezgah.Step(peek, inputs={"thing": "thing"}, outputs=["seen"], name="look"),
    ], name="root")
    pipe.validate(["thing"])
    problems = aliasing_problems(pipe, registry)
    assert len(problems) == 1 and "bump mutates ['box']" in problems[0].message and "look" in problems[0].message


@pytest.mark.parametrize("config", ["configs/01_mlp_regression.yaml", "examples/alad/config.yaml",
                                    "configs/07_wgan_gp.yaml", "configs/10_char_lm.yaml"])
def test_the_std_flow_has_no_aliasing_hazard(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prepared = prepare([str(ROOT / config)], parse_sets([]))
    assert prepared.aliasing == []


def test_thread_executor_gives_the_serial_history(workdir):
    path = write_config(workdir / "cfg.yaml", minimal(training__epochs=2))
    serial = run([str(path)], parse_sets([]), when="serial")
    threaded = run([str(path)], parse_sets([]), executor="thread", workers=3, when="thread")
    first = read_history(serial.record)
    second = read_history(threaded.record)
    assert [line["val/rmse"] for line in first] == [line["val/rmse"] for line in second]
