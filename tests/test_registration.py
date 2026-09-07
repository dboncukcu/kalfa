"""kalfa.lego: the kind comes from the first segment of the URI, cirak stores its own kind names for triggers."""

import pytest
from cirak.registry import registry

import kalfa
from kalfa.kinds import KINDS, cirak_kind, kalfa_kind, kind_of


def test_kind_is_the_first_uri_segment():
    assert kind_of("/criterion/kalfa/mae") == "criterion" and kind_of("/turn/proj/x") == "turn"
    assert kind_of("/data/kalfa/class_weights") == "data" and kind_of("/builder/kalfa/module") == "builder"
    for bad in ("criterion/kalfa/mae", "/loss/kalfa/mae", "/criterion", "/std/kalfa/pack", 3):
        with pytest.raises(ValueError):
            kind_of(bad)
    assert kalfa_kind("/loss/kalfa/mae") is None and kalfa_kind("/plot/kalfa/loss_curve") == "plot"
    assert cirak_kind("trigger") == "predicate" and cirak_kind("layer") == "layer" and cirak_kind("data") == "data"
    assert "predicate" not in KINDS


def test_kalfa_lego_registers_with_the_derived_kind():
    @kalfa.lego("/metric/test/registered", state=True, description="a test metric")
    def registered():
        return None

    entry = registry.lookup("/metric/test/registered")
    assert entry is not None and entry.facts.kind == "metric" and entry.facts.state is True
    assert kalfa_kind("/metric/test/registered") == "metric"

    @kalfa.lego("/trigger/test/always", partial=True)
    def always(metrics, turn_index, state):
        return True

    assert registry.facts("/trigger/test/always").kind == "predicate"
    assert kalfa_kind("/trigger/test/always") == "trigger"
    with pytest.raises(ValueError, match="kalfa kind"):
        kalfa.lego("/nope/test/x")(lambda: None)
    with pytest.raises(ValueError, match="drop kind="):
        kalfa.lego("/layer/test/x", kind="layer")(lambda: None)


def test_every_std_uri_follows_the_rule():
    from kalfa.std import STD_URIS

    for uri in STD_URIS:
        assert kalfa_kind(uri) is not None, uri
        stored = registry.facts(uri).kind
        assert stored == cirak_kind(kalfa_kind(uri)), uri
