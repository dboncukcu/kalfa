import kalfa
from kalfa.core import protocols


def test_package_imports():
    assert kalfa.__version__.startswith("3.")


def test_every_protocol_is_exported_at_the_top_level():
    for name in ("Action", "ContextMetric", "Logger", "Objective", "Step", "Trigger"):
        assert getattr(kalfa, name) is getattr(protocols, name) and name in kalfa.__all__
