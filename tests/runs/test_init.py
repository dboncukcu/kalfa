from pathlib import Path

import pytest
import torch

from helpers import minimal, write_config
from kalfa.api import run
from kalfa.config import parse_sets


pytestmark = pytest.mark.slow


def test_init_calls_are_built_and_applied_on_the_first_batch(workdir):
    config = minimal()
    config["params"]["lr"] = 0.0
    config["model"]["models"]["net"]["init"] = {"bias": {"uri": "zeros"},
                                               "patterns": [{"match": "*", "weights": {"uri": "zeros"}}]}
    config["record"] = "runs/init_$datetime$"
    result = run([write_config(workdir / "cfg.yaml", config)], parse_sets([]), when="fixed")
    state = torch.load(Path(result.record) / "final" / "state.pt", weights_only=False)["models"]["net"]
    assert len(state) == 6
    assert all(float(value.abs().sum()) == 0.0 for value in state.values())
