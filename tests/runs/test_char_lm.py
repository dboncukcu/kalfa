import pickle
from pathlib import Path

import pytest
import torch
from runs.conftest import SMALL

from kalfa.api import check, generate, resume
from kalfa.config import parse_sets
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

SETS, PARAMS = SMALL["10_char_lm"]


def test_check_reads_the_text_config(dataset):
    dataset("10_char_lm")
    prepared = check(["config.yaml"], parse_sets(SETS, PARAMS))
    assert prepared.problems == []
    assert prepared.sizes == {"train": 1960, "valid": 40, "test": 0}
    document = prepared.document
    assert document["flow"]["training"]["params"]["steps"] == {"total": 8, "turn": 2}
    assert document["flow"]["training"]["params"]["turn_params"] == {"accumulate": 2, "amp": False, "grad_clip": 1.0}
    assert document["flow"]["data"]["params"]["feed"] == {"uri": "/feed/kalfa/next_token", "params": {"seq_len": 16}}
    assert document["flow"]["training"]["params"]["metrics_keys"] == {"perplexity": {"target": "targets"}}
    nodes = document["blocks"]["gpt"]["spec"]
    assert nodes[0]["params"]["num"] == {"uri": "/data/kalfa/vocab_size"}
    assert nodes[2]["params"]["out_features"] == {"uri": "/data/kalfa/vocab_size"}


def test_steps_mode_the_tokenizer_resume_and_sampling(trained):
    result = trained("10_char_lm")
    record = Path(result.record)
    history = History.read(record)
    assert [line["turn"] for line in history] == [1, 2, 3, 4]
    assert [line["global_step"] for line in history] == [2, 4, 6, 8]
    assert {"train/lm", "train/perplexity", "val/lm", "val/perplexity", "lr/main"} <= set(history[0])
    assert all(line["val/perplexity"] > 1.0 for line in history)
    assert (record / "fitted" / "preprocessors" / "tokenizer.pkl").exists()
    text = (record / "samples" / "samples.txt").read_text()
    assert text.startswith("ROMEO:") and len(text) == len("ROMEO:") + 20
    payload = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    assert payload["counters"] == {"global_step": 8, "turn": 4}
    with (record / "fitted" / "preprocessors" / "tokenizer.pkl").open("rb") as stream:
        tokenizer = pickle.load(stream)["text"]
    head = next(name for name in payload["models"]["gpt"] if name.endswith("weight") and "nodes.s2" in name)
    assert payload["models"]["gpt"][head].shape[0] == tokenizer.size
    assert payload["models"]["gpt"]["nodes.s0.weight"].shape == (tokenizer.size, 16)
    assert (record / "plugins" / "gpt_legos.py").read_text() == Path("gpt_legos.py").read_text()
    continued = resume(result.record, parse_sets(["training.steps.total=12"]), when="more")
    more = History.read(continued.record)
    assert [line["turn"] for line in more] == [5, 6] and more[-1]["global_step"] == 12
    again = generate(result.record)
    assert again.samples.startswith("ROMEO:") and again.path.endswith("samples.txt")
