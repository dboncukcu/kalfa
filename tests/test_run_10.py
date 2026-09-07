"""Config 10 end to end: text lines, the tokenizer in the record, the steps mode across turns and resumes, sampling."""

from pathlib import Path

import pytest
import torch

import kalfa  # noqa: F401
from conftest import ROOT
from kalfa.api import check, generate, resume, run
from kalfa.config import parse_sets
from kalfa.record import read_history
from kalfa.synthetic import write_text

CONFIG = str(ROOT / "configs" / "10_char_lm.yaml")
SMALL = ["device=cpu", "data.batch=4",
         "model.models.gpt.nodes=[{uri: embedding, params: {num: {uri: vocab_size}, dim: 16}}, "
         "{uri: gpt, params: {d_model: 16, layers: 1, heads: 2, seq_len: 16}}, "
         "{uri: linear, params: {out_features: {uri: vocab_size}}}]",
         "optimizers.main.schedule={uri: warmup_cosine, params: {warmup: 1, total: 12}}",
         "generate.params.max_new_tokens=20"]
PARAMS = ["total_steps=8", "turn_steps=2", "seq_len=16"]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    write_text(tmp_path / "data" / "shakespeare.txt", lines=200)
    return tmp_path


def test_check_10(corpus):
    prepared = check([CONFIG], parse_sets(SMALL, PARAMS))
    assert prepared.problems == []
    assert prepared.sizes == {"train": 196, "valid": 4, "test": 0}
    document = prepared.document
    assert document["flow"]["training"]["params"]["steps"] == {"total": 8, "turn": 2}
    assert document["flow"]["training"]["params"]["turn_params"] == {"accumulate": 2, "amp": True, "grad_clip": 1.0}
    assert document["flow"]["data"]["params"]["feed"] == {"uri": "/feed/kalfa/next_token", "params": {"seq_len": 16}}
    assert document["flow"]["training"]["params"]["metrics_keys"] == {"perplexity": {"target": "targets"}}
    nodes = document["blocks"]["gpt"]["spec"]
    assert nodes[0]["params"]["num"] == {"uri": "/data/kalfa/vocab_size"}
    assert nodes[2]["params"]["out_features"] == {"uri": "/data/kalfa/vocab_size"}


def test_run_10_steps_mode_and_resume(corpus):
    result = run([CONFIG], parse_sets(SMALL, PARAMS), when="fixed")
    record = Path(result.record)
    history = read_history(record)
    assert [line["turn"] for line in history] == [1, 2, 3, 4]
    assert [line["global_step"] for line in history] == [2, 4, 6, 8]
    assert {"train/lm", "train/perplexity", "val/lm", "val/perplexity", "lr/main"} <= set(history[0])
    assert all(line["val/perplexity"] > 1.0 for line in history)
    assert (record / "preprocessors" / "tokenizer.pkl").exists()
    text = (record / "samples" / "samples.txt").read_text()
    assert text.startswith("ROMEO:") and len(text) == len("ROMEO:") + 20
    payload = torch.load(record / "checkpoints" / "last.pt", weights_only=False)
    assert payload["counters"] == {"global_step": 8, "turn": 4}
    import pickle

    tokenizer = pickle.load((record / "preprocessors" / "tokenizer.pkl").open("rb"))["text"]
    head = next(name for name in payload["models"]["gpt"] if name.endswith("weight") and "nodes.s2" in name)
    assert payload["models"]["gpt"][head].shape[0] == tokenizer.size
    assert payload["models"]["gpt"]["nodes.s0.weight"].shape == (tokenizer.size, 16)
    continued = resume(result.record, parse_sets(["training.steps.total=12"]), when="more")
    more = read_history(continued.record)
    assert [line["turn"] for line in more] == [5, 6] and more[-1]["global_step"] == 12
    again = generate(result.record)
    assert again.samples.startswith("ROMEO:") and again.path.endswith("samples.txt")
