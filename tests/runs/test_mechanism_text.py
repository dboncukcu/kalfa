import json
import math
import pickle
from pathlib import Path

import pytest

from helpers import copied
from kalfa.api import generate, resume
from kalfa.config import parse_sets
from kalfa.std.checkpoint.base import load
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

KEYS = ["turn", "global_step", "train/lm", "train/perplexity", "val/lm", "val/perplexity", "lr/main", "minimizes/main",
        "seconds", "rules"]


@pytest.fixture
def text(trained, at_root):
    return trained("text")


def cosine(step, warmup, total, lr):
    if step < warmup:
        return lr * step / warmup
    return lr * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def test_steps_mode_counts_turns_by_updates(text):
    history = History.read(text.record)
    manifest = json.loads((Path(text.record) / "manifest.json").read_text())
    assert manifest["turn"] == "steps"
    assert [line["turn"] for line in history] == [1, 2, 3, 4]
    assert [line["global_step"] for line in history] == [2, 4, 6, 8]
    assert [list(line) for line in history] == [KEYS] * 4
    assert [line["lr/main"] for line in history] == pytest.approx([cosine(step, 2, 8, 3e-3) for step in (2, 4, 6, 8)])
    steps = History.read_steps(text.record)
    assert len(steps) == 8 and list(steps[0]) == ["step", "turn", "loss/main", "lr/main", "grad_norm/main"]


def test_the_vocabulary_of_the_fitted_tokenizer_sizes_the_model(text):
    with (Path(text.record) / "fitted" / "preprocessors" / "tokenizer.pkl").open("rb") as stream:
        tokenizer = pickle.load(stream)["text"]
    size = tokenizer.size
    last = load(Path(text.record) / "checkpoints" / "last.pt")["models"]["lm"]
    assert last["nodes.s0.weight"].shape == (size, 8)
    assert last["nodes.s2.weight"].shape[0] == size
    assert not (Path(text.record) / "checkpoints" / "best.pt").exists()
    data = json.loads((Path(text.record) / "data.json").read_text())
    assert data["fit"] == {"fields": 1, "features": 1, "targets": [], "preprocessors": {"tokenizer": 1}, "extras": []}


def test_the_sample_writer_and_the_generate_step_write_text(text):
    samples = Path(text.record) / "samples"
    assert sorted(path.name for path in samples.iterdir()) == ["samples.txt", "turn_0002.txt", "turn_0004.txt"]
    generated = samples.joinpath("samples.txt").read_text()
    assert generated.startswith("ROMEO:") and len(generated) == 26


def test_generate_and_resume_continue_from_the_record(text, tmp_path):
    copy = copied(text.record, tmp_path / "text_copy")
    generated = generate(copy)
    assert generated.path == str(copy / "samples" / "samples.txt")
    assert isinstance(generated.samples, str) and generated.samples.startswith("ROMEO:")
    resumed = resume(copy, parse_sets(["training.steps.total=12"]), when="resumed")
    history = History.read(resumed.record)
    assert [line["turn"] for line in history] == [5, 6]
    assert [line["global_step"] for line in history] == [10, 12]
