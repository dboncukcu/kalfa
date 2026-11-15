import json
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

import kalfa
from kalfa.core import aliases
from kalfa.core import run as running
from kalfa.core.errors import BuildError
from kalfa.core.rng import Rng
from kalfa.records.reader import read_lines
from kalfa.std.data.text import CharTokenizer, TextDataset, TextLines


LINES = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "a stitch in time",
    "saves nine",
    "all that glitters",
    "is not gold",
    "the early bird",
    "catches the worm",
    "actions speak louder",
    "than words",
    "practice makes perfect",
    "time flies like an arrow",
    "fruit flies like a banana",
    "better late than never",
    "two heads are better",
    "than one",
    "when in rome",
    "do as the romans do",
    "Zebra Quartz!",
    "Xylophone 42?",
]

SEQUENTIAL = {"use": "sequential", "ratios": [0.6, 0.2, 0.2]}

MODEL_SOURCE = '''from torch import nn


class TinyLm(nn.Module):
    def __init__(self, vocab, width=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids)).permute(0, 2, 1)
'''


class ByteTokenizer:
    def __init__(self, offset=1):
        self.offset = offset
        self.fitted_on = None

    @property
    def size(self):
        return 256 + self.offset

    def fit(self, texts):
        self.fitted_on = len(texts)

    def encode(self, text):
        return [byte + self.offset for byte in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(int(value) - self.offset for value in ids if int(value) >= self.offset).decode("utf-8")

    def to_json(self):
        return {"offset": self.offset}

    @classmethod
    def from_json(cls, data):
        return cls(offset=data["offset"])


class PlainBytes:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(value) for value in ids).decode("utf-8")


def make_ctx(directory, seed=7):
    return SimpleNamespace(seed=seed, rng=Rng(seed), config=SimpleNamespace(directories=[str(directory)]))


def joined(lines):
    return "\n".join(lines)


def expected_windows(text, seq_len, stride):
    return 0 if len(text) < seq_len + 1 else (len(text) - seq_len - 1) // stride + 1


@pytest.fixture
def corpus(tmp_path):
    body = []
    for position, line in enumerate(LINES):
        body.append(line)
        if position in (3, 9):
            body.append("   " if position == 9 else "")
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def fitted(path, ctx=None, **overrides):
    params = {"seq_len": 8, "split": SEQUENTIAL, "batch": 4, **overrides}
    data = TextLines(path.name if ctx is None else str(path), **params)
    data.fit(make_ctx(path.parent) if ctx is None else ctx)
    return data


def test_char_tokenizer_round_trip_and_unknown_characters():
    tokenizer = CharTokenizer().fit(["hello world", "abc"])
    assert tokenizer.vocab[0] == "<unk>" and "\n" in tokenizer.vocab
    assert tokenizer.vocab[1:] == sorted(set("hello worldabc\n"))
    assert tokenizer.size == len(tokenizer.vocab)
    ids = tokenizer.encode("hello\nabc")
    assert all(isinstance(value, int) for value in ids) and 0 not in ids
    assert tokenizer.decode(ids) == "hello\nabc"
    assert tokenizer.decode(torch.tensor(ids)) == "hello\nabc"
    unknown = tokenizer.encode("hey!")
    assert unknown[:2] == tokenizer.encode("he") and unknown[2:] == [0, 0]
    assert tokenizer.decode(unknown) == "he" and tokenizer.decode([0, 0]) == ""
    data = json.loads(json.dumps(tokenizer.to_json()))
    assert data == {"kind": "char", "vocab": tokenizer.vocab, "lowercase": False}
    again = CharTokenizer.from_json(data)
    assert again.vocab == tokenizer.vocab and again.encode("hey!") == unknown
    assert CharTokenizer.from_json(json.dumps(data)).index == tokenizer.index


def test_char_tokenizer_lowercase_and_errors():
    lower = CharTokenizer(lowercase=True).fit(["Hello"])
    assert "H" not in lower.vocab and lower.encode("HELLO") == lower.encode("hello")
    assert CharTokenizer.from_json(lower.to_json()).lowercase is True
    with pytest.raises(BuildError, match="not fitted"):
        CharTokenizer().encode("x")
    with pytest.raises(BuildError, match="starts with"):
        CharTokenizer.from_json({"vocab": ["a", "b"]})
    with pytest.raises(BuildError, match="not char"):
        CharTokenizer.from_json({"kind": "bpe", "vocab": ["<unk>"]})


def test_sets_follow_the_split_and_the_window_counts(corpus):
    data = fitted(corpus)
    texts = {"train": joined(LINES[:12]), "valid": joined(LINES[12:16]), "test": joined(LINES[16:])}
    assert data.lines == {"train": 12, "valid": 4, "test": 4}
    assert data.tokens == {name: len(text) for name, text in texts.items()}
    assert data.counts == {name: expected_windows(text, 8, 8) for name, text in texts.items()}
    assert data.counts == {"train": 26, "valid": 9, "test": 7}
    for name, count in data.counts.items():
        assert len(data.datasets[name]) == count
        assert data.rows(name).tolist() == list(range(count))
    strided = fitted(corpus, stride=4)
    assert strided.stride == 4
    assert strided.counts == {name: expected_windows(text, 8, 4) for name, text in texts.items()}
    assert strided.datasets["train"].starts.tolist()[:3] == [0, 4, 8]
    report = data.report()
    assert report["rows"] == data.counts and report["lines"] == data.lines and report["tokens"] == data.tokens
    assert report["vocab_size"] == data.vocab_size and report["seq_len"] == 8 and report["tokenizer"] == "char"
    with pytest.raises(BuildError, match="no 'other' set"):
        data.rows("other")


def test_items_shift_input_ids_by_one_token(corpus):
    data = fitted(corpus)
    text = joined(LINES[:12])
    item = data.datasets["train"][1]
    assert set(item) == {"input_ids", "targets"}
    for name in ("input_ids", "targets"):
        assert item[name].dtype == torch.int64 and tuple(item[name].shape) == (8,)
    assert torch.equal(item["input_ids"][1:], item["targets"][:-1])
    assert data.decode("input_ids", item["input_ids"]) == text[8:16]
    assert data.decode("targets", item["targets"]) == text[9:17]
    last = data.datasets["train"][-1]
    assert data.decode("targets", last["targets"]) == text[25 * 8 + 1:25 * 8 + 9]


def test_vocab_size_exposes_and_protocol(corpus):
    data = fitted(corpus)
    train = joined(LINES[:12])
    assert data.vocab_size == len(set(train) | {"\n"}) + 1
    assert isinstance(data.tokenizer, CharTokenizer) and data.tokenizer.vocab[0] == "<unk>"
    assert data.seq_len == 8 and data.exposes == ("vocab_size", "tokenizer", "seq_len")
    assert data.inputs == ("input_ids",) and data.target_names == ["targets"] and data.predictions is False
    assert data.is_label("targets") and not data.is_label("input_ids")
    assert data.rescales("targets") is False and data.spectators("test") is None
    values = torch.arange(3)
    assert data.inverse("targets", values) is values
    test_ids = data.datasets["test"].ids
    unknown = sum(1 for character in joined(LINES[16:]) if character not in data.tokenizer.index)
    assert unknown > 0 and int((test_ids == 0).sum()) == unknown


def test_decode_ids_rows_and_logits(corpus):
    data = fitted(corpus)
    ids = torch.tensor(data.tokenizer.encode("the fox"))
    assert data.decode("targets", ids) == "the fox"
    assert data.decode("targets", ids.numpy().tolist()) == "the fox"
    rows = torch.stack([ids, ids.flip(0)])
    assert data.decode("targets", rows) == ["the fox", "xof eht"]
    logits = torch.nn.functional.one_hot(rows, data.vocab_size).float()
    assert data.decode("targets", logits) == ["the fox", "xof eht"]
    with pytest.raises(BuildError, match="unknown target"):
        data.decode("label", ids)


def test_loaders_shuffle_the_train_set_through_the_loader_stream(corpus):
    data = fitted(corpus)
    loaders = data.loaders()
    assert set(loaders) == {"train", "valid", "test"}
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert loaders["train"].generator is not None
    assert isinstance(loaders["valid"].sampler, SequentialSampler)
    batch = next(iter(loaders["train"]))
    assert tuple(batch["input_ids"].shape) == (4, 8) and batch["targets"].dtype == torch.int64
    again = fitted(corpus)
    assert torch.equal(next(iter(again.loaders()["train"]))["input_ids"], batch["input_ids"])
    first = next(iter(loaders["valid"]))
    assert torch.equal(first["input_ids"][0], data.datasets["valid"][0]["input_ids"])
    with pytest.raises(BuildError, match="not fitted"):
        TextLines(str(corpus), split=[0.8, 0.2]).loaders()


def test_random_split_is_seeded_and_covers_every_line(corpus):
    data = fitted(corpus, split=[0.7, 0.15, 0.15])
    assert data.split.seed == 7
    assert data.lines == {"train": 14, "valid": 3, "test": 3}
    again = fitted(corpus, split=[0.7, 0.15, 0.15])
    for name in data.datasets:
        assert torch.equal(again.datasets[name].ids, data.datasets[name].ids)
    other = fitted(corpus, ctx=make_ctx(corpus.parent, seed=8), split=[0.7, 0.15, 0.15])
    assert not all(torch.equal(other.datasets[name].ids, data.datasets[name].ids) for name in data.datasets)


def test_lowercase_lines_and_short_sets(corpus, caplog):
    lower = fitted(corpus, lowercase=True)
    assert lower.tokenizer.lowercase is True
    assert all(character == character.lower() for character in lower.tokenizer.vocab[1:])
    assert lower.decode("targets", lower.datasets["test"].ids).startswith("when in rome\ndo as")
    with caplog.at_level("WARNING", logger="kalfa.data"):
        short = fitted(corpus, seq_len=64)
    assert short.counts == {"train": 3, "valid": 1}
    assert "test set is left out" in caplog.text
    with pytest.raises(BuildError, match="fewer than seq_len"):
        fitted(corpus, seq_len=500)


def test_save_load_and_loader_for_new_strings(corpus, tmp_path):
    data = fitted(corpus)
    data.save(tmp_path / "run")
    saved = json.loads((tmp_path / "run" / "fitted" / "tokenizer.json").read_text(encoding="utf-8"))
    assert saved == {"kind": "char", "vocab": data.tokenizer.vocab, "lowercase": False, "seq_len": 8, "stride": 8}
    loaded = TextLines(str(corpus), seq_len=32, split=SEQUENTIAL, batch=4).load(tmp_path / "run")
    assert loaded.seq_len == 8 and loaded.stride == 8 and loaded.vocab_size == data.vocab_size
    assert loaded.tokenizer.vocab == data.tokenizer.vocab
    strings = ["the lazy fox jumps", "over the brown dog"]
    text = joined(strings)
    loader = loaded.loader_for(strings)
    assert len(loader.dataset) == expected_windows(text, 8, 8) == 4
    assert isinstance(loader.sampler, SequentialSampler)
    batch = next(iter(loader))
    assert tuple(batch["input_ids"].shape) == (4, 8) and batch["input_ids"].dtype == torch.int64
    assert loaded.decode("input_ids", batch["input_ids"][0]) == text[:8]
    assert loaded.decode("targets", batch["targets"][1]) == text[9:17]
    from_frame = next(iter(loaded.loader_for(pandas.DataFrame({"text": strings}))))
    assert torch.equal(from_frame["input_ids"], batch["input_ids"])
    other = tmp_path / "new.txt"
    other.write_text("\n".join(strings) + "\n\n", encoding="utf-8")
    assert torch.equal(next(iter(loaded.loader_for(other)))["targets"], batch["targets"])
    assert torch.equal(next(iter(loaded.loader_for(str(other))))["targets"], batch["targets"])
    short = next(iter(loaded.loader_for(["hi there"])))
    assert tuple(short["input_ids"].shape) == (1, 7)
    assert loaded.decode("targets", short["targets"][0]) == "i there"
    with pytest.raises(BuildError, match="fewer than two tokens"):
        loaded.loader_for(["x"])
    with pytest.raises(BuildError, match="text column"):
        loaded.loader_for(pandas.DataFrame({"words": strings}))
    with pytest.raises(BuildError, match="list of strings"):
        loaded.loader_for(42)
    with pytest.raises(BuildError, match="not fitted"):
        TextLines(str(corpus), split=SEQUENTIAL).loader_for(strings)
    with pytest.raises(BuildError, match="lacks tokenizer.json"):
        TextLines(str(corpus), split=SEQUENTIAL).load(tmp_path / "nothing")


def test_rebuild_reproduces_the_sets(corpus, tmp_path):
    data = fitted(corpus, split=[0.7, 0.15, 0.15])
    data.save(tmp_path / "run")
    fresh = TextLines(corpus.name, seq_len=8, split=[0.7, 0.15, 0.15], batch=4)
    with pytest.raises(BuildError, match="not fitted"):
        fresh.rebuild(make_ctx(corpus.parent))
    fresh.load(tmp_path / "run")
    with pytest.raises(BuildError, match="not fitted"):
        fresh.loaders()
    fresh.rebuild(make_ctx(corpus.parent))
    assert fresh.counts == data.counts and fresh.tokens == data.tokens
    for name in data.datasets:
        assert torch.equal(fresh.datasets[name].ids, data.datasets[name].ids)
    assert set(fresh.loaders()) == set(data.loaders())


def test_tokenizer_forms(corpus, tmp_path):
    for spec in ("char", "char_tokenizer", "kalfa.std.data.text.CharTokenizer", CharTokenizer):
        assert isinstance(TextLines(str(corpus), tokenizer=spec, split=SEQUENTIAL).tokenizer, CharTokenizer)
    called = TextLines(str(corpus), tokenizer={"use": "char_tokenizer", "lowercase": True}, split=SEQUENTIAL)
    assert called.tokenizer.lowercase is True
    with pytest.raises(BuildError, match="encode and decode"):
        TextLines(str(corpus), tokenizer=42, split=SEQUENTIAL)
    custom = ByteTokenizer()
    data = fitted(corpus, tokenizer=custom)
    assert data.tokenizer is custom and custom.fitted_on == 12 and data.vocab_size == 257
    assert data.decode("targets", data.datasets["train"][0]["targets"]) == joined(LINES[:12])[1:9]
    data.save(tmp_path / "bytes")
    saved = json.loads((tmp_path / "bytes" / "fitted" / "tokenizer.json").read_text(encoding="utf-8"))
    assert saved["state"] == {"offset": 1} and saved["kind"].endswith("ByteTokenizer")
    restored = TextLines(str(corpus), tokenizer=ByteTokenizer(offset=5), split=SEQUENTIAL).load(tmp_path / "bytes")
    assert isinstance(restored.tokenizer, ByteTokenizer) and restored.tokenizer.offset == 1
    plain = fitted(corpus, tokenizer=PlainBytes())
    assert plain.vocab_size == 256 and plain.report()["tokenizer"].endswith("PlainBytes")
    plain.save(tmp_path / "plain")
    assert (tmp_path / "plain" / "fitted" / "tokenizer.joblib").exists()
    back = TextLines(str(corpus), tokenizer=PlainBytes(), split=SEQUENTIAL).load(tmp_path / "plain")
    assert back.decode("targets", back.tokenizer.encode("fox")) == "fox"


def test_constructor_rejects_bad_values(corpus):
    with pytest.raises(BuildError, match="seq_len must be a positive integer"):
        TextLines(str(corpus), seq_len=0, split=SEQUENTIAL)
    with pytest.raises(BuildError, match="stride must be a positive integer"):
        TextLines(str(corpus), stride=0, split=SEQUENTIAL)
    with pytest.raises(BuildError, match="data.split is required"):
        TextLines(str(corpus))
    with pytest.raises(BuildError, match="batch"):
        TextLines(str(corpus), split=SEQUENTIAL, batch={"sizes": 3})
    with pytest.raises(BuildError, match="not found"):
        TextLines("absent.txt", split=SEQUENTIAL).fit(make_ctx(corpus.parent))
    blank = corpus.parent / "blank.txt"
    blank.write_text("\n  \n\n", encoding="utf-8")
    with pytest.raises(BuildError, match="no non empty lines"):
        TextLines(str(blank), split=SEQUENTIAL).fit(make_ctx(corpus.parent))


def test_batch_mapping_without_size_takes_the_text_default_and_ratios_are_checked(corpus):
    assert TextLines(str(corpus), split=SEQUENTIAL, batch={"workers": 0}).batch["size"] == 64
    assert TextLines(str(corpus), split=SEQUENTIAL).batch["size"] == 64
    problems = []
    TextLines.check({"file": corpus.name, "split": [0.5, 0.5, 0.5], "seq_len": 8}, problems, [str(corpus.parent)])
    assert [(problem.code, problem.path) for problem in problems] == [("data_value", "data.split")]
    assert "sum to 1" in problems[0].message


def test_check_reports_problems(corpus):
    problems = []
    params = {"file": "missing.txt", "seq_len": 0, "stride": -1, "batch": "big", "tokenizer": "no_such_tokenizer"}
    TextLines.check(params, problems, [str(corpus.parent)])
    found = {(problem.code, problem.path): problem.severity for problem in problems}
    assert found == {("split_required", "data.split"): "error", ("data_file_missing", "data.file"): "warning",
                     ("data_value", "data.seq_len"): "error", ("data_value", "data.stride"): "error",
                     ("data_value", "data.batch"): "error", ("data_value", "data.tokenizer"): "error"}
    numeric = []
    TextLines.check({"file": 3, "split": [0.8, 0.2]}, numeric)
    assert [(problem.code, problem.path) for problem in numeric] == [("data_value", "data.file")]
    good = []
    TextLines.check({"file": corpus.name, "split": [0.8, 0.2], "seq_len": 8, "stride": 4, "batch": 16,
                     "tokenizer": "char"}, good, [str(corpus.parent)])
    assert good == []


def test_aliases_name_the_classes():
    assert aliases.resolve("text_lines") == "kalfa.std.data.text.TextLines"
    assert aliases.resolve("char_tokenizer") == "kalfa.std.data.text.CharTokenizer"
    assert isinstance(TextDataset(torch.arange(20), 8, 8), TextDataset)


def test_a_char_model_trains_and_reopens(corpus, tmp_path, monkeypatch):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "kalfa_text_tinylm.py").write_text(MODEL_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(modules))
    record = tmp_path / "run"
    result = kalfa.Run(
        seed=7,
        device="cpu",
        data=kalfa.call("text_lines", file=str(corpus), seq_len=8, split=[0.6, 0.2, 0.2], batch=8),
        models={"lm": kalfa.call("kalfa_text_tinylm.TinyLm", vocab="@data.vocab_size", width=16)},
        losses={"ce": "cross_entropy"},
        optimizers={"main": kalfa.call("adam", lr=0.01, over="@models.lm", minimizes="@losses.ce")},
        train=kalfa.Train(epochs=2),
        record=str(record),
    ).start()
    history = read_lines(Path(result.record) / "history.jsonl")
    assert [line["turn"] for line in history] == [1, 2]
    assert "train/ce" in history[-1] and "val/ce" in history[-1]
    assert (record / "fitted" / "tokenizer.json").exists()
    assert not (record / "predictions.parquet").exists()
    trained = result.context.data
    opened = running.open_record(record)
    assert opened.data.counts == trained.counts and opened.data.vocab_size == trained.vocab_size
    for name in trained.datasets:
        assert torch.equal(opened.data.datasets[name].ids, trained.datasets[name].ids)
