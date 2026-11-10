import json
import logging
from pathlib import Path

import joblib
import numpy
import pandas
import torch
from torch.utils.data import Dataset

from kalfa.core.aliases import register, target_path
from kalfa.core.errors import BuildError, error, warning
from kalfa.std.common import positive_int
from kalfa.std.data import flow
from kalfa.std.data.base import Loaders, config_directories, seeded_split
from kalfa.std.data.columns import name_problems, target_of
from kalfa.std.data.options import batch_of, locate, split_check, split_of


logger = logging.getLogger("kalfa.data")

TEXT_COLUMNS = ("text", "line")
TARGETS = ("targets", "input_ids")
BATCH_SIZE = 64


def stride_of(value, seq_len):
    return seq_len if value is None else positive_int(value, "stride")


def id_list(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return [int(value) for value in numpy.asarray(values).reshape(-1).tolist()]


def id_tensor(encoded):
    if isinstance(encoded, torch.Tensor):
        return encoded.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    return torch.from_numpy(numpy.array(encoded, dtype="int64").reshape(-1))


def window_count(tokens, seq_len, stride):
    return 0 if tokens < seq_len + 1 else (tokens - seq_len - 1) // stride + 1


def size_of(tokenizer):
    for name in ("size", "vocab_size"):
        value = getattr(tokenizer, name, None)
        if callable(value):
            value = value()
        if value is not None:
            return int(value)
    if hasattr(tokenizer, "__len__"):
        return len(tokenizer)
    return None


def is_tokenizer(value):
    return callable(getattr(value, "encode", None)) and callable(getattr(value, "decode", None))


def tokenizer_of(spec, lowercase=False):
    if spec is None or (isinstance(spec, str) and spec == "char"):
        return CharTokenizer(lowercase)
    if isinstance(spec, dict):
        if "use" not in spec:
            raise BuildError(f"tokenizer: a mapping needs use, got the keys {sorted(spec)}")
        params = {key: value for key, value in spec.items() if key != "use"}
        made = target_of(spec["use"], "tokenizer")(**params)
    elif isinstance(spec, (str, type)):
        target = target_of(spec, "tokenizer") if isinstance(spec, str) else spec
        chars = isinstance(target, type) and issubclass(target, CharTokenizer)
        made = target(lowercase) if chars else target()
    else:
        made = spec
    if not is_tokenizer(made):
        raise BuildError(f"tokenizer is char, a short name, an import path, a call or an object with encode and "
                         f"decode, got {type(made).__name__}")
    return made


def tokenizer_check(value):
    if isinstance(value, str) and value != "char":
        target_of(value)
    return value


class CharTokenizer:
    unknown = "<unk>"

    def __init__(self, lowercase=False):
        self.lowercase = bool(lowercase)
        self.vocab = []
        self.index = {}

    @property
    def size(self):
        return len(self.vocab)

    def prepared(self, text):
        text = str(text)
        return text.lower() if self.lowercase else text

    def adopt(self, vocab):
        self.vocab = [str(item) for item in vocab]
        self.index = {character: position for position, character in enumerate(self.vocab) if position}
        return self

    def fit(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        characters = {"\n"}
        for text in texts:
            characters.update(self.prepared(text))
        return self.adopt([self.unknown, *sorted(characters)])

    def encode(self, text):
        if not self.vocab:
            raise BuildError("the char tokenizer is not fitted; call fit(texts) or from_json(data) first")
        return [self.index.get(character, 0) for character in self.prepared(text)]

    def decode(self, ids):
        size = len(self.vocab)
        return "".join(self.vocab[position] for position in id_list(ids) if 0 < position < size)

    def to_json(self):
        return {"kind": "char", "vocab": list(self.vocab), "lowercase": self.lowercase}

    @classmethod
    def from_json(cls, data):
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise BuildError(f"a char tokenizer is read from a mapping with vocab, got {type(data).__name__}")
        kind = data.get("kind", "char")
        if kind != "char":
            raise BuildError(f"the tokenizer kind {kind!r} is not char")
        vocab = data.get("vocab")
        if not isinstance(vocab, list) or not vocab or vocab[0] != cls.unknown:
            raise BuildError(f"a char tokenizer's vocab is a list that starts with {cls.unknown!r}")
        return cls(lowercase=data.get("lowercase", False)).adopt(vocab)


class TextDataset(Dataset):
    def __init__(self, ids, seq_len, stride, short=False):
        self.ids = ids
        self.length = int(seq_len)
        count = window_count(len(ids), self.length, stride)
        if not count and short and len(ids) >= 2:
            self.length = len(ids) - 1
            count = 1
        self.starts = numpy.arange(count, dtype="int64") * int(stride)
        self.rows = numpy.arange(count, dtype="int64")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, position):
        start = int(self.starts[position])
        chunk = self.ids[start:start + self.length + 1]
        return {"input_ids": chunk[:-1], "targets": chunk[1:]}


VALIDATORS = (
    ("seq_len", lambda value: positive_int(value, "seq_len")),
    ("stride", lambda value: stride_of(value, 1)),
    ("batch", lambda value: batch_of(value, BATCH_SIZE)),
    ("tokenizer", tokenizer_check),
    ("split", split_check),
)


class TextLines(Loaders):
    inputs = ("input_ids",)
    exposes = ("vocab_size", "tokenizer", "seq_len")
    predictions = False

    @classmethod
    def check(cls, params, problems, directories=()):
        if params.get("split") is None:
            problems.append(error("split_required", "data.split is required: a list of ratios such as "
                                  "[0.9, 0.05, 0.05] or a call such as {use: sequential, ratios: [0.9, 0.1]}",
                                  "data.split"))
        file = params.get("file")
        if isinstance(file, (str, Path)) and str(file):
            try:
                locate(file, directories)
            except BuildError as failure:
                problems.append(warning("data_file_missing", str(failure), "data.file",
                                        hint="the run looks in the working directory, then next to the config"))
        elif "file" in params:
            problems.append(error("data_value", f"file is the path of a utf-8 text file, got {file!r}", "data.file"))
        for name, validator in VALIDATORS:
            if name not in params:
                continue
            try:
                validator(params[name])
            except BuildError as failure:
                problems.append(error("data_value", str(failure), f"data.{name}"))
        name_problems(params.get("split"), "data.split", problems)

    def __init__(self, file, tokenizer="char", seq_len=128, split=None, batch=BATCH_SIZE, stride=None,
                 lowercase=False):
        self.file = file
        self.seq_len = positive_int(seq_len, "seq_len")
        self.stride = stride_of(stride, self.seq_len)
        self.lowercase = bool(lowercase)
        self.tokenizer = tokenizer_of(tokenizer, self.lowercase)
        self.split = split_of(split)
        self.batch = batch_of(batch, BATCH_SIZE)
        self.target_names = ["targets"]
        self.spectator_patterns = []
        self.datasets = {}
        self.counts = {}
        self.lines = {}
        self.tokens = {}
        self.read_lines = None
        self.rng = None
        self.directories = ()
        self.loaders_by_set = None
        self.fitted = False

    @property
    def vocab_size(self):
        return size_of(self.tokenizer)

    def cleaned(self, lines):
        kept = [str(line) for line in lines if str(line).strip()]
        return [line.lower() for line in kept] if self.lowercase else kept

    def read(self, path):
        try:
            text = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as failure:
            raise BuildError(f"{path} is not utf-8 text: {failure}") from failure
        return self.cleaned(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

    def sets_of(self, ctx):
        self.directories = config_directories(ctx)
        lines = self.read(locate(self.file, self.directories))
        if not lines:
            raise BuildError(f"{self.file}: the file has no non empty lines")
        self.read_lines = len(lines)
        frame = pandas.DataFrame({"line": lines, "index": numpy.arange(len(lines))})
        parts = {name: part for name, part in seeded_split(self.split, ctx).split(frame).items() if len(part)}
        if "train" not in parts:
            raise BuildError("the split leaves no train lines")
        return {name: (part.sort_values("index") if "index" in part.columns else part)["line"].tolist()
                for name, part in parts.items()}

    def encoded(self, lines):
        return id_tensor(self.tokenizer.encode("\n".join(lines)))

    def materialize(self, sets, ctx):
        self.rng = getattr(ctx, "rng", None)
        self.datasets = {}
        self.counts = {}
        self.lines = {}
        self.tokens = {}
        for name, lines in sets.items():
            ids = self.encoded(lines)
            dataset = TextDataset(ids, self.seq_len, self.stride)
            self.lines[name] = len(lines)
            self.tokens[name] = int(len(ids))
            if not len(dataset):
                message = f"the {name} text has {len(ids)} tokens, fewer than seq_len + 1 = {self.seq_len + 1}"
                if name == "train":
                    raise BuildError(f"{message}; lower seq_len or give more text")
                logger.warning(f"{message}; the {name} set is left out")
                continue
            self.datasets[name] = dataset
            self.counts[name] = len(dataset)
        self.loaders_by_set = None
        windows = ", ".join(f"{name} {count}" for name, count in self.counts.items())
        logger.info(f"data: {windows} windows of {self.seq_len} tokens; vocabulary {self.vocab_size}")

    def fit(self, ctx):
        sets = self.sets_of(ctx)
        fitter = getattr(self.tokenizer, "fit", None)
        if callable(fitter):
            fitter(list(sets["train"]))
        self.fitted = True
        self.materialize(sets, ctx)

    def rebuild(self, ctx):
        if not self.fitted:
            raise BuildError("the data is not fitted; load(directory) first")
        self.materialize(self.sets_of(ctx), ctx)

    def lines_of(self, source):
        if isinstance(source, pandas.DataFrame):
            column = next((name for name in TEXT_COLUMNS if name in source.columns), None)
            if column is None:
                raise BuildError(f"loader_for takes a frame with a text column, got the columns "
                                 f"{list(source.columns)}")
            return self.cleaned(source[column].dropna().astype(str).tolist())
        if isinstance(source, (str, Path)):
            return self.read(locate(source, self.directories))
        if isinstance(source, (list, tuple)) and all(isinstance(item, str) for item in source):
            return self.cleaned(source)
        raise BuildError(f"loader_for takes a list of strings, a frame with a text column or a file path, got "
                         f"{type(source).__name__}")

    def loader_for(self, frame, set_name="test"):
        if not self.fitted:
            raise BuildError("the data is not fitted; call fit(ctx) or load(directory) first")
        dataset = TextDataset(self.encoded(self.lines_of(frame)), self.seq_len, self.stride, short=True)
        if not len(dataset):
            raise BuildError(f"loader_for: the {set_name} text has fewer than two tokens")
        return self.make_loader(dataset, train=False)

    def spectators(self, set_name):
        return None

    def decode(self, name, values):
        if name not in TARGETS:
            raise BuildError(f"unknown target {name!r}; the targets are {self.target_names}")
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        array = numpy.asarray(values)
        if array.dtype.kind == "f" and array.ndim >= 2 and array.shape[-1] == self.vocab_size:
            array = array.argmax(axis=-1)
        if array.ndim <= 1:
            return self.tokenizer.decode(id_list(array))
        rows = array.reshape(int(numpy.prod(array.shape[:-1])), array.shape[-1])
        return [self.tokenizer.decode(id_list(row)) for row in rows]

    def is_label(self, name):
        return name == "targets"

    def rescales(self, name):
        return False

    def inverse(self, name, values):
        return values

    def state(self):
        lowercase = self.lowercase or bool(getattr(self.tokenizer, "lowercase", False))
        common = {"lowercase": lowercase, "seq_len": self.seq_len, "stride": self.stride}
        if isinstance(self.tokenizer, CharTokenizer):
            return {**self.tokenizer.to_json(), **common}
        writer = getattr(self.tokenizer, "to_json", None)
        return {"kind": target_path(type(self.tokenizer)), "state": writer() if callable(writer) else None, **common}

    def save(self, directory):
        if not self.fitted:
            raise BuildError("the data is not fitted; there is nothing to save")
        root = Path(directory) / "fitted"
        root.mkdir(parents=True, exist_ok=True)
        state = self.state()
        if state["kind"] != "char" and state["state"] is None:
            joblib.dump(self.tokenizer, root / "tokenizer.joblib")
        (root / "tokenizer.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def restored(self, data, root):
        kind = data.get("kind", "char")
        if kind == "char":
            return CharTokenizer.from_json(data)
        current = self.tokenizer
        if data.get("state") is not None:
            source = type(current) if target_path(type(current)) == kind else target_of(kind)
            reader = getattr(source, "from_json", None)
            if not callable(reader):
                raise BuildError(f"{root}: the tokenizer {kind} has no from_json to read its saved state")
            return reader(data["state"])
        if (root / "tokenizer.joblib").exists():
            return joblib.load(root / "tokenizer.joblib")
        if target_path(type(current)) == kind:
            return current
        raise BuildError(f"{root}: the tokenizer {kind} cannot be restored")

    def load(self, directory):
        root = Path(directory) / "fitted"
        try:
            data = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        except FileNotFoundError as failure:
            raise BuildError(f"{root} lacks tokenizer.json; save(directory) writes the fitted files") from failure
        self.tokenizer = self.restored(data, root)
        self.seq_len = positive_int(data.get("seq_len", self.seq_len), "seq_len")
        self.stride = stride_of(data.get("stride"), self.seq_len)
        self.lowercase = bool(data.get("lowercase", self.lowercase))
        self.datasets = {}
        self.counts = {}
        self.lines = {}
        self.tokens = {}
        self.read_lines = None
        self.loaders_by_set = None
        self.fitted = True
        return self

    def report(self):
        kind = "char" if isinstance(self.tokenizer, CharTokenizer) else target_path(type(self.tokenizer))
        return {"rows": dict(self.counts), "lines": dict(self.lines), "tokens": dict(self.tokens),
                "vocab_size": self.vocab_size, "seq_len": self.seq_len, "stride": self.stride, "tokenizer": kind}

    def set_text(self, name):
        if name not in self.lines:
            return ""
        return f"{flow.amount(self.lines[name])} lines, {flow.amount(self.tokens.get(name))} tokens"

    def pipeline(self):
        graph = flow.Flow()
        lines = None if self.read_lines is None else f"{flow.amount(self.read_lines)} lines"
        source = flow.source_node(graph, self.file, self.read_lines, [lines])
        split = flow.split_node(graph, source, self.split)
        details = {name: ", ".join(part for part in (self.set_text(name), f"{flow.amount(count)} windows") if part)
                   for name, count in self.counts.items()}
        made = flow.set_nodes(graph, split, self.counts, details, flow.shares_of(self.split))
        for name in self.lines:
            if name not in self.counts:
                graph.step(split, f"drop:{name}", "drop", f"{name} left out",
                           f"{self.set_text(name)}, fewer than seq_len + 1 = {self.seq_len + 1}")
        kind = self.report()["tokenizer"]
        lowercase = self.lowercase or bool(getattr(self.tokenizer, "lowercase", False))
        tokenizer = graph.node("tokenizer", "tokenizer", "tokenizer", kind + (", lowercase" if lowercase else ""),
                               self.vocab_size)
        for name, key in made.items():
            graph.edge(key, tokenizer, "fit" if name == "train" else "")
        if not made:
            graph.edge(split, tokenizer)
        windows = graph.step(tokenizer, "windows", "windows", "windows",
                             f"seq_len {self.seq_len}, stride {self.stride}",
                             sum(self.counts.values()) if self.counts else None)
        tensors = [graph.step(windows, "tensor:input_ids", "tensor", "input_ids", f"int64, {self.seq_len} token ids"),
                   graph.step(windows, "tensor:targets", "tensor", "targets",
                              f"int64, the next {self.seq_len} token ids")]
        flow.batch_node(graph, tensors, self.batch)
        return graph.result()


register("text_lines", "kalfa.std.data.text.TextLines", slot="data",
         doc="a text file read line by line: a tokenizer fitted on the train lines, sequences of seq_len tokens; the "
             "batch carries input_ids and targets (the next tokens)")
register("char_tokenizer", "kalfa.std.data.text.CharTokenizer", slot="value",
         doc="one token per character, vocabulary from the train lines")
