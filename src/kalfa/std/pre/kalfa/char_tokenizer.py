import numpy

from kalfa.std.pre.base import Tokenizer


class CharTokenizer(Tokenizer):
    dtype = "int64"

    def fit(self, values):
        chars = {"\n"}
        for text in values:
            chars.update(str(text))
        self.chars = sorted(chars)
        self.lookup = {char: position for position, char in enumerate(self.chars)}

    def encode(self, text):
        unknown = self.lookup.get(" ", 0)
        return numpy.array([self.lookup.get(char, unknown) for char in str(text)], dtype="int64")

    def apply(self, value):
        if isinstance(value, numpy.ndarray) and value.dtype != object and value.ndim == 1 and value.dtype.kind == "i":
            return value
        return self.encode(value)

    def decode(self, ids):
        return "".join(self.chars[int(position)] for position in numpy.asarray(ids).reshape(-1))

    @property
    def size(self):
        return len(self.chars)
