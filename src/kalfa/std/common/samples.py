import re

import numpy
from ruamel.yaml import YAML


class Samples:
    def __init__(self, source, positions=None):
        self.source = source
        self.positions = numpy.arange(len(source)) if positions is None else numpy.asarray(positions, dtype="int64")

    @property
    def fields(self):
        return list(self.source.fields)

    @property
    def dtypes(self):
        return dict(self.source.dtypes)

    @property
    def index(self):
        return self.positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, position):
        return self.source[int(self.positions[position])]

    def column(self, name):
        values = numpy.asarray(self.source.column(name))
        return values[self.positions]

    def subset(self, positions):
        return Samples(self.source, self.positions[numpy.asarray(positions, dtype="int64")])

    def query(self, text):
        keep = mask_of(self, text)
        return self.subset(numpy.flatnonzero(keep))


def is_samples(value):
    return isinstance(value, Samples)


def mask_of(samples, text):
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|in|not in)\s*(.+?)\s*", text)
    if match is None:
        raise ValueError(f"a Dataset source takes field equality filters only (field == value, field != value, "
                         f"field in [a, b]), got {text!r}")
    field, operator, literal = match.groups()
    if field not in samples.fields:
        raise ValueError(f"filter names field {field!r}; the fields are {samples.fields}")
    value = YAML(typ="safe").load(literal)
    column = samples.column(field)
    if operator == "==":
        return column == value
    if operator == "!=":
        return column != value
    members = numpy.isin(column, list(value))
    return members if operator == "in" else ~members
