"""Dataset sources: a sample store with named fields, subsets by position and per field column access."""

import numpy


class Samples:
    """A Dataset source as kalfa sees it: named fields per item, row ids, and columns readable without the items.

    ``source`` yields a mapping of field name to value per item and exposes ``fields``, ``dtypes`` and
    ``column(name)`` (the values of a light field, labels typically, for the whole source).
    """

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
        """Rows a field equality query keeps: ``field == value``, ``field != value``, ``field in [a, b]``."""
        keep = _mask(self, text)
        return self.subset(numpy.flatnonzero(keep))


def is_samples(value):
    return isinstance(value, Samples)


def _literal(text):
    from ruamel.yaml import YAML

    return YAML(typ="safe").load(text)


def _mask(samples, text):
    import re

    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|in|not in)\s*(.+?)\s*", text)
    if match is None:
        raise ValueError(f"a Dataset source takes field equality filters only (field == value, field != value, "
                         f"field in [a, b]), got {text!r}")
    field, operator, literal = match.groups()
    if field not in samples.fields:
        raise ValueError(f"filter names field {field!r}; the fields are {samples.fields}")
    value = _literal(literal)
    column = samples.column(field)
    if operator == "==":
        return column == value
    if operator == "!=":
        return column != value
    members = numpy.isin(column, list(value))
    return members if operator == "in" else ~members
